"""MuJoCo environment for healthy-torque replay and prosthesis residual control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import mujoco
import numpy as np

from .constants import DEFAULT_RESIDUAL_LIMITS, DEFAULT_TORQUE_LIMITS
from .control import compose_prosthesis_torque, root_up_z
from .schema import TorqueReplayDataset
from .upstream import build_fullbody_env


ReplayMode = Literal["all", "split"]


@dataclass(frozen=True)
class ReplayConfig:
    """Runtime settings. All torque values use MuJoCo generalized-force units."""

    episode_steps: int = 256
    random_start: bool = True
    healthy_kp: float = 0.0
    healthy_kd: float = 0.0
    healthy_pd_limit: float = 25.0
    residual_scale: float = 1.0
    residual_limits: tuple[float, ...] = DEFAULT_RESIDUAL_LIMITS
    torque_limits: tuple[float, ...] = DEFAULT_TORQUE_LIMITS
    fall_height: float = 0.55
    fall_up_z: float = 0.35
    replay_mode: ReplayMode = "split"
    exact_baseline: bool = False


def _scalar_joint_maps(metadata: dict[str, Any]) -> tuple[dict[int, int], dict[int, int]]:
    """Return qvel->qpos and qpos->qvel maps for scalar joints."""

    dof_to_qpos: dict[int, int] = {}
    qpos_to_dof: dict[int, int] = {}
    for row in metadata["joint_metadata"]:
        if int(row["nq"]) == 1 and int(row["nv"]) == 1:
            dof = int(row["dofadr"])
            qpos = int(row["qposadr"])
            dof_to_qpos[dof] = qpos
            qpos_to_dof[qpos] = dof
    return dof_to_qpos, qpos_to_dof


class TorqueReplayEnv:
    """Replay the healthy human and learn four left-leg residual torques.

    The original muscle/motor actuators are disabled. At each MuJoCo physics
    step, their recorded net generalized force is injected via ``qfrc_applied``.
    In ``split`` mode, non-prosthesis DOFs receive replay torque while the four
    prosthesis DOFs receive healthy baseline plus the policy residual.
    """

    action_size = 4

    def __init__(
        self,
        dataset_path: str | Path | Sequence[str | Path],
        checkpoint_path: str,
        config: ReplayConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        raw_paths = [dataset_path] if isinstance(dataset_path, (str, Path)) else list(dataset_path)
        if not raw_paths:
            raise ValueError("at least one replay dataset is required")
        self.dataset_paths = tuple(Path(path).resolve() for path in raw_paths)
        self.datasets = tuple(TorqueReplayDataset.load(path) for path in self.dataset_paths)
        self.dataset_path = self.dataset_paths[0]
        self.dataset = self.datasets[0]
        self.config = config or ReplayConfig()
        self._rng = np.random.default_rng(seed)
        self._source_env, _cfg, _state, _metadata = build_fullbody_env(
            checkpoint_path, self.dataset.motion_path
        )
        self.model = self._source_env.model
        self.data = self._source_env.data
        for candidate in self.datasets:
            self._validate_model(candidate)

        self.prosthesis_dofs = np.asarray(
            self.dataset.metadata["prosthesis_dof_indices"], dtype=np.int32
        )
        self.prosthesis_qpos = np.asarray(
            self.dataset.metadata["prosthesis_qpos_indices"], dtype=np.int32
        )
        self.root_dofs = np.asarray(self.dataset.metadata["root_dof_indices"], dtype=np.int32)
        excluded = set(self.prosthesis_dofs.tolist()) | set(self.root_dofs.tolist())
        self.healthy_dofs = np.asarray(
            [dof for dof in range(self.model.nv) if dof not in excluded], dtype=np.int32
        )
        dof_to_qpos, _ = _scalar_joint_maps(self.dataset.metadata)
        self.healthy_scalar_dofs = np.asarray(
            [dof for dof in self.healthy_dofs if int(dof) in dof_to_qpos], dtype=np.int32
        )
        self.healthy_qpos = np.asarray(
            [dof_to_qpos[int(dof)] for dof in self.healthy_scalar_dofs], dtype=np.int32
        )
        self.residual_limits = np.asarray(self.config.residual_limits, dtype=np.float64)
        self.torque_limits = np.asarray(self.config.torque_limits, dtype=np.float64)
        if self.residual_limits.shape != (4,) or self.torque_limits.shape != (4,):
            raise ValueError("residual_limits and torque_limits must each contain four values")

        self._disable_original_actuators()
        self._step_index = 0
        self._dataset_index = 0
        self._episode_start = 0
        self._episode_count = 0
        self._previous_action = np.zeros(4, dtype=np.float64)
        self._last_info: dict[str, Any] = {}
        self.reset(seed=seed)

    @property
    def observation_size(self) -> int:
        return int(self._observation().size)

    @property
    def step_index(self) -> int:
        return self._step_index

    def _validate_model(self, dataset: TorqueReplayDataset) -> None:
        expected = dataset.metadata
        if self.model.nq != dataset.nq or self.model.nv != dataset.nv:
            raise ValueError(
                f"dataset/model dimensions differ: dataset=({dataset.nq},{dataset.nv}), "
                f"model=({self.model.nq},{self.model.nv})"
            )
        current_names = [
            str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, idx) or "")
            for idx in range(self.model.njnt)
        ]
        if current_names != list(expected["joint_names"]):
            raise ValueError("dataset joint order differs from the current model")
        if not np.isclose(self.model.opt.timestep, dataset.dt_physics, rtol=0.0, atol=1e-12):
            raise ValueError("dataset physics timestep differs from the current model")

    def _disable_original_actuators(self) -> None:
        """Remove every original actuator force path to prevent double actuation."""

        self.model.actuator_gainprm[:] = 0.0
        self.model.actuator_biasprm[:] = 0.0
        self.data.ctrl[:] = 0.0
        if self.data.act.size:
            self.data.act[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        leakage = float(np.max(np.abs(self.data.qfrc_actuator))) if self.data.qfrc_actuator.size else 0.0
        if leakage > 1e-10:
            raise RuntimeError(f"original actuator disabling failed; max generalized-force leakage={leakage}")

    def _max_start(self) -> int:
        length = min(max(1, int(self.config.episode_steps)), self.dataset.n_steps)
        return max(0, self.dataset.n_steps - length)

    def reset(
        self,
        *,
        seed: int | None = None,
        start_step: int | None = None,
        dataset_index: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if dataset_index is None:
            dataset_index = int(self._rng.integers(0, len(self.datasets))) if len(self.datasets) > 1 else 0
        if not 0 <= int(dataset_index) < len(self.datasets):
            raise ValueError(f"dataset_index must be in [0,{len(self.datasets) - 1}]")
        self._dataset_index = int(dataset_index)
        self.dataset = self.datasets[self._dataset_index]
        self.dataset_path = self.dataset_paths[self._dataset_index]
        if start_step is None:
            if self.config.random_start and self._max_start() > 0:
                start_step = int(self._rng.integers(0, self._max_start() + 1))
            else:
                start_step = 0
        if not 0 <= int(start_step) < self.dataset.n_steps:
            raise ValueError(f"start_step must be in [0,{self.dataset.n_steps - 1}]")
        self._step_index = int(start_step)
        self._episode_start = int(start_step)
        self._episode_count = 0
        self._previous_action[:] = 0.0
        self.data.qpos[:] = self.dataset.rollout_qpos[self._step_index]
        self.data.qvel[:] = self.dataset.rollout_qvel[self._step_index]
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0
        if self.data.act.size:
            self.data.act[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_info = self._metrics(np.zeros(4), np.zeros(4), 0.0, np.zeros(4))
        return self._observation(), dict(self._last_info)

    def _targets(self, step: int, substep: int) -> tuple[np.ndarray, np.ndarray]:
        alpha = float(substep) / float(self.dataset.n_substeps)
        qpos = (1.0 - alpha) * self.dataset.rollout_qpos[step] + alpha * self.dataset.rollout_qpos[step + 1]
        qvel = (1.0 - alpha) * self.dataset.rollout_qvel[step] + alpha * self.dataset.rollout_qvel[step + 1]
        return qpos, qvel

    def _healthy_pd(self, target_qpos: np.ndarray, target_qvel: np.ndarray) -> np.ndarray:
        if self.config.healthy_kp == 0.0 and self.config.healthy_kd == 0.0:
            return np.zeros(self.healthy_scalar_dofs.size, dtype=np.float64)
        correction = (
            float(self.config.healthy_kp) * (target_qpos[self.healthy_qpos] - self.data.qpos[self.healthy_qpos])
            + float(self.config.healthy_kd)
            * (target_qvel[self.healthy_scalar_dofs] - self.data.qvel[self.healthy_scalar_dofs])
        )
        return np.clip(correction, -self.config.healthy_pd_limit, self.config.healthy_pd_limit)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be four finite values, got shape={action.shape}")
        action = np.clip(action, -1.0, 1.0)
        if self._step_index >= self.dataset.n_steps:
            raise RuntimeError("episode ended; call reset before step")

        previous_action = self._previous_action.copy()
        residual = np.zeros(4, dtype=np.float64)
        commanded = np.zeros(4, dtype=np.float64)
        for substep in range(self.dataset.n_substeps):
            self.data.ctrl[:] = 0.0
            if self.data.act.size:
                self.data.act[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            recorded = self.dataset.qfrc_actuator[self._step_index, substep]
            if self.config.replay_mode == "all":
                self.data.qfrc_applied[:] = recorded
                commanded = recorded[self.prosthesis_dofs].copy()
            elif self.config.replay_mode == "split":
                self.data.qfrc_applied[self.healthy_dofs] = recorded[self.healthy_dofs]
                target_qpos, target_qvel = self._targets(self._step_index, substep)
                correction = self._healthy_pd(target_qpos, target_qvel)
                self.data.qfrc_applied[self.healthy_scalar_dofs] += correction
                commanded, residual = compose_prosthesis_torque(
                    recorded[self.prosthesis_dofs],
                    action,
                    self.residual_limits,
                    self.torque_limits,
                    residual_scale=self.config.residual_scale,
                    exact_baseline=self.config.exact_baseline,
                )
                self.data.qfrc_applied[self.prosthesis_dofs] = commanded
            else:
                raise ValueError(f"unsupported replay_mode={self.config.replay_mode!r}")
            mujoco.mj_step(self.model, self.data, 1)

        self._step_index += 1
        self._episode_count += 1
        metrics = self._metrics(action, residual, float(np.linalg.norm(commanded)), action - previous_action)
        self._previous_action = action.copy()
        reward = self._reward(metrics)
        fell = metrics["root_height"] < self.config.fall_height or metrics["root_up_z"] < self.config.fall_up_z
        end_of_data = self._step_index >= self.dataset.n_steps
        timeout = self._episode_count >= int(self.config.episode_steps)
        terminated = bool(fell or end_of_data)
        truncated = bool(timeout and not terminated)
        metrics.update(
            {
                "fell": bool(fell),
                "end_of_data": bool(end_of_data),
                "timeout": bool(timeout),
                "step_index": int(self._step_index),
                "dataset_index": int(self._dataset_index),
                "motion_path": self.dataset.motion_path,
                "reward": float(reward),
            }
        )
        self._last_info = metrics
        return self._observation(), float(reward), terminated, truncated, dict(metrics)

    def _metrics(
        self,
        action: np.ndarray,
        residual: np.ndarray,
        command_norm: float,
        action_delta: np.ndarray,
    ) -> dict[str, Any]:
        ref_idx = min(self._step_index, self.dataset.n_steps)
        ref_qpos = self.dataset.rollout_qpos[ref_idx]
        ref_qvel = self.dataset.rollout_qvel[ref_idx]
        h_pos = self.data.qpos[self.healthy_qpos] - ref_qpos[self.healthy_qpos]
        h_vel = self.data.qvel[self.healthy_scalar_dofs] - ref_qvel[self.healthy_scalar_dofs]
        p_pos = self.data.qpos[self.prosthesis_qpos] - ref_qpos[self.prosthesis_qpos]
        p_vel = self.data.qvel[self.prosthesis_dofs] - ref_qvel[self.prosthesis_dofs]
        return {
            "healthy_pos_rms": float(np.sqrt(np.mean(np.square(h_pos)))) if h_pos.size else 0.0,
            "healthy_vel_rms": float(np.sqrt(np.mean(np.square(h_vel)))) if h_vel.size else 0.0,
            "prosthesis_pos_rms": float(np.sqrt(np.mean(np.square(p_pos)))),
            "prosthesis_vel_rms": float(np.sqrt(np.mean(np.square(p_vel)))),
            "root_height": float(self.data.qpos[2]),
            "root_up_z": root_up_z(self.data.qpos),
            "residual_norm": float(np.linalg.norm(residual)),
            "action_norm": float(np.linalg.norm(action)),
            "action_rate_norm": float(np.linalg.norm(action_delta)),
            "command_norm": float(command_norm),
            "contact_count": int(self.data.ncon),
        }

    @staticmethod
    def _reward(metrics: dict[str, Any]) -> float:
        healthy = np.exp(-8.0 * metrics["healthy_pos_rms"] ** 2 - 0.08 * metrics["healthy_vel_rms"] ** 2)
        prosthesis = np.exp(
            -10.0 * metrics["prosthesis_pos_rms"] ** 2 - 0.10 * metrics["prosthesis_vel_rms"] ** 2
        )
        upright = np.clip((metrics["root_up_z"] + 1.0) * 0.5, 0.0, 1.0)
        height = np.exp(-8.0 * max(0.0, 0.75 - metrics["root_height"]) ** 2)
        residual_cost = 1e-4 * metrics["residual_norm"] ** 2
        rate_cost = 2e-3 * metrics["action_rate_norm"] ** 2
        return float(
            0.35 * healthy + 0.30 * prosthesis + 0.20 * upright + 0.15 * height - residual_cost - rate_cost
        )

    def _observation(self) -> np.ndarray:
        idx = min(self._step_index, self.dataset.n_steps - 1)
        phase = float(idx) / float(max(1, self.dataset.n_steps - 1))
        ref_qpos = self.dataset.rollout_qpos[min(self._step_index, self.dataset.n_steps)]
        ref_qvel = self.dataset.rollout_qvel[min(self._step_index, self.dataset.n_steps)]
        baseline = self.dataset.qfrc_actuator_mean[idx, self.prosthesis_dofs]
        h_pos = self.data.qpos[self.healthy_qpos] - ref_qpos[self.healthy_qpos]
        h_vel = self.data.qvel[self.healthy_scalar_dofs] - ref_qvel[self.healthy_scalar_dofs]
        obs = np.concatenate(
            [
                np.asarray([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)]),
                self.data.qpos[self.prosthesis_qpos],
                self.data.qvel[self.prosthesis_dofs],
                self.data.qpos[self.prosthesis_qpos] - ref_qpos[self.prosthesis_qpos],
                self.data.qvel[self.prosthesis_dofs] - ref_qvel[self.prosthesis_dofs],
                baseline / np.maximum(self.torque_limits, 1e-6),
                self._previous_action,
                np.asarray([self.data.qpos[2], root_up_z(self.data.qpos)]),
                self.data.qvel[:6],
                np.asarray(
                    [
                        np.sqrt(np.mean(np.square(h_pos))) if h_pos.size else 0.0,
                        np.sqrt(np.mean(np.square(h_vel))) if h_vel.size else 0.0,
                        np.max(np.abs(h_pos)) if h_pos.size else 0.0,
                        np.max(np.abs(h_vel)) if h_vel.size else 0.0,
                        min(float(self.data.ncon), 20.0) / 20.0,
                    ]
                ),
            ]
        )
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("non-finite replay observation")
        return obs.astype(np.float32)

    def current_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos.copy(), self.data.qvel.copy()

    def close(self) -> None:
        self._source_env.stop()

    def __enter__(self) -> "TorqueReplayEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
