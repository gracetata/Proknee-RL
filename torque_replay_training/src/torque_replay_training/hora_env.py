"""Torch vector adapter for HORA PPO over schema-v3 MuJoCo torque replay."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import torch

from .control import compose_prosthesis_torque, root_up_z
from .compact_schema import CompactTorqueReplayDataset


@dataclass(frozen=True)
class HoraReplayEnvConfig:
    num_envs: int = 16
    num_threads: int = 16
    episode_steps: int = 512
    random_start: bool = True
    healthy_kp: float = 4.0
    healthy_kd: float = 0.15
    healthy_pd_limit: float = 25.0
    residual_scale: float = 1.0
    exact_baseline: bool = True
    residual_limits: tuple[float, ...] = (20.0, 20.0, 8.0, 5.0)
    torque_limits: tuple[float, ...] = (260.0, 280.0, 80.0, 15.0)
    fall_height: float = 0.55
    fall_up_z: float = 0.35


@dataclass
class _ActorState:
    data: mujoco.MjData
    rng: np.random.Generator
    dataset_index: int = 0
    step_index: int = 0
    episode_count: int = 0
    previous_action: np.ndarray | None = None


class _Box:
    def __init__(self, size: int) -> None:
        self.shape = (int(size),)
        self.low = np.full(size, -1.0, dtype=np.float32)
        self.high = np.full(size, 1.0, dtype=np.float32)


def _scalar_joint_maps(metadata: dict[str, Any]) -> dict[int, int]:
    return {
        int(row["dofadr"]): int(row["qposadr"])
        for row in metadata["joint_metadata"]
        if int(row["nq"]) == 1 and int(row["nv"]) == 1
    }


def _warmstart(dataset: CompactTorqueReplayDataset, step: int) -> np.ndarray:
    return np.asarray(dataset.qacc_warmstart[step], dtype=np.float64)


class HoraTorqueReplayVecEnv:
    """HORA-compatible threaded MuJoCo vector environment.

    Physics remains native MuJoCo on CPU. Each actor owns an independent
    ``MjData`` while all actors share one immutable ``MjModel`` and one loaded
    replay dataset catalog. Policy observations and actions live on the HORA
    PyTorch device.
    """

    action_size = 4

    def __init__(
        self,
        model_path: str | Path,
        dataset_paths: Sequence[str | Path],
        config: HoraReplayEnvConfig,
        *,
        device: str,
        seed: int,
    ) -> None:
        if not dataset_paths:
            raise ValueError("at least one replay dataset is required")
        self.config = config
        self.device = torch.device(device)
        self.model_path = Path(model_path).resolve()
        self.dataset_paths = tuple(Path(path).resolve() for path in dataset_paths)
        self.datasets = tuple(
            CompactTorqueReplayDataset.load(path) for path in self.dataset_paths
        )
        self.model = mujoco.MjModel.from_binary_path(str(self.model_path))
        first = self.datasets[0]
        if self.model.nq != first.nq or self.model.nv != first.nv:
            raise ValueError("MJB and replay dimensions differ")
        joint_names = [
            str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index) or "")
            for index in range(self.model.njnt)
        ]
        for dataset in self.datasets:
            if dataset.nq != self.model.nq or dataset.nv != self.model.nv:
                raise ValueError(f"dataset/model dimensions differ: {dataset.motion_path}")
            if list(dataset.metadata["joint_names"]) != joint_names:
                raise ValueError(f"dataset/model joint order differs: {dataset.motion_path}")
            if not np.isclose(
                dataset.dt_physics,
                self.model.opt.timestep,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"dataset/model timestep differs: {dataset.motion_path}")
        if np.max(np.abs(self.model.actuator_gainprm)) > 1e-12:
            raise ValueError("replay MJB still has active actuator gains")
        if np.max(np.abs(self.model.actuator_biasprm)) > 1e-12:
            raise ValueError("replay MJB still has active actuator biases")

        metadata = first.metadata
        self.prosthesis_dofs = np.asarray(
            metadata["prosthesis_dof_indices"], dtype=np.int32
        )
        self.prosthesis_qpos = np.asarray(
            metadata["prosthesis_qpos_indices"], dtype=np.int32
        )
        dof_to_qpos = _scalar_joint_maps(metadata)
        prosthesis = {int(value) for value in self.prosthesis_dofs}
        self.healthy_dofs = np.asarray(
            [value for value in range(self.model.nv) if value not in prosthesis],
            dtype=np.int32,
        )
        self.healthy_scalar_dofs = np.asarray(
            [value for value in self.healthy_dofs if int(value) in dof_to_qpos],
            dtype=np.int32,
        )
        self.healthy_qpos = np.asarray(
            [dof_to_qpos[int(value)] for value in self.healthy_scalar_dofs],
            dtype=np.int32,
        )
        self.residual_limits = np.asarray(config.residual_limits, dtype=np.float64)
        self.torque_limits = np.asarray(config.torque_limits, dtype=np.float64)
        if self.residual_limits.shape != (4,) or self.torque_limits.shape != (4,):
            raise ValueError("four residual and torque limits are required")
        self.num_envs = int(config.num_envs)
        self.num_threads = min(int(config.num_threads), self.num_envs)
        if self.num_envs <= 0 or self.num_threads <= 0:
            raise ValueError("num_envs and num_threads must be positive")
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(self.num_envs)
        self.actors = [
            _ActorState(
                data=mujoco.MjData(self.model),
                rng=np.random.default_rng(child_seed),
                previous_action=np.zeros(4, dtype=np.float64),
            )
            for child_seed in child_seeds
        ]
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_threads,
            thread_name_prefix="mujoco-replay",
        )
        self.observation_size = int(self._reset_actor(0).size)
        self.observation_space = _Box(self.observation_size)
        self.action_space = _Box(self.action_size)
        self.reset()

    def _max_start(self, dataset: CompactTorqueReplayDataset) -> int:
        length = min(max(1, self.config.episode_steps), dataset.n_steps)
        return max(0, dataset.n_steps - length)

    def _reset_actor(self, actor_index: int) -> np.ndarray:
        actor = self.actors[actor_index]
        actor.dataset_index = int(actor.rng.integers(0, len(self.datasets)))
        dataset = self.datasets[actor.dataset_index]
        maximum = self._max_start(dataset)
        actor.step_index = (
            int(actor.rng.integers(0, maximum + 1))
            if self.config.random_start and maximum > 0
            else 0
        )
        actor.episode_count = 0
        actor.previous_action[:] = 0.0
        data = actor.data
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = dataset.rollout_qpos[actor.step_index]
        data.qvel[:] = dataset.rollout_qvel[actor.step_index]
        data.time = actor.step_index * dataset.dt_control
        data.qacc_warmstart[:] = _warmstart(dataset, actor.step_index)
        data.qfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0
        if data.act.size:
            data.act[:] = 0.0
        mujoco.mj_forward(self.model, data)
        return self._observation(actor, dataset)

    def reset(self) -> torch.Tensor:
        observations = list(self.executor.map(self._reset_actor, range(self.num_envs)))
        return torch.as_tensor(
            np.stack(observations),
            dtype=torch.float32,
            device=self.device,
        )

    def _targets(
        self,
        dataset: CompactTorqueReplayDataset,
        step: int,
        substep: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = substep / dataset.n_substeps
        qpos = (
            (1.0 - alpha) * dataset.rollout_qpos[step]
            + alpha * dataset.rollout_qpos[step + 1]
        )
        qvel = (
            (1.0 - alpha) * dataset.rollout_qvel[step]
            + alpha * dataset.rollout_qvel[step + 1]
        )
        return qpos, qvel

    def _step_actor(
        self,
        actor_index: int,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        actor = self.actors[actor_index]
        dataset = self.datasets[actor.dataset_index]
        data = actor.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        residual = np.zeros(4, dtype=np.float64)
        commanded = np.zeros(4, dtype=np.float64)
        for substep in range(dataset.n_substeps):
            data.ctrl[:] = 0.0
            if data.act.size:
                data.act[:] = 0.0
            data.qfrc_applied[:] = 0.0
            recorded = dataset.qfrc_actuator[actor.step_index, substep]
            data.qfrc_applied[self.healthy_dofs] = recorded[self.healthy_dofs]
            target_qpos, target_qvel = self._targets(
                dataset,
                actor.step_index,
                substep,
            )
            correction = (
                self.config.healthy_kp
                * (
                    target_qpos[self.healthy_qpos]
                    - data.qpos[self.healthy_qpos]
                )
                + self.config.healthy_kd
                * (
                    target_qvel[self.healthy_scalar_dofs]
                    - data.qvel[self.healthy_scalar_dofs]
                )
            )
            data.qfrc_applied[self.healthy_scalar_dofs] += np.clip(
                correction,
                -self.config.healthy_pd_limit,
                self.config.healthy_pd_limit,
            )
            commanded, residual = compose_prosthesis_torque(
                recorded[self.prosthesis_dofs],
                action,
                self.residual_limits,
                self.torque_limits,
                residual_scale=self.config.residual_scale,
                exact_baseline=self.config.exact_baseline,
            )
            data.qfrc_applied[self.prosthesis_dofs] = commanded
            mujoco.mj_step(self.model, data, 1)
        actor.step_index += 1
        actor.episode_count += 1
        metrics = self._metrics(
            actor,
            dataset,
            action,
            residual,
            float(np.linalg.norm(commanded)),
            action - actor.previous_action,
        )
        actor.previous_action[:] = action
        reward = self._reward(metrics)
        fell = (
            metrics["root_height"] < self.config.fall_height
            or metrics["root_up_z"] < self.config.fall_up_z
        )
        end_of_data = actor.step_index >= dataset.n_steps
        timeout = actor.episode_count >= self.config.episode_steps
        done = bool(fell or end_of_data or timeout)
        terminal = {
            **metrics,
            "fell": bool(fell),
            "time_out": bool(timeout and not fell and not end_of_data),
            "end_of_data": bool(end_of_data),
        }
        observation = (
            self._reset_actor(actor_index)
            if done
            else self._observation(actor, dataset)
        )
        return observation, reward, done, terminal

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        array = actions.detach().to("cpu", dtype=torch.float32).numpy()
        results = list(
            self.executor.map(
                self._step_actor,
                range(self.num_envs),
                [array[index] for index in range(self.num_envs)],
            )
        )
        observations, rewards, dones, metrics = zip(*results, strict=True)
        info: dict[str, torch.Tensor] = {}
        for name in metrics[0]:
            dtype = torch.bool if isinstance(metrics[0][name], bool) else torch.float32
            info[name] = torch.as_tensor(
                [row[name] for row in metrics],
                dtype=dtype,
                device=self.device,
            )
        info["time_outs"] = info.pop("time_out")
        return (
            torch.as_tensor(
                np.stack(observations),
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.uint8, device=self.device),
            info,
        )

    def _metrics(
        self,
        actor: _ActorState,
        dataset: CompactTorqueReplayDataset,
        action: np.ndarray,
        residual: np.ndarray,
        command_norm: float,
        action_delta: np.ndarray,
    ) -> dict[str, float]:
        reference_index = min(actor.step_index, dataset.n_steps)
        reference_qpos = dataset.rollout_qpos[reference_index]
        reference_qvel = dataset.rollout_qvel[reference_index]
        healthy_position = (
            actor.data.qpos[self.healthy_qpos]
            - reference_qpos[self.healthy_qpos]
        )
        healthy_velocity = (
            actor.data.qvel[self.healthy_scalar_dofs]
            - reference_qvel[self.healthy_scalar_dofs]
        )
        prosthesis_position = (
            actor.data.qpos[self.prosthesis_qpos]
            - reference_qpos[self.prosthesis_qpos]
        )
        prosthesis_velocity = (
            actor.data.qvel[self.prosthesis_dofs]
            - reference_qvel[self.prosthesis_dofs]
        )
        return {
            "healthy_pos_rms": float(np.sqrt(np.mean(np.square(healthy_position)))),
            "healthy_vel_rms": float(np.sqrt(np.mean(np.square(healthy_velocity)))),
            "prosthesis_pos_rms": float(
                np.sqrt(np.mean(np.square(prosthesis_position)))
            ),
            "prosthesis_vel_rms": float(
                np.sqrt(np.mean(np.square(prosthesis_velocity)))
            ),
            "root_height": float(actor.data.qpos[2]),
            "root_up_z": root_up_z(actor.data.qpos),
            "residual_norm": float(np.linalg.norm(residual)),
            "action_norm": float(np.linalg.norm(action)),
            "action_rate_norm": float(np.linalg.norm(action_delta)),
            "command_norm": command_norm,
        }

    @staticmethod
    def _reward(metrics: dict[str, float]) -> float:
        healthy = np.exp(
            -8.0 * metrics["healthy_pos_rms"] ** 2
            - 0.08 * metrics["healthy_vel_rms"] ** 2
        )
        prosthesis = np.exp(
            -10.0 * metrics["prosthesis_pos_rms"] ** 2
            - 0.10 * metrics["prosthesis_vel_rms"] ** 2
        )
        upright = np.clip((metrics["root_up_z"] + 1.0) * 0.5, 0.0, 1.0)
        height = np.exp(
            -8.0 * max(0.0, 0.75 - metrics["root_height"]) ** 2
        )
        return float(
            0.35 * healthy
            + 0.30 * prosthesis
            + 0.20 * upright
            + 0.15 * height
            - 1e-4 * metrics["residual_norm"] ** 2
            - 2e-3 * metrics["action_rate_norm"] ** 2
        )

    def _observation(
        self,
        actor: _ActorState,
        dataset: CompactTorqueReplayDataset,
    ) -> np.ndarray:
        step = min(actor.step_index, dataset.n_steps - 1)
        reference_index = min(actor.step_index, dataset.n_steps)
        phase = step / max(1, dataset.n_steps - 1)
        reference_qpos = dataset.rollout_qpos[reference_index]
        reference_qvel = dataset.rollout_qvel[reference_index]
        baseline = dataset.qfrc_actuator_mean[step, self.prosthesis_dofs]
        healthy_position = (
            actor.data.qpos[self.healthy_qpos]
            - reference_qpos[self.healthy_qpos]
        )
        healthy_velocity = (
            actor.data.qvel[self.healthy_scalar_dofs]
            - reference_qvel[self.healthy_scalar_dofs]
        )
        observation = np.concatenate(
            (
                np.asarray(
                    [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)]
                ),
                actor.data.qpos[self.prosthesis_qpos],
                actor.data.qvel[self.prosthesis_dofs],
                actor.data.qpos[self.prosthesis_qpos]
                - reference_qpos[self.prosthesis_qpos],
                actor.data.qvel[self.prosthesis_dofs]
                - reference_qvel[self.prosthesis_dofs],
                baseline / np.maximum(self.torque_limits, 1e-6),
                actor.previous_action,
                np.asarray([actor.data.qpos[2], root_up_z(actor.data.qpos)]),
                actor.data.qvel[:6],
                np.asarray(
                    [
                        np.sqrt(np.mean(np.square(healthy_position))),
                        np.sqrt(np.mean(np.square(healthy_velocity))),
                        np.max(np.abs(healthy_position)),
                        np.max(np.abs(healthy_velocity)),
                        min(float(actor.data.ncon), 20.0) / 20.0,
                    ]
                ),
            )
        )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("non-finite vector replay observation")
        return observation.astype(np.float32)

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self) -> "HoraTorqueReplayVecEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
