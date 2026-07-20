"""Remaining-muscle-only distillation helpers with reference 4-DoF prosthesis PD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mujoco
import numpy as np
from loco_mujoco.core.utils.env import Box, MDPInfo

from musclemimic.distill.mapping import DistillMapping
from musclemimic.distill.obs_mask import MaskedObsSpec, apply_obs_mask
from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.prosthesis.constants import DEFAULT_DISABLED_MUSCLE_NAMES

from .osl_harness import actuator_ids_for_names


class RemainingOnlyEnvView:
    """Network-shape view: masked observation space and remaining-muscle action space."""

    def __init__(
        self,
        env,
        spec: MaskedObsSpec,
        n_remaining_muscles: int,
        *,
        append_prosthesis_state: bool = True,
    ):
        self.env = env
        self.spec = spec
        self.n_remaining_muscles = int(n_remaining_muscles)
        self.append_prosthesis_state = bool(append_prosthesis_state)
        keep = np.asarray(spec.keep_indices, dtype=np.int32)
        obs_low = np.asarray(env.info.observation_space.low, dtype=np.float32)[keep]
        obs_high = np.asarray(env.info.observation_space.high, dtype=np.float32)[keep]
        if self.append_prosthesis_state:
            obs_low = np.concatenate([obs_low, np.full(16, -np.inf, dtype=np.float32)], axis=0)
            obs_high = np.concatenate([obs_high, np.full(16, np.inf, dtype=np.float32)], axis=0)
        act_low = -np.ones(self.n_remaining_muscles, dtype=np.float32)
        act_high = np.ones(self.n_remaining_muscles, dtype=np.float32)
        self.info = MDPInfo(
            Box(obs_low, obs_high),
            Box(act_low, act_high),
            env.info.gamma,
            env.info.horizon,
            getattr(env.info, "dt", getattr(env, "dt", 0.01)),
        )
        self.mdp_info = self.info
        self.obs_container = env.obs_container

    def __getattr__(self, name):
        return getattr(self.env, name)


def prosthesis_state_features(env, mapping: DistillMapping) -> np.ndarray:
    """Return current/ref prosthesis qpos/qvel features for the 4 left-leg DoFs."""
    qpos_indices = np.asarray(mapping.prosthesis_qpos_indices, dtype=np.int32)
    qvel_indices = np.asarray(mapping.prosthesis_qvel_indices, dtype=np.int32)
    q = np.asarray(env.data.qpos[qpos_indices], dtype=np.float32)
    qd = np.asarray(env.data.qvel[qvel_indices], dtype=np.float32)
    ref_q, ref_qd = _reference_prosthesis_state(env, qpos_indices, qvel_indices)
    return np.concatenate(
        [
            q,
            qd,
            np.asarray(ref_q, dtype=np.float32),
            np.asarray(ref_qd, dtype=np.float32),
        ],
        axis=0,
    )


def make_remaining_policy_obs(
    obs: np.ndarray,
    spec: MaskedObsSpec,
    env,
    mapping: DistillMapping,
    *,
    append_prosthesis_state: bool = True,
) -> np.ndarray:
    masked_obs = apply_obs_mask(obs, spec)
    if not append_prosthesis_state:
        return masked_obs
    return np.concatenate([masked_obs, prosthesis_state_features(env, mapping)], axis=0).astype(np.float32)


def remaining_to_full_action(remaining_action: np.ndarray, mapping: DistillMapping) -> np.ndarray:
    """Embed a remaining-muscle action into the teacher MyoFullBody action order."""
    remaining = np.asarray(remaining_action, dtype=np.float32).reshape(-1)
    if remaining.shape[0] != mapping.n_remaining_muscles:
        raise ValueError(f"Expected remaining action dim {mapping.n_remaining_muscles}, got {remaining.shape[0]}")
    action = np.zeros(mapping.teacher_action_dim, dtype=np.float32)
    action[np.asarray(mapping.teacher_remaining_action_indices, dtype=np.int32)] = remaining
    return action


def teacher_action_to_remaining(teacher_action: np.ndarray, mapping: DistillMapping) -> np.ndarray:
    action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
    return action[np.asarray(mapping.teacher_remaining_action_indices, dtype=np.int32)].copy()


def _reference_prosthesis_state(env, qpos_indices: np.ndarray, qvel_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(env.data.qpos[qpos_indices], dtype=np.float64)
    qd = np.asarray(env.data.qvel[qvel_indices], dtype=np.float64)
    if getattr(env, "th", None) is None or env._additional_carry is None:
        return q, qd
    carry = env._additional_carry
    traj_state = carry.traj_state
    if not (hasattr(traj_state, "traj_no") and hasattr(traj_state, "subtraj_step_no")):
        return q, qd
    traj_data = env.th.get_current_traj_data(carry, np)
    return (
        np.asarray(traj_data.qpos[qpos_indices], dtype=np.float64),
        np.asarray(traj_data.qvel[qvel_indices], dtype=np.float64),
    )


def _prosthesis_ref_error(env, qpos_indices: np.ndarray, qvel_indices: np.ndarray) -> tuple[float, float]:
    """Return max and RMS tracking error between current and reference prosthesis state."""
    ref_q, ref_qd = _reference_prosthesis_state(env, qpos_indices, qvel_indices)
    q = np.asarray(env.data.qpos[qpos_indices], dtype=np.float64)
    qd = np.asarray(env.data.qvel[qvel_indices], dtype=np.float64)
    err = np.concatenate([q - ref_q, qd - ref_qd], axis=0)
    return float(np.max(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def _set_prosthesis_to_reference(env, qpos_indices: np.ndarray, qvel_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hard-lock prosthesis qpos/qvel to the current reference trajectory frame."""
    ref_q, ref_qd = _reference_prosthesis_state(env, qpos_indices, qvel_indices)
    env.data.qpos[qpos_indices] = ref_q
    env.data.qvel[qvel_indices] = ref_qd
    mujoco.mj_forward(env.model, env.data)
    return ref_q, ref_qd


@dataclass
class _Reference4DofHarnessBase:
    """Shared helpers for remaining-muscle execution with prosthesis-side control."""

    env: Any
    mapping: DistillMapping
    masked_obs_spec: MaskedObsSpec
    disabled_actuators: np.ndarray
    audit: Any

    @property
    def qpos_indices(self) -> np.ndarray:
        return np.asarray(self.mapping.prosthesis_qpos_indices, dtype=np.int32)

    @property
    def qvel_indices(self) -> np.ndarray:
        return np.asarray(self.mapping.prosthesis_qvel_indices, dtype=np.int32)

    def _zero_disabled_muscles(self) -> None:
        if not self.disabled_actuators.size:
            return
        self.env.data.ctrl[self.disabled_actuators] = 0.0
        actadr = np.asarray(self.env.model.actuator_actadr[self.disabled_actuators], dtype=np.int32)
        for adr in actadr:
            if int(adr) >= 0:
                self.env.data.act[int(adr)] = 0.0

    def _finalize_step(
        self,
        full_action: np.ndarray,
        carry,
        prosthesis_tau: np.ndarray,
        prosthesis_tau_abs_max: float,
        *,
        lock_reference_after_finalize: bool = False,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        cur_obs, carry = self.env._create_observation(self.env._model, self.env._data, carry)
        cur_info = self.env._info.copy()
        cur_obs, self.env._data, cur_info, carry = self.env._step_finalize(
            cur_obs, self.env._model, self.env._data, cur_info, carry
        )
        cur_info = self.env._update_info_dictionary(cur_info, cur_obs, self.env._data, carry)
        absorbing, carry = self.env._is_absorbing(cur_obs, cur_info, self.env._data, carry)
        reward, carry = self.env._reward(
            self.env._obs, full_action, cur_obs, absorbing, cur_info, self.env._model, self.env._data, carry
        )
        done = self.env._is_done(cur_obs, absorbing, cur_info, self.env._data, carry)
        carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
        self.env._additional_carry = carry
        if lock_reference_after_finalize:
            _set_prosthesis_to_reference(self.env, self.qpos_indices, self.qvel_indices)
            cur_obs, carry = self.env._create_observation(self.env._model, self.env._data, carry)
            self.env._additional_carry = carry
        self.env._obs = cur_obs

        ref_q, ref_qd = _reference_prosthesis_state(self.env, self.qpos_indices, self.qvel_indices)
        q = np.asarray(self.env.data.qpos[self.qpos_indices], dtype=np.float32)
        qd = np.asarray(self.env.data.qvel[self.qvel_indices], dtype=np.float32)
        ref_err_max, ref_err_rms = _prosthesis_ref_error(self.env, self.qpos_indices, self.qvel_indices)
        policy_obs = make_remaining_policy_obs(cur_obs, self.masked_obs_spec, self.env, self.mapping)
        diag = {
            "root_height": float(self.env.data.qpos[2]),
            "done": bool(done),
            "ref_prosthesis_qpos": ref_q.astype(np.float32),
            "ref_prosthesis_qvel": ref_qd.astype(np.float32),
            "prosthesis_qpos": q,
            "prosthesis_qvel": qd,
            "prosthesis_tau": np.asarray(prosthesis_tau, dtype=np.float32),
            "prosthesis_tau_abs_max": float(prosthesis_tau_abs_max),
            "prosthesis_ref_error_max": float(ref_err_max),
            "prosthesis_ref_error_rms": float(ref_err_rms),
            "disabled_ctrl_norm": float(np.linalg.norm(self.env.data.ctrl[self.disabled_actuators]))
            if self.disabled_actuators.size
            else 0.0,
        }
        return policy_obs, float(np.asarray(reward).item()), bool(done), diag


@dataclass
class Reference4DofPDHarness(_Reference4DofHarnessBase):
    """Execute remaining muscles while tracking reference left-leg 4-DoF with PD."""

    kp: np.ndarray
    kd: np.ndarray
    torque_limit: np.ndarray
    torque_slew: np.ndarray | None
    last_torque: np.ndarray

    def reset(self) -> np.ndarray:
        self.last_torque[:] = 0.0
        obs = self.env.reset()
        self._zero_disabled_muscles()
        return make_remaining_policy_obs(obs, self.masked_obs_spec, self.env, self.mapping)

    def step(self, remaining_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        full_action = remaining_to_full_action(remaining_action, self.mapping)
        cur_info = self.env._info.copy()
        carry = self.env._additional_carry.replace(last_action=full_action)
        processed_action, carry = self.env._preprocess_action(full_action, self.env._model, self.env._data, carry)
        self.env._model, self.env._data, carry = self.env._simulation_pre_step(self.env._model, self.env._data, carry)

        torque_sum = np.zeros(4, dtype=np.float64)
        torque_abs_max = 0.0
        for _ in range(self.env._n_intermediate_steps):
            self.env._data.qfrc_applied[:] = 0.0
            ctrl_action, carry = self.env._compute_action(processed_action, self.env._model, self.env._data, carry)
            self.env._data.ctrl[self.env._action_indices] = np.asarray(ctrl_action, dtype=np.float64).reshape(-1)
            self._zero_disabled_muscles()

            ref_q, ref_qd = _reference_prosthesis_state(self.env, self.qpos_indices, self.qvel_indices)
            q = np.asarray(self.env._data.qpos[self.qpos_indices], dtype=np.float64)
            qd = np.asarray(self.env._data.qvel[self.qvel_indices], dtype=np.float64)
            torque = self.kp * (ref_q - q) + self.kd * (ref_qd - qd)
            torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            if self.torque_slew is not None:
                delta = np.clip(torque - self.last_torque, -self.torque_slew, self.torque_slew)
                torque = self.last_torque + delta
            self.last_torque[:] = torque
            torque_sum += torque
            torque_abs_max = max(torque_abs_max, float(np.max(np.abs(torque))))
            self.env._data.qfrc_applied[self.qvel_indices] = torque
            mujoco.mj_step(self.env._model, self.env._data, self.env._n_substeps)
            self.env._data.qfrc_applied[:] = 0.0

        self.env._data, carry = self.env._simulation_post_step(self.env._model, self.env._data, carry)
        avg_tau = torque_sum / max(self.env._n_intermediate_steps, 1)
        return self._finalize_step(full_action, carry, avg_tau, torque_abs_max)


@dataclass
class Reference4DofLockedHarness(_Reference4DofHarnessBase):
    """Execute remaining muscles while hard-locking left-leg 4-DoF to reference."""

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._zero_disabled_muscles()
        _set_prosthesis_to_reference(self.env, self.qpos_indices, self.qvel_indices)
        obs, carry = self.env._create_observation(self.env._model, self.env._data, self.env._additional_carry)
        self.env._obs = obs
        self.env._additional_carry = carry
        return make_remaining_policy_obs(obs, self.masked_obs_spec, self.env, self.mapping)

    def step(self, remaining_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        full_action = remaining_to_full_action(remaining_action, self.mapping)
        carry = self.env._additional_carry.replace(last_action=full_action)
        processed_action, carry = self.env._preprocess_action(full_action, self.env._model, self.env._data, carry)
        self.env._model, self.env._data, carry = self.env._simulation_pre_step(self.env._model, self.env._data, carry)

        for _ in range(self.env._n_intermediate_steps):
            self.env._data.qfrc_applied[:] = 0.0
            ctrl_action, carry = self.env._compute_action(processed_action, self.env._model, self.env._data, carry)
            self.env._data.ctrl[self.env._action_indices] = np.asarray(ctrl_action, dtype=np.float64).reshape(-1)
            self._zero_disabled_muscles()
            _set_prosthesis_to_reference(self.env, self.qpos_indices, self.qvel_indices)
            mujoco.mj_step(self.env._model, self.env._data, self.env._n_substeps)
            _set_prosthesis_to_reference(self.env, self.qpos_indices, self.qvel_indices)
            self.env._data.qfrc_applied[:] = 0.0

        self.env._data, carry = self.env._simulation_post_step(self.env._model, self.env._data, carry)
        zero_tau = np.zeros(4, dtype=np.float64)
        return self._finalize_step(full_action, carry, zero_tau, 0.0, lock_reference_after_finalize=True)


Reference4DofHarness = Reference4DofPDHarness | Reference4DofLockedHarness


def build_reference4dof_harness(
    env,
    mapping: DistillMapping,
    masked_obs_spec: MaskedObsSpec,
    *,
    kp: tuple[float, float, float, float] = (240.0, 180.0, 60.0, 40.0),
    kd: tuple[float, float, float, float] = (24.0, 18.0, 6.0, 4.0),
    torque_limit: tuple[float, float, float, float] = (140.0, 120.0, 60.0, 60.0),
    torque_slew_limit: float | None = 35.0,
    mode: str = "pd",
) -> Reference4DofHarness:
    disabled_actuators = actuator_ids_for_names(env.model, tuple(DEFAULT_DISABLED_MUSCLE_NAMES))
    audit = audit_myofullbody_left_leg(env.model)
    common = {
        "env": env,
        "mapping": mapping,
        "masked_obs_spec": masked_obs_spec,
        "disabled_actuators": disabled_actuators,
        "audit": audit,
    }
    if mode == "lock":
        return Reference4DofLockedHarness(**common)
    if mode == "pd":
        slew = None if torque_slew_limit is None else np.full(4, float(torque_slew_limit), dtype=np.float64)
        return Reference4DofPDHarness(
            **common,
            kp=np.asarray(kp, dtype=np.float64),
            kd=np.asarray(kd, dtype=np.float64),
            torque_limit=np.asarray(torque_limit, dtype=np.float64),
            torque_slew=slew,
            last_torque=np.zeros(4, dtype=np.float64),
        )
    raise ValueError(f"Unknown reference control mode={mode!r}; expected 'pd' or 'lock'")


@dataclass
class RemainingOnlyDataset:
    dataset_dir: str | Path | list[str | Path] | tuple[str | Path, ...]
    split: str = "train"
    val_fraction: float = 0.1
    seed: int = 0
    max_files: int | None = None
    max_frames: int | None = None

    def __post_init__(self):
        dirs = [Path(p) for p in self.dataset_dir] if isinstance(self.dataset_dir, list | tuple) else [Path(self.dataset_dir)]
        files: list[Path] = []
        for directory in dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
            files.extend(sorted(directory.glob("*.npz")))
        if self.max_files is not None:
            files = files[: int(self.max_files)]
        if not files:
            raise FileNotFoundError(f"No .npz rollout files found in {', '.join(str(d) for d in dirs)}")
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(files))
        rng.shuffle(order)
        n_val = max(1, int(round(len(files) * self.val_fraction))) if len(files) > 1 else 0
        val_idx = set(order[:n_val].tolist())
        if self.split == "train":
            self.files = [f for i, f in enumerate(files) if i not in val_idx]
        elif self.split in {"val", "validation"}:
            self.files = [f for i, f in enumerate(files) if i in val_idx]
        elif self.split == "all":
            self.files = files
        else:
            raise ValueError(f"Unknown split={self.split!r}")
        if not self.files:
            self.files = files
        self._data: dict[str, np.ndarray] | None = None

    def _load_all(self) -> dict[str, np.ndarray]:
        chunks = {
            "obs": [],
            "target_remaining": [],
            "ref_prosthesis_qpos": [],
            "ref_prosthesis_qvel": [],
        }
        total = 0
        for path in self.files:
            with np.load(path, allow_pickle=True) as data:
                n = int(data["obs_student"].shape[0])
                if self.max_frames is not None:
                    remaining = int(self.max_frames) - total
                    if remaining <= 0:
                        break
                    n = min(n, remaining)
                obs_key = "obs_policy" if "obs_policy" in data.files else "obs_student"
                chunks["obs"].append(np.asarray(data[obs_key][:n], dtype=np.float32))
                chunks["target_remaining"].append(np.asarray(data["target_remaining_muscle_action"][:n], dtype=np.float32))
                chunks["ref_prosthesis_qpos"].append(np.asarray(data["ref_prosthesis_qpos"][:n], dtype=np.float32))
                chunks["ref_prosthesis_qvel"].append(np.asarray(data["ref_prosthesis_qvel"][:n], dtype=np.float32))
                total += n
        return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}

    @property
    def data(self) -> dict[str, np.ndarray]:
        if self._data is None:
            self._data = self._load_all()
        return self._data

    def __len__(self) -> int:
        return int(self.data["obs"].shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.data["obs"].shape[-1])

    @property
    def n_remaining_muscles(self) -> int:
        return int(self.data["target_remaining"].shape[-1])

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        n = len(self)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(self.seed if seed is None else seed)
            rng.shuffle(indices)
        for start in range(0, n, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            if drop_last and batch_idx.shape[0] < int(batch_size):
                continue
            yield {key: value[batch_idx] for key, value in self.data.items()}
