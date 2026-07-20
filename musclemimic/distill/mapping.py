"""Teacher/student mapping helpers for prosthesis distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import mujoco
import numpy as np

from musclemimic.prosthesis.mapping import prosthesis_body_descendants
from musclemimic.prosthesis.constants import PROSTHESIS_JOINT_NAMES


@dataclass(frozen=True)
class DistillMapping:
    remaining_muscle_names: tuple[str, ...]
    disabled_muscle_names: tuple[str, ...]
    prosthesis_joint_names: tuple[str, ...]
    prosthesis_motor_names: tuple[str, ...]
    teacher_remaining_action_indices: tuple[int, ...]
    teacher_disabled_action_indices: tuple[int, ...]
    student_remaining_action_indices: tuple[int, ...]
    student_prosthesis_action_indices: tuple[int, ...]
    prosthesis_qpos_indices: tuple[int, ...]
    prosthesis_qvel_indices: tuple[int, ...]
    prosthesis_dof_ids: tuple[int, ...]
    prosthesis_actuator_ids: tuple[int, ...]
    prosthesis_torque_limits: tuple[float, ...]
    prosthesis_body_ids: tuple[int, ...]
    student_action_dim: int
    teacher_action_dim: int
    student_obs_dim: int
    teacher_obs_dim: int

    @property
    def n_remaining_muscles(self) -> int:
        return len(self.remaining_muscle_names)

    @property
    def target_action_dim(self) -> int:
        return self.n_remaining_muscles + len(self.prosthesis_joint_names)

    def to_json_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> DistillMapping:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in payload.items()})


def validate_distill_dimensions(
    mapping: DistillMapping,
    *,
    obs_dim: int,
    n_remaining: int,
    n_prosthesis: int = 4,
    student_obs_dim: int | None = None,
) -> None:
    """Abort early if teacher/student muscle masking dimensions are inconsistent."""
    expected_obs = student_obs_dim if student_obs_dim is not None else mapping.student_obs_dim
    expected_target_action_dim = mapping.n_remaining_muscles + int(n_prosthesis)
    errors: list[str] = []
    if mapping.n_remaining_muscles + len(mapping.disabled_muscle_names) != mapping.teacher_action_dim:
        errors.append(
            "remaining+disabled muscles "
            f"{mapping.n_remaining_muscles}+{len(mapping.disabled_muscle_names)} != teacher_action_dim {mapping.teacher_action_dim}"
        )
    if mapping.target_action_dim != expected_target_action_dim:
        errors.append(f"target_action_dim={mapping.target_action_dim} expected {expected_target_action_dim}")
    if mapping.student_action_dim != expected_target_action_dim:
        errors.append(f"student_action_dim={mapping.student_action_dim} expected {expected_target_action_dim}")
    if int(obs_dim) != int(expected_obs):
        errors.append(f"obs_dim={obs_dim} expected {expected_obs}")
    if int(n_remaining) != mapping.n_remaining_muscles:
        errors.append(f"n_remaining={n_remaining} expected {mapping.n_remaining_muscles}")
    if len(mapping.prosthesis_joint_names) != n_prosthesis:
        errors.append(f"prosthesis joints={len(mapping.prosthesis_joint_names)} expected {n_prosthesis}")
    if errors:
        raise ValueError("Distill dimension validation failed: " + "; ".join(errors))


def spot_check_rollout_npz(path: str | Path, mapping: DistillMapping) -> None:
    with np.load(path, allow_pickle=True) as data:
        obs = np.asarray(data["obs_student"])
        rem = np.asarray(data["target_remaining_muscle_action"])
        prost = np.asarray(data["target_prosthesis_action"])
    if obs.shape[-1] != mapping.student_obs_dim:
        raise ValueError(f"{path}: obs_student dim {obs.shape[-1]} != {mapping.student_obs_dim}")
    if rem.shape[-1] != mapping.n_remaining_muscles:
        raise ValueError(f"{path}: target_remaining dim {rem.shape[-1]} != {mapping.n_remaining_muscles}")
    if prost.shape[-1] != len(mapping.prosthesis_joint_names):
        raise ValueError(f"{path}: target_prosthesis dim {prost.shape[-1]} != 4")


def rollout_npz_is_valid(path: str | Path) -> bool:
    """Return True if the rollout npz exists and can be read."""
    return rollout_npz_invalid_reason(path) is None


def rollout_npz_invalid_reason(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return "missing file"
    if path.stat().st_size == 0:
        return "empty file"
    try:
        with np.load(path, allow_pickle=True) as data:
            if "obs_student" not in data.files:
                return "missing obs_student"
            n_frames = int(np.asarray(data["obs_student"]).shape[0])
            if n_frames <= 0:
                return f"zero frames (shape={np.asarray(data['obs_student']).shape})"
    except (OSError, EOFError, ValueError, KeyError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _action_names(env) -> list[str]:
    return [env.model.actuator(int(aid)).name for aid in getattr(env, "_action_indices", range(env.model.nu))]


def _indices_for(names: tuple[str, ...], action_names: list[str], label: str) -> tuple[int, ...]:
    missing = [name for name in names if name not in action_names]
    if missing:
        raise ValueError(f"{label} names missing from action spec: {missing[:20]}")
    return tuple(int(action_names.index(name)) for name in names)


def build_distill_mapping(teacher_env, student_env) -> DistillMapping:
    student_mapping = student_env.prosthesis_mapping
    teacher_joint_ids = []
    teacher_qpos = []
    teacher_qvel = []
    for name in PROSTHESIS_JOINT_NAMES:
        jid = mujoco.mj_name2id(teacher_env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Teacher model missing prosthesis joint: {name}")
        teacher_joint_ids.append(int(jid))
        teacher_qpos.append(int(teacher_env.model.jnt_qposadr[jid]))
        teacher_qvel.append(int(teacher_env.model.jnt_dofadr[jid]))

    if student_mapping.prosthesis_joint_names != PROSTHESIS_JOINT_NAMES:
        raise ValueError("Teacher/student prosthesis joint names differ")

    teacher_actions = _action_names(teacher_env)
    student_actions = _action_names(student_env)
    remaining = tuple(student_mapping.remaining_muscle_names)
    disabled = tuple(student_mapping.disabled_muscle_names)
    motors = tuple(student_mapping.prosthesis_motor_names)

    student_remaining = _indices_for(remaining, student_actions, "student remaining muscle")
    student_motors = _indices_for(motors, student_actions, "student prosthesis motor")
    if tuple(student_actions[-len(motors):]) != motors:
        raise ValueError(f"Student prosthesis actions must be the final 4 dims, got {student_actions[-len(motors):]}")

    return DistillMapping(
        remaining_muscle_names=remaining,
        disabled_muscle_names=disabled,
        prosthesis_joint_names=tuple(student_mapping.prosthesis_joint_names),
        prosthesis_motor_names=motors,
        teacher_remaining_action_indices=_indices_for(remaining, teacher_actions, "teacher remaining muscle"),
        teacher_disabled_action_indices=_indices_for(disabled, teacher_actions, "teacher disabled muscle"),
        student_remaining_action_indices=student_remaining,
        student_prosthesis_action_indices=student_motors,
        prosthesis_qpos_indices=tuple(teacher_qpos),
        prosthesis_qvel_indices=tuple(teacher_qvel),
        prosthesis_dof_ids=tuple(teacher_qvel),
        prosthesis_actuator_ids=tuple(int(x) for x in student_mapping.prosthesis_actuator_ids),
        prosthesis_torque_limits=tuple(float(x) for x in student_mapping.prosthesis_torque_limits),
        prosthesis_body_ids=tuple(sorted(prosthesis_body_descendants(teacher_env.model, teacher_joint_ids))),
        student_action_dim=int(student_env.info.action_space.shape[0]),
        teacher_action_dim=int(teacher_env.info.action_space.shape[0]),
        student_obs_dim=int(student_env.info.observation_space.shape[0]),
        teacher_obs_dim=int(teacher_env.info.observation_space.shape[0]),
    )


def validate_qfrc_tau(model: mujoco.MjModel, data, dof_ids: tuple[int, ...]) -> np.ndarray:
    if len(dof_ids) != 4:
        raise ValueError(f"Expected 4 prosthesis dof ids, got {len(dof_ids)}")
    qfrc = np.asarray(data.qfrc_actuator)
    if qfrc.ndim != 1 or max(dof_ids) >= qfrc.shape[0]:
        raise ValueError(f"qfrc_actuator shape={qfrc.shape} cannot index dof_ids={dof_ids}")
    tau = qfrc[np.asarray(dof_ids, dtype=np.int32)].astype(np.float32)
    if not np.all(np.isfinite(tau)):
        joint_lines = []
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            joint_lines.append(f"{jid}:{name}:dof={int(model.jnt_dofadr[jid])}")
        raise ValueError(f"Non-finite qfrc_actuator tau={tau}; joints={joint_lines}")
    return tau
