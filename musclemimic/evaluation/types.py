"""Shared types for locomotion evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MetricValue:
    value: float | int | str | list | dict | None
    unit: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value}
        if self.unit is not None:
            out["unit"] = self.unit
        if self.reason is not None:
            out["reason"] = self.reason
        return out


@dataclass
class StepResult:
    obs: Any
    reward: float
    absorbing: bool
    done: bool
    info: dict[str, Any]
    reference: dict[str, Any]


@dataclass
class RolloutBuffer:
    """Time-series rollout storage."""

    motion_path: str
    dt: float
    time: list[float] = field(default_factory=list)
    qpos: list = field(default_factory=list)
    qvel: list = field(default_factory=list)
    root_pos: list = field(default_factory=list)
    root_quat: list = field(default_factory=list)
    reference_root_pos: list = field(default_factory=list)
    reference_qpos: list = field(default_factory=list)
    site_pos: list = field(default_factory=list)
    reference_site_pos: list = field(default_factory=list)
    left_grf: list = field(default_factory=list)
    right_grf: list = field(default_factory=list)
    left_grf_raw: list = field(default_factory=list)
    right_grf_raw: list = field(default_factory=list)
    left_grf_world: list = field(default_factory=list)
    right_grf_world: list = field(default_factory=list)
    contact_left: list = field(default_factory=list)
    contact_right: list = field(default_factory=list)
    joint_torque: list = field(default_factory=list)
    prosthesis_tau: list = field(default_factory=list)
    reward_per_step: list = field(default_factory=list)
    done_flags: list = field(default_factory=list)
    info_steps: list = field(default_factory=list)
    episode_return: float = 0.0
    steps: int = 0
    traj_length: int = 0
    done_reason: str = "unknown"

    def append_step(
        self,
        *,
        t: float,
        qpos,
        qvel,
        root_pos,
        root_quat,
        ref_root_pos,
        ref_qpos,
        site_pos,
        ref_site_pos,
        left_grf,
        right_grf,
        left_grf_raw=None,
        right_grf_raw=None,
        left_grf_world,
        right_grf_world,
        contact_left: bool,
        contact_right: bool,
        joint_torque,
        prosthesis_tau,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        self.time.append(float(t))
        self.qpos.append(qpos)
        self.qvel.append(qvel)
        self.root_pos.append(root_pos)
        self.root_quat.append(root_quat)
        self.reference_root_pos.append(ref_root_pos)
        self.reference_qpos.append(ref_qpos)
        self.site_pos.append(site_pos)
        self.reference_site_pos.append(ref_site_pos)
        self.left_grf.append(left_grf)
        self.right_grf.append(right_grf)
        if left_grf_raw is not None:
            self.left_grf_raw.append(left_grf_raw)
        if right_grf_raw is not None:
            self.right_grf_raw.append(right_grf_raw)
        self.left_grf_world.append(left_grf_world)
        self.right_grf_world.append(right_grf_world)
        self.contact_left.append(bool(contact_left))
        self.contact_right.append(bool(contact_right))
        self.joint_torque.append(np.asarray(joint_torque, dtype=np.float64).copy())
        self.prosthesis_tau.append(np.asarray(prosthesis_tau, dtype=np.float64).reshape(-1).copy())
        self.reward_per_step.append(float(reward))
        self.done_flags.append(bool(done))
        self.info_steps.append(dict(info))
        self.episode_return += float(reward)
        self.steps += 1

    def to_npz_dict(self) -> dict[str, Any]:
        import numpy as np

        def stack(key: str):
            vals = getattr(self, key)
            if not vals:
                return np.asarray([])
            first = vals[0]
            if isinstance(first, (float, int, bool, np.floating, np.integer)):
                return np.asarray(vals)
            return np.stack([np.asarray(v) for v in vals], axis=0)

        return {
            "motion_path": np.asarray(self.motion_path),
            "dt": np.asarray(self.dt),
            "time": stack("time"),
            "qpos": stack("qpos"),
            "qvel": stack("qvel"),
            "root_pos": stack("root_pos"),
            "root_quat": stack("root_quat"),
            "reference_root_pos": stack("reference_root_pos"),
            "reference_qpos": stack("reference_qpos"),
            "site_pos": stack("site_pos"),
            "reference_site_pos": stack("reference_site_pos"),
            "left_GRF": stack("left_grf"),
            "right_GRF": stack("right_grf"),
            "left_GRF_raw": stack("left_grf_raw"),
            "right_GRF_raw": stack("right_grf_raw"),
            "left_GRF_world": stack("left_grf_world"),
            "right_GRF_world": stack("right_grf_world"),
            "contact_left": stack("contact_left"),
            "contact_right": stack("contact_right"),
            "joint_torque": stack("joint_torque"),
            "prosthesis_tau": stack("prosthesis_tau"),
            "reward_per_step": stack("reward_per_step"),
            "done": stack("done_flags"),
            "episode_return": np.asarray(self.episode_return),
            "steps": np.asarray(self.steps),
            "traj_length": np.asarray(self.traj_length),
            "done_reason": np.asarray(self.done_reason),
        }


@dataclass
class EvalConfig:
    config_name: str
    checkpoint_path: str
    controller_type: str
    env_type: str
    output_dir: Path
    eval_seed: int = 0
    n_steps: str | int = "auto"
    use_mujoco: bool = True
    save_video: bool = True
    save_plots: bool = True
    show_ghost: bool = True
    no_termination: bool = False
    prosthesis_control_mode: str = "eval_policy"
    prosthesis_controller: str = "reference_pd"
    force_reference_path: str | None = None
    healthy_teacher_path: str | None = None
    eval_force: dict[str, Any] = field(default_factory=dict)
    proknee_checkpoint: str | None = None
    git_commit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionResult:
    motion_path: str
    motion_type: str
    safe_name: str
    success: bool
    metrics: dict[str, Any]
    output_dir: Path
    error_message: str | None = None
