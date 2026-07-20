"""Deploy masked full-muscle policies with pluggable left-leg prosthesis controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.prosthesis.constants import DEFAULT_DISABLED_MUSCLE_NAMES, PROSTHESIS_JOINT_NAMES
from musclemimic.prosthesis.controllers import (
    FSMImpedanceProsthesisController,
    ReferencePDProsthesisController,
)
from musclemimic.prosthesis.controllers.base import ProsthesisController
from musclemimic.prosthesis.mapping import contact_load, prosthesis_body_descendants

from .osl_harness import OSLHarness, actuator_ids_for_names, build_osl_harness


def mask_disabled_muscle_actions(
    teacher_action: np.ndarray,
    disabled_action_indices: np.ndarray,
) -> np.ndarray:
    """Zero policy outputs for the 19 prosthesis-boundary muscles before execution."""
    action = np.asarray(teacher_action, dtype=np.float32).reshape(-1).copy()
    if disabled_action_indices.size:
        action[np.asarray(disabled_action_indices, dtype=np.int32)] = 0.0
    return action


def apply_disabled_muscle_mask(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    disabled_actuators: np.ndarray,
    scale: float,
) -> None:
    """Scale disabled muscle ctrl and clear activation when fully masked."""
    if not disabled_actuators.size:
        return
    clipped = float(np.clip(scale, 0.0, 1.0))
    data.ctrl[disabled_actuators] *= clipped
    if clipped <= 0.0:
        actadr = np.asarray(model.actuator_actadr[disabled_actuators], dtype=np.int32)
        for adr in actadr:
            if int(adr) >= 0:
                data.act[int(adr)] = 0.0


def prosthesis_joint_indices(model: mujoco.MjModel) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    joint_ids: list[int] = []
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    for name in PROSTHESIS_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Missing prosthesis joint: {name}")
        joint_ids.append(int(jid))
        qpos_indices.append(int(model.jnt_qposadr[jid]))
        qvel_indices.append(int(model.jnt_dofadr[jid]))
    return tuple(joint_ids), np.asarray(qpos_indices, dtype=np.int32), np.asarray(qvel_indices, dtype=np.int32)


def build_prosthesis_controller_obs(env, *, joint_ids: tuple[int, ...], qpos_indices: np.ndarray, qvel_indices: np.ndarray) -> dict:
    ref_q = np.asarray(env.data.qpos[qpos_indices], dtype=np.float32)
    ref_qd = np.asarray(env.data.qvel[qvel_indices], dtype=np.float32)
    if getattr(env, "th", None) is not None and env._additional_carry is not None:
        carry = env._additional_carry
        traj_state = carry.traj_state
        if hasattr(traj_state, "traj_no") and hasattr(traj_state, "subtraj_step_no"):
            traj_data = env.th.get_current_traj_data(carry, np)
            ref_q = np.asarray(traj_data.qpos[qpos_indices], dtype=np.float32)
            ref_qd = np.asarray(traj_data.qvel[qvel_indices], dtype=np.float32)
    body_ids = prosthesis_body_descendants(env.model, joint_ids)
    foot_contact, grf = contact_load(env.model, env.data, body_ids)
    return {
        "q": np.asarray(env.data.qpos[qpos_indices], dtype=np.float32),
        "qd": np.asarray(env.data.qvel[qvel_indices], dtype=np.float32),
        "ref_q": ref_q,
        "ref_qd": ref_qd,
        "foot_contact": foot_contact,
        "grf": grf,
    }


def resolve_prosthesis_controller(name: str) -> ProsthesisController | str:
    key = str(name).strip().lower()
    if key in {"reference_pd", "ref_pd", "reference-pd"}:
        return ReferencePDProsthesisController()
    if key in {"fsm_impedance", "fsm-impedance", "fsm"}:
        return FSMImpedanceProsthesisController()
    if key in {"osl_fsm", "osl-fsm", "osl"}:
        return "osl_fsm"
    raise ValueError(
        f"Unknown prosthesis controller {name!r}; expected reference_pd, fsm_impedance, or osl_fsm"
    )


@dataclass
class TorqueProsthesisHarness:
    """Generic qfrc_applied prosthesis deploy on MyoFullBody (controller-agnostic)."""

    env: Any
    audit: Any
    disabled_actuators: np.ndarray
    disabled_action_indices: np.ndarray
    controller: ProsthesisController
    joint_ids: tuple[int, ...]
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    zero_kp: np.ndarray
    zero_kd: np.ndarray
    torque_limit: np.ndarray
    torque_slew: np.ndarray
    last_torque: np.ndarray
    disabled_muscle_scale: float = 0.0

    def reset(self) -> None:
        self.controller.reset(self.env)
        self.last_torque[:] = 0.0

    def step(self, teacher_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        from record_stage1_testset_visual import IndexedValues, step_with_stage1_pd

        action = mask_disabled_muscle_actions(teacher_action, self.disabled_action_indices)
        pobs = build_prosthesis_controller_obs(
            self.env,
            joint_ids=self.joint_ids,
            qpos_indices=self.qpos_indices,
            qvel_indices=self.qvel_indices,
        )
        ff_torque = np.asarray(self.controller.update(pobs), dtype=np.float64)
        current_q = np.asarray(self.env.data.qpos[self.audit.qpos_indices], dtype=np.float64)
        obs, reward, _absorbing, done, _info = step_with_stage1_pd(
            self.env,
            action,
            IndexedValues(self.audit.qpos_indices, current_q),
            self.audit.qvel_indices,
            self.disabled_actuators,
            float(self.disabled_muscle_scale),
            self.zero_kp,
            self.zero_kd,
            self.torque_limit,
            self.torque_slew,
            self.last_torque,
            ff_torque,
        )
        diag = {
            "root_height": float(self.env.data.qpos[2]),
            "disabled_ctrl_norm": float(np.linalg.norm(self.env.data.ctrl[self.disabled_actuators]))
            if self.disabled_actuators.size
            else 0.0,
            "prosthesis_torque": ff_torque.astype(np.float32),
        }
        return np.asarray(obs), float(np.asarray(reward).item()), bool(done), diag


def build_torque_prosthesis_harness(
    env,
    *,
    controller: ProsthesisController,
    disabled_action_indices: np.ndarray,
    knee_torque_limit: float = 140.0,
    ankle_torque_limit: float = 120.0,
    foot_torque_limit: float = 0.0,
    torque_slew_limit: float = 35.0,
    disabled_muscle_scale: float = 0.0,
) -> TorqueProsthesisHarness:
    audit = audit_myofullbody_left_leg(env.model)
    disabled_actuators = actuator_ids_for_names(env.model, tuple(DEFAULT_DISABLED_MUSCLE_NAMES))
    joint_ids, qpos_indices, qvel_indices = prosthesis_joint_indices(env.model)
    torque_limit = np.asarray(
        [knee_torque_limit, ankle_torque_limit, foot_torque_limit, foot_torque_limit],
        dtype=np.float64,
    )
    return TorqueProsthesisHarness(
        env=env,
        audit=audit,
        disabled_actuators=disabled_actuators,
        disabled_action_indices=np.asarray(disabled_action_indices, dtype=np.int32),
        controller=controller,
        joint_ids=joint_ids,
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        zero_kp=np.zeros(4, dtype=np.float64),
        zero_kd=np.zeros(4, dtype=np.float64),
        torque_limit=torque_limit,
        torque_slew=np.full(4, float(torque_slew_limit), dtype=np.float64),
        last_torque=np.zeros(4, dtype=np.float64),
        disabled_muscle_scale=float(disabled_muscle_scale),
    )


@dataclass
class OSLProsthesisHarness:
    """OSL FSM backend; still masks disabled muscle policy outputs before stepping."""

    harness: OSLHarness
    disabled_action_indices: np.ndarray
    disabled_muscle_scale: float = 0.0

    def reset(self) -> None:
        self.harness.reset()

    def step(self, teacher_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = mask_disabled_muscle_actions(teacher_action, self.disabled_action_indices)
        return self.harness.step(
            action,
            disabled_muscle_scale=float(self.disabled_muscle_scale),
            use_osl_torque=True,
        )


def build_deploy_harness(
    env,
    *,
    controller_name: str,
    disabled_action_indices: np.ndarray,
    disabled_muscle_scale: float = 0.0,
    knee_torque_limit: float = 140.0,
    ankle_torque_limit: float = 120.0,
    foot_torque_limit: float = 0.0,
    torque_slew_limit: float = 35.0,
    loadcell_scale: float | None = None,
    body_weight_n: float | None = None,
    jetson_package_dir: str | None = None,
    jetson_ramp_incline: float = 0.0,
    jetson_subject_height: float = 1.75,
    jetson_subject_weight: float = 70.0,
    jetson_contact_threshold_n: float = 20.0,
    jetson_min_phase_time: float = 0.08,
    jetson_knee_position_kp: float = 35.0,
    jetson_knee_position_kd: float = 4.0,
    jetson_knee_output_sign: float = 1.0,
    control_dt: float = 0.01,
) -> TorqueProsthesisHarness | OSLProsthesisHarness | Any:
    key = str(controller_name).strip().lower()
    if key in {"jetson_limited", "jetson-limited"}:
        from .jetson_harness import build_jetson_harness

        return build_jetson_harness(
            env,
            controller_preset="limited",
            disabled_action_indices=disabled_action_indices,
            package_dir=jetson_package_dir,
            control_dt=control_dt,
            disabled_muscle_scale=disabled_muscle_scale,
            ramp_incline=jetson_ramp_incline,
            subject_height=jetson_subject_height,
            subject_weight=jetson_subject_weight,
            contact_threshold_n=jetson_contact_threshold_n,
            min_phase_time=jetson_min_phase_time,
            knee_torque_limit=knee_torque_limit,
            ankle_torque_limit=ankle_torque_limit,
            foot_torque_limit=foot_torque_limit,
            torque_slew_limit=torque_slew_limit,
            knee_position_kp=jetson_knee_position_kp,
            knee_position_kd=jetson_knee_position_kd,
            knee_output_sign=jetson_knee_output_sign,
        )
    if key in {"jetson_full", "jetson-full"}:
        from .jetson_harness import build_jetson_harness

        return build_jetson_harness(
            env,
            controller_preset="full",
            disabled_action_indices=disabled_action_indices,
            package_dir=jetson_package_dir,
            control_dt=control_dt,
            disabled_muscle_scale=disabled_muscle_scale,
            ramp_incline=jetson_ramp_incline,
            subject_height=jetson_subject_height,
            subject_weight=jetson_subject_weight,
            contact_threshold_n=jetson_contact_threshold_n,
            min_phase_time=jetson_min_phase_time,
            knee_torque_limit=knee_torque_limit,
            ankle_torque_limit=ankle_torque_limit,
            foot_torque_limit=foot_torque_limit,
            torque_slew_limit=torque_slew_limit,
            knee_position_kp=jetson_knee_position_kp,
            knee_position_kd=jetson_knee_position_kd,
            knee_output_sign=jetson_knee_output_sign,
        )

    resolved = resolve_prosthesis_controller(controller_name)
    if resolved == "osl_fsm":
        osl_kwargs: dict[str, float] = {
            "knee_torque_limit": knee_torque_limit,
            "ankle_torque_limit": ankle_torque_limit,
            "foot_torque_limit": foot_torque_limit,
            "torque_slew_limit": torque_slew_limit,
        }
        if loadcell_scale is not None:
            osl_kwargs["loadcell_scale"] = float(loadcell_scale)
        if body_weight_n is not None:
            osl_kwargs["body_weight_n"] = float(body_weight_n)
        return OSLProsthesisHarness(
            harness=build_osl_harness(env, **osl_kwargs),
            disabled_action_indices=np.asarray(disabled_action_indices, dtype=np.int32),
            disabled_muscle_scale=float(disabled_muscle_scale),
        )
    return build_torque_prosthesis_harness(
        env,
        controller=resolved,
        disabled_action_indices=disabled_action_indices,
        disabled_muscle_scale=disabled_muscle_scale,
        knee_torque_limit=knee_torque_limit,
        ankle_torque_limit=ankle_torque_limit,
        foot_torque_limit=foot_torque_limit,
        torque_slew_limit=torque_slew_limit,
    )
