"""Jetson ONNX prosthesis controller harness for masked full-muscle deploy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort

from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.prosthesis.constants import DEFAULT_DISABLED_MUSCLE_NAMES

from .osl_harness import actuator_ids_for_names


def _mask_disabled_muscle_actions(teacher_action: np.ndarray, disabled_action_indices: np.ndarray) -> np.ndarray:
    action = np.asarray(teacher_action, dtype=np.float32).reshape(-1).copy()
    if disabled_action_indices.size:
        action[np.asarray(disabled_action_indices, dtype=np.int32)] = 0.0
    return action

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_JETSON_PACKAGE_DIR = WORKSPACE_ROOT / "jetson" / "src" / "my_robot_base"

MODEL_PRESETS = {
    "limited": {
        "stance_model": "Models/Ramp_Model_Limited_v1/stance/rank1_fs_w2_5layer_wd5e-05/onnx_self_contained/model.onnx",
        "swing_model": "Models/Ramp_Model_Limited_v1/swing/rank1_sw_w3_wd5e4_f4_s3/onnx_self_contained/model.onnx",
    },
    "full": {
        "stance_model": "Models/Ramp_Model_Full_v1/stance/rank1_ramp_st_freq6_sm8sg3/onnx_self_contained/model.onnx",
        "swing_model": "Models/Ramp_Model_Full_v1/swing/rank1_ramp_sw_sm0_sg2/onnx_self_contained/model.onnx",
    },
}


def _resolve_resource(package_dir: Path, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return (package_dir / path).resolve()


def _joint_qpos_qvel(model: mujoco.MjModel, joint_name: str) -> tuple[int | None, int | None]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None, None
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _collect_body_descendants(model: mujoco.MjModel, root_body_ids: np.ndarray) -> set[int]:
    roots = {int(x) for x in np.asarray(root_body_ids).reshape(-1)}
    descendants: set[int] = set()
    for body_id in range(model.nbody):
        cur = int(body_id)
        while cur >= 0:
            if cur in roots:
                descendants.add(int(body_id))
                break
            cur = int(model.body_parentid[cur])
            if cur == 0 and cur not in roots:
                break
    return descendants


def prosthesis_contact_load_n(model: mujoco.MjModel, data: mujoco.MjData, body_ids: set[int]) -> float:
    total_normal_force = 0.0
    force6 = np.zeros(6, dtype=np.float64)
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        if body1 not in body_ids and body2 not in body_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_id, force6)
        total_normal_force += max(float(force6[0]), 0.0)
    return total_normal_force


class ContactPhaseTracker:
    def __init__(self, contact_threshold_n: float, min_phase_time_s: float):
        self.contact_threshold_n = float(contact_threshold_n)
        self.min_phase_time_s = float(min_phase_time_s)
        self.phase = "STANCE"
        self.phase_start_time = 0.0

    def reset(self) -> None:
        self.phase = "STANCE"
        self.phase_start_time = 0.0

    def update(self, contact_load_n: float, sim_time_s: float) -> tuple[str, float, bool]:
        desired = "STANCE" if float(contact_load_n) >= self.contact_threshold_n else "SWING"
        changed = False
        if desired != self.phase and (sim_time_s - self.phase_start_time) >= self.min_phase_time_s:
            self.phase = desired
            self.phase_start_time = sim_time_s
            changed = True
        return self.phase, max(0.0, sim_time_s - self.phase_start_time), changed


class JetsonOnnxController:
    def __init__(
        self,
        package_dir: Path,
        controller: str,
        ramp_incline: float = 0.0,
        subject_height: float = 1.75,
        subject_weight: float = 70.0,
    ):
        if controller not in MODEL_PRESETS:
            raise ValueError(f"Unknown Jetson controller preset: {controller}")
        preset = MODEL_PRESETS[controller]
        self.controller = controller
        self.package_dir = package_dir
        self.stance_model_path = _resolve_resource(package_dir, preset["stance_model"])
        self.swing_model_path = _resolve_resource(package_dir, preset["swing_model"])
        self.raw_condition = np.asarray([[ramp_incline, subject_height, subject_weight]], dtype=np.float32)
        self.stance_session = ort.InferenceSession(str(self.stance_model_path), providers=["CPUExecutionProvider"])
        self.swing_session = ort.InferenceSession(str(self.swing_model_path), providers=["CPUExecutionProvider"])
        self.stance_input_dim = self._raw_input_dim(self.stance_session)
        self.swing_input_dim = self._raw_input_dim(self.swing_session)

    @staticmethod
    def _raw_input_dim(session: ort.InferenceSession) -> int:
        for item in session.get_inputs():
            if item.name == "raw_input":
                return int(item.shape[1])
        raise ValueError("Jetson ONNX model does not expose a raw_input input")

    def predict(
        self,
        phase: str,
        hip_sagittal_deg: float,
        time_in_phase_s: float,
        knee_angle_deg: float,
        knee_velocity_deg_s: float,
    ) -> tuple[float, float, int]:
        if phase == "STANCE":
            session = self.stance_session
            input_dim = self.stance_input_dim
            control_mode = 1
        else:
            session = self.swing_session
            input_dim = self.swing_input_dim
            control_mode = 2

        if input_dim == 2:
            raw_input = np.asarray([[hip_sagittal_deg, time_in_phase_s]], dtype=np.float32)
        elif input_dim == 4:
            raw_input = np.asarray(
                [[hip_sagittal_deg, time_in_phase_s, knee_angle_deg, knee_velocity_deg_s]],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported Jetson raw_input dimension: {input_dim}")

        predicted_phase, predicted_output = session.run(
            None,
            {"raw_input": raw_input, "raw_condition": self.raw_condition},
        )
        return float(predicted_phase[0][0]), float(predicted_output[0][0]), control_mode


@dataclass
class JetsonProsthesisHarness:
    env: Any
    audit: Any
    disabled_actuators: np.ndarray
    disabled_action_indices: np.ndarray
    controller: JetsonOnnxController
    phase_tracker: ContactPhaseTracker
    prosthesis_body_ids: set[int]
    hip_qpos_idx: int | None
    control_dt: float
    knee_torque_limit: float
    ankle_torque_limit: float
    foot_torque_limit: float
    torque_slew_limit: float
    knee_position_kp: float
    knee_position_kd: float
    knee_output_sign: float
    zero_kp: np.ndarray
    zero_kd: np.ndarray
    torque_limit: np.ndarray
    torque_slew: np.ndarray
    last_torque: np.ndarray
    knee_target: np.ndarray
    disabled_muscle_scale: float = 0.0

    def reset(self) -> None:
        self.phase_tracker.reset()
        self.last_torque[:] = 0.0
        self.knee_target[:] = np.asarray(self.env.data.qpos[self.audit.qpos_indices], dtype=np.float64)

    def step(self, teacher_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        from record_stage1_testset_visual import IndexedValues, step_with_stage1_pd

        action = _mask_disabled_muscle_actions(teacher_action, self.disabled_action_indices)
        q = np.asarray(self.env.data.qpos[self.audit.qpos_indices], dtype=np.float64)
        qd = np.asarray(self.env.data.qvel[self.audit.qvel_indices], dtype=np.float64)
        sim_time = float(self.env.data.time)
        contact_load_n = prosthesis_contact_load_n(self.env.model, self.env.data, self.prosthesis_body_ids)
        phase, time_in_phase, _changed = self.phase_tracker.update(contact_load_n, sim_time)

        hip_sagittal_deg = float(np.rad2deg(self.env.data.qpos[self.hip_qpos_idx])) if self.hip_qpos_idx is not None else 0.0
        predicted_phi, predicted_output, control_mode = self.controller.predict(
            phase,
            hip_sagittal_deg,
            time_in_phase,
            float(np.rad2deg(q[0])),
            float(np.rad2deg(qd[0])),
        )

        kp = self.zero_kp.copy()
        kd = self.zero_kd.copy()
        ff_torque = np.zeros(4, dtype=np.float64)
        target_q = q.copy()
        signed_output = float(self.knee_output_sign) * float(predicted_output)
        if control_mode == 1:
            ff_torque[0] = np.clip(signed_output, -self.knee_torque_limit, self.knee_torque_limit)
            self.knee_target[0] = q[0]
        else:
            target_q[0] = np.deg2rad(signed_output)
            self.knee_target[0] = target_q[0]
            kp[0] = float(self.knee_position_kp)
            kd[0] = float(self.knee_position_kd)

        obs, reward, _absorbing, done, _info = step_with_stage1_pd(
            self.env,
            action,
            IndexedValues(self.audit.qpos_indices, target_q),
            self.audit.qvel_indices,
            self.disabled_actuators,
            float(self.disabled_muscle_scale),
            kp,
            kd,
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
            "prosthesis_torque": self.last_torque.astype(np.float32).copy(),
            "phase": phase,
            "contact_load_n": float(contact_load_n),
            "predicted_output": float(predicted_output),
            "control_mode": int(control_mode),
        }
        return np.asarray(obs), float(np.asarray(reward).item()), bool(done), diag


def build_jetson_harness(
    env,
    *,
    controller_preset: str,
    disabled_action_indices: np.ndarray,
    package_dir: str | Path | None = None,
    control_dt: float = 0.01,
    disabled_muscle_scale: float = 0.0,
    ramp_incline: float = 0.0,
    subject_height: float = 1.75,
    subject_weight: float = 70.0,
    contact_threshold_n: float = 20.0,
    min_phase_time: float = 0.08,
    knee_torque_limit: float = 140.0,
    ankle_torque_limit: float = 0.0,
    foot_torque_limit: float = 0.0,
    torque_slew_limit: float = 35.0,
    knee_position_kp: float = 35.0,
    knee_position_kd: float = 4.0,
    knee_output_sign: float = 1.0,
) -> JetsonProsthesisHarness:
    audit = audit_myofullbody_left_leg(env.model)
    disabled_actuators = actuator_ids_for_names(env.model, tuple(DEFAULT_DISABLED_MUSCLE_NAMES))
    hip_qpos_idx, _ = _joint_qpos_qvel(env.model, "hip_flexion_l")
    root_body_ids = np.asarray([env.model.jnt_bodyid[j.joint_id] for j in audit.joints], dtype=np.int32)
    prosthesis_body_ids = _collect_body_descendants(env.model, root_body_ids)
    pkg = Path(package_dir) if package_dir is not None else DEFAULT_JETSON_PACKAGE_DIR
    controller = JetsonOnnxController(
        package_dir=pkg,
        controller=controller_preset,
        ramp_incline=ramp_incline,
        subject_height=subject_height,
        subject_weight=subject_weight,
    )
    torque_limit = np.asarray(
        [knee_torque_limit, ankle_torque_limit, foot_torque_limit, foot_torque_limit],
        dtype=np.float64,
    )
    return JetsonProsthesisHarness(
        env=env,
        audit=audit,
        disabled_actuators=disabled_actuators,
        disabled_action_indices=np.asarray(disabled_action_indices, dtype=np.int32),
        controller=controller,
        phase_tracker=ContactPhaseTracker(contact_threshold_n, min_phase_time),
        prosthesis_body_ids=prosthesis_body_ids,
        hip_qpos_idx=hip_qpos_idx,
        control_dt=float(control_dt),
        knee_torque_limit=float(knee_torque_limit),
        ankle_torque_limit=float(ankle_torque_limit),
        foot_torque_limit=float(foot_torque_limit),
        torque_slew_limit=float(torque_slew_limit),
        knee_position_kp=float(knee_position_kp),
        knee_position_kd=float(knee_position_kd),
        knee_output_sign=float(knee_output_sign),
        zero_kp=np.zeros(4, dtype=np.float64),
        zero_kd=np.zeros(4, dtype=np.float64),
        torque_limit=torque_limit,
        torque_slew=np.full(4, float(torque_slew_limit), dtype=np.float64),
        last_torque=np.zeros(4, dtype=np.float64),
        knee_target=np.zeros(4, dtype=np.float64),
        disabled_muscle_scale=float(disabled_muscle_scale),
    )
