#!/usr/bin/env python
"""Record masked full-muscle policy replay with the Jetson ONNX prosthesis controller."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.obs_mask import MaskedObservationEnvView, apply_obs_mask
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.prosthesis.constants import DEFAULT_DISABLED_MUSCLE_NAMES
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless
from record_masked_obs_full_muscle_replay import find_mask_spec
from record_stage1_testset_visual import IndexedValues, step_with_stage1_pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
JETSON_PACKAGE_DIR = WORKSPACE_ROOT / "jetson" / "src" / "my_robot_base"

DEFAULT_CHECKPOINT = (
    "outputs/full_muscle_masked_obs_dagger1_focused_round3_distill/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "outputs/eval_masked_dagger_compare/replay_four/dagger1_focused_round3_jetson_nn"


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


@dataclass
class JetsonStepDiagnostics:
    phase: str
    contact_load_n: float
    hip_sagittal_deg: float
    time_in_phase_s: float
    knee_angle_deg: float
    knee_velocity_deg_s: float
    predicted_phi: float
    predicted_output: float
    control_mode: int
    knee_target_rad: float
    knee_torque: float
    ankle_torque: float
    subtalar_torque: float
    mtp_torque: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--motion-path", required=True)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-steps", type=int, default=0, help="0 = full trajectory length")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--jetson-controller", choices=sorted(MODEL_PRESETS), default="limited")
    p.add_argument("--jetson-package-dir", default=str(JETSON_PACKAGE_DIR))
    p.add_argument("--ramp-incline", type=float, default=0.0)
    p.add_argument("--subject-height", type=float, default=1.75)
    p.add_argument("--subject-weight", type=float, default=70.0)
    p.add_argument("--contact-threshold-n", type=float, default=20.0)
    p.add_argument("--min-phase-time", type=float, default=0.08)
    p.add_argument("--knee-torque-limit", type=float, default=140.0)
    p.add_argument("--ankle-torque-limit", type=float, default=0.0)
    p.add_argument("--foot-torque-limit", type=float, default=0.0)
    p.add_argument("--torque-slew-limit", type=float, default=35.0)
    p.add_argument("--knee-position-kp", type=float, default=35.0)
    p.add_argument("--knee-position-kd", type=float, default=4.0)
    p.add_argument("--knee-output-sign", type=float, default=1.0)
    p.add_argument(
        "--disabled-muscle-scale",
        type=float,
        default=0.0,
        help="Scale the 19 left prosthesis-boundary muscle controls at execution; 0 = fully masked, 1 = unchanged.",
    )
    p.add_argument("--keep-raw", action="store_true")
    return p.parse_args()


def _resolve_resource(package_dir: Path, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return (package_dir / path).resolve()


def _actuator_ids_for_names(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids: list[int] = []
    missing: list[str] = []
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            missing.append(name)
        else:
            ids.append(int(aid))
    if missing:
        raise KeyError(f"Missing disabled muscle actuators: {missing}")
    return np.asarray(sorted(set(ids)), dtype=np.int32)


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


def _prosthesis_contact_load_n(model: mujoco.MjModel, data: mujoco.MjData, body_ids: set[int]) -> float:
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
    """Approximate the Jetson macro phase from simulated prosthesis contact."""

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
    """Small ROS-free adapter for the Jetson self-contained ONNX gait models."""

    def __init__(
        self,
        package_dir: Path,
        controller: str,
        ramp_incline: float,
        subject_height: float,
        subject_weight: float,
    ):
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
                shape = item.shape
                return int(shape[1])
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


def build_env_and_policy(args: argparse.Namespace):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    config, agent_state, metadata = load_checkpoint(str(checkpoint))
    OmegaConf.set_struct(config, False)

    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBody"
    env_params["headless"] = True
    env_params.pop("prosthesis", None)
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [args.motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass

    control_dt = apply_temporal_params(config)
    env = TaskFactory.get_factory_cls(config.experiment.task_factory.name).make(**{**env_params, **task_params})
    spec = find_mask_spec(checkpoint, env)
    if int(spec.action_dim) != int(env.info.action_space.shape[0]):
        raise ValueError(f"Policy action dim {spec.action_dim} != env action dim {env.info.action_space.shape[0]}")
    env_view = MaskedObservationEnvView(env, spec)
    agent_conf = PPOJax.init_agent_conf(env_view, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    runner = PolicyRunner.from_agent_state(
        agent_conf,
        agent_state,
        env_view,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    return checkpoint, config, metadata, env, spec, runner, control_dt


def main() -> int:
    args = parse_args()
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    safe_motion = args.motion_path.replace("/", "_")
    motion_dir = out_dir / "per_motion" / safe_motion
    motion_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, _config, metadata, env, spec, runner, control_dt = build_env_and_policy(args)
    audit = audit_myofullbody_left_leg(env.model)
    disabled_actuators = _actuator_ids_for_names(env.model, tuple(DEFAULT_DISABLED_MUSCLE_NAMES))
    if disabled_actuators.size != len(DEFAULT_DISABLED_MUSCLE_NAMES):
        raise RuntimeError(
            f"Expected {len(DEFAULT_DISABLED_MUSCLE_NAMES)} disabled muscles, got {disabled_actuators.size}"
        )

    package_dir = Path(args.jetson_package_dir)
    controller = JetsonOnnxController(
        package_dir=package_dir,
        controller=args.jetson_controller,
        ramp_incline=args.ramp_incline,
        subject_height=args.subject_height,
        subject_weight=args.subject_weight,
    )

    hip_qpos_idx, _hip_qvel_idx = _joint_qpos_qvel(env.model, "hip_flexion_l")
    if hip_qpos_idx is None:
        print("WARNING: hip_flexion_l not found; Jetson hip_sagittal input will be fixed at 0 deg.")

    root_body_ids = np.asarray([env.model.jnt_bodyid[j.joint_id] for j in audit.joints], dtype=np.int32)
    prosthesis_body_ids = _collect_body_descendants(env.model, root_body_ids)
    phase_tracker = ContactPhaseTracker(args.contact_threshold_n, args.min_phase_time)

    n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
    fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
    raw_path = motion_dir / f".{safe_motion}_masked_policy_jetson_nn_raw.mp4"
    video_path = web_mp4_path(motion_dir / f"{safe_motion}_masked_policy_jetson_nn.mp4")
    log_path = motion_dir / "masked_policy_jetson_nn.csv"
    meta_path = motion_dir / "masked_policy_jetson_nn_meta.json"

    print(f"Checkpoint: {checkpoint}")
    print(f"Motion: {args.motion_path} | steps={n_steps} | fps={fps}")
    print(f"Jetson controller: {args.jetson_controller}")
    print(f"Jetson stance model: {controller.stance_model_path}")
    print(f"Jetson swing model: {controller.swing_model_path}")
    print(f"Disabled muscles: {disabled_actuators.size} | disabled_muscle_scale={args.disabled_muscle_scale}")
    print(f"Recording -> {video_path}")

    obs = env.reset()
    obs_policy = runner.reset_obs(apply_obs_mask(obs, spec))
    phase_tracker.reset()

    knee_target = np.asarray(env.data.qpos[audit.qpos_indices], dtype=np.float64)
    zero_kp = np.zeros(4, dtype=np.float64)
    zero_kd = np.zeros(4, dtype=np.float64)
    torque_limit = np.asarray(
        [args.knee_torque_limit, args.ankle_torque_limit, args.foot_torque_limit, args.foot_torque_limit],
        dtype=np.float64,
    )
    torque_slew = np.full(4, float(args.torque_slew_limit), dtype=np.float64)
    last_torque = np.zeros(4, dtype=np.float64)

    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 6.0
    cam.elevation = -20.0
    cam.azimuth = 90.0

    episode_return = 0.0
    done_count = 0
    disabled_ctrl_max = 0.0
    rows = 0
    phase_switches = 0

    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(
        [
            "step",
            "phase",
            "contact_load_n",
            "hip_sagittal_deg",
            "time_in_phase_s",
            "knee_angle_deg",
            "knee_velocity_deg_s",
            "predicted_phi",
            "predicted_output",
            "control_mode",
            "knee_target_rad",
            "knee_torque",
            "ankle_torque",
            "subtalar_torque",
            "mtp_torque",
            "disabled_ctrl_norm",
            "root_height",
            "done",
        ]
    )

    try:
        with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
            for step in range(n_steps):
                policy_action, _value = runner.act(obs_policy)
                policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)

                q = np.asarray(env.data.qpos[audit.qpos_indices], dtype=np.float64)
                qd = np.asarray(env.data.qvel[audit.qvel_indices], dtype=np.float64)
                contact_load_n = _prosthesis_contact_load_n(env.model, env.data, prosthesis_body_ids)
                phase, time_in_phase, changed = phase_tracker.update(contact_load_n, step * control_dt)
                phase_switches += int(changed)

                hip_sagittal_deg = (
                    float(np.rad2deg(env.data.qpos[hip_qpos_idx])) if hip_qpos_idx is not None else 0.0
                )
                knee_angle_deg = float(np.rad2deg(q[0]))
                knee_velocity_deg_s = float(np.rad2deg(qd[0]))
                predicted_phi, predicted_output, control_mode = controller.predict(
                    phase,
                    hip_sagittal_deg,
                    time_in_phase,
                    knee_angle_deg,
                    knee_velocity_deg_s,
                )

                kp = zero_kp.copy()
                kd = zero_kd.copy()
                ff_torque = np.zeros(4, dtype=np.float64)
                target_q = q.copy()
                signed_output = float(args.knee_output_sign) * float(predicted_output)
                if control_mode == 1:
                    ff_torque[0] = np.clip(signed_output, -args.knee_torque_limit, args.knee_torque_limit)
                    knee_target[0] = q[0]
                else:
                    target_q[0] = np.deg2rad(signed_output)
                    knee_target[0] = target_q[0]
                    kp[0] = float(args.knee_position_kp)
                    kd[0] = float(args.knee_position_kd)

                obs, reward, _absorbing, done, _info = step_with_stage1_pd(
                    env,
                    policy_action,
                    IndexedValues(audit.qpos_indices, target_q),
                    audit.qvel_indices,
                    disabled_actuators,
                    args.disabled_muscle_scale,
                    kp,
                    kd,
                    torque_limit,
                    torque_slew,
                    last_torque,
                    ff_torque,
                )
                obs_policy = runner.update_obs(apply_obs_mask(obs, spec))
                episode_return += float(np.asarray(reward).item())
                done_count += int(bool(done))
                disabled_norm = float(np.linalg.norm(env.data.ctrl[disabled_actuators])) if disabled_actuators.size else 0.0
                disabled_ctrl_max = max(disabled_ctrl_max, disabled_norm)

                diag = JetsonStepDiagnostics(
                    phase=phase,
                    contact_load_n=float(contact_load_n),
                    hip_sagittal_deg=hip_sagittal_deg,
                    time_in_phase_s=float(time_in_phase),
                    knee_angle_deg=knee_angle_deg,
                    knee_velocity_deg_s=knee_velocity_deg_s,
                    predicted_phi=float(predicted_phi),
                    predicted_output=float(predicted_output),
                    control_mode=int(control_mode),
                    knee_target_rad=float(knee_target[0]),
                    knee_torque=float(last_torque[0]),
                    ankle_torque=float(last_torque[1]),
                    subtalar_torque=float(last_torque[2]),
                    mtp_torque=float(last_torque[3]),
                )
                log_w.writerow(
                    [
                        step + 1,
                        diag.phase,
                        f"{diag.contact_load_n:.6f}",
                        f"{diag.hip_sagittal_deg:.6f}",
                        f"{diag.time_in_phase_s:.6f}",
                        f"{diag.knee_angle_deg:.6f}",
                        f"{diag.knee_velocity_deg_s:.6f}",
                        f"{diag.predicted_phi:.6f}",
                        f"{diag.predicted_output:.6f}",
                        diag.control_mode,
                        f"{diag.knee_target_rad:.6f}",
                        f"{diag.knee_torque:.6f}",
                        f"{diag.ankle_torque:.6f}",
                        f"{diag.subtalar_torque:.6f}",
                        f"{diag.mtp_torque:.6f}",
                        f"{disabled_norm:.9f}",
                        f"{float(env.data.qpos[2]):.6f}",
                        int(bool(done)),
                    ]
                )
                rows += 1

                cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())

                if step % 200 == 0:
                    print(
                        f"step={step+1}/{n_steps} phase={phase} "
                        f"knee={q[0]:.3f} out={predicted_output:.2f} mode={control_mode} "
                        f"disabled_norm={disabled_norm:.3e}",
                        flush=True,
                    )
    finally:
        renderer.close()
        env.stop()
        log_f.close()

    encode_web_mp4(raw_path, video_path, remove_src=not args.keep_raw)

    meta = {
        "replay_mode": "masked_full_muscle_policy_with_jetson_onnx_prosthesis_controller",
        "description": (
            "DAgger masked-observation full-muscle policy controls the full model; "
            "the 19 left prosthesis-boundary muscles are hard-masked at execution; "
            "the Jetson ONNX prosthesis controller drives the left knee. "
            "Stance output is applied as knee torque; swing output is tracked as a knee angle target."
        ),
        "checkpoint": str(checkpoint),
        "motion_path": args.motion_path,
        "n_steps": n_steps,
        "recorded_steps": rows,
        "fps": fps,
        "duration_s": float(rows * control_dt),
        "episode_return": episode_return,
        "done_count": done_count,
        "phase_switches": phase_switches,
        "disabled_muscle_count": int(disabled_actuators.size),
        "disabled_muscle_names": list(DEFAULT_DISABLED_MUSCLE_NAMES),
        "disabled_muscle_scale": float(args.disabled_muscle_scale),
        "disabled_ctrl_max_norm": disabled_ctrl_max,
        "masked_obs_dim": spec.masked_obs_dim,
        "raw_obs_dim": spec.raw_obs_dim,
        "policy_action_dim": spec.action_dim,
        "jetson_controller": args.jetson_controller,
        "jetson_package_dir": str(package_dir),
        "jetson_stance_model": str(controller.stance_model_path),
        "jetson_swing_model": str(controller.swing_model_path),
        "jetson_inputs": {
            "limited": ["hip_sagittal_deg", "time_in_phase_s"],
            "full": ["hip_sagittal_deg", "time_in_phase_s", "knee_angle_deg", "knee_velocity_deg_s"],
        }[args.jetson_controller],
        "jetson_outputs": {
            "stance": "knee torque Nm",
            "swing": "knee angle deg",
        },
        "prosthesis_dofs": [j.name for j in audit.joints],
        "contact_threshold_n": float(args.contact_threshold_n),
        "min_phase_time": float(args.min_phase_time),
        "ramp_incline": float(args.ramp_incline),
        "subject_height": float(args.subject_height),
        "subject_weight": float(args.subject_weight),
        "knee_output_sign": float(args.knee_output_sign),
        "knee_position_kp": float(args.knee_position_kp),
        "knee_position_kd": float(args.knee_position_kd),
        "torque_limits": torque_limit.tolist(),
        "torque_slew_limit": float(args.torque_slew_limit),
        "video": str(video_path),
        "deploy_log_csv": str(log_path),
        "distill_metadata": metadata,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Saved video: {video_path}")
    print(f"Saved log: {log_path}")
    print(f"Saved meta: {meta_path}")
    print(f"episode_return={episode_return:.3f} done_count={done_count} disabled_ctrl_max_norm={disabled_ctrl_max:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
