#!/usr/bin/env python
"""Record masked full-muscle policy replay with a pluggable left-leg prosthesis controller."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
OSL_DIR = REPO_ROOT.parent / "musclebenchmark" / "opensourceleg_fsm"
for path in (str(REPO_ROOT), str(OSL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from official_fsm import OpenSourceLegFSMController  # noqa: E402
from sim_adapter import (  # noqa: E402
    DEFAULT_MUJOCO_LOADCELL_SCALE,
    MuscleMimicOSLAdapter,
    OSLAdapterConfig,
    model_body_weight_n,
)

from fullbody._eval_terminal import apply_eval_terminal_defaults  # noqa: E402
from loco_mujoco.task_factories import TaskFactory  # noqa: E402
from musclemimic.algorithms import PPOJax  # noqa: E402
from musclemimic.distill.obs_mask import MaskedObservationEnvView, apply_obs_mask  # noqa: E402
from musclemimic.distill.mapping import build_distill_mapping  # noqa: E402
from musclemimic.distill.osl_deploy_defaults import (  # noqa: E402
    DEFAULT_OSL_ANKLE_TORQUE_LIMIT,
    DEFAULT_OSL_FOOT_TORQUE_LIMIT,
    DEFAULT_OSL_KNEE_TORQUE_LIMIT,
    DEFAULT_OSL_LOADCELL_SCALE,
    DEFAULT_OSL_TORQUE_SLEW_LIMIT,
)
from musclemimic.distill.osl_harness import scheduled_muscle_scale  # noqa: E402
from musclemimic.distill.policy import PolicyRunner  # noqa: E402
from musclemimic.distill.prosthesis_deploy import build_deploy_harness  # noqa: E402
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path  # noqa: E402
from musclemimic.proknee.constants import audit_myofullbody_left_leg  # noqa: E402
from musclemimic.prosthesis.constants import DEFAULT_DISABLED_MUSCLE_NAMES  # noqa: E402
from musclemimic.runner.eval_utils import (  # noqa: E402
    align_agent_state,
    apply_temporal_params,
    load_checkpoint,
    setup_headless,
)
from record_masked_obs_full_muscle_replay import find_mask_spec  # noqa: E402
from record_stage1_testset_visual import IndexedValues, step_with_stage1_pd  # noqa: E402


DEFAULT_CHECKPOINT = (
    "outputs/full_muscle_masked_obs_dagger1_focused_round3_distill/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "outputs/eval_masked_dagger_compare/replay_four/dagger1_focused_round3_osl_fsm"


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
    p.add_argument("--knee-torque-limit", type=float, default=DEFAULT_OSL_KNEE_TORQUE_LIMIT)
    p.add_argument("--ankle-torque-limit", type=float, default=DEFAULT_OSL_ANKLE_TORQUE_LIMIT)
    p.add_argument("--foot-torque-limit", type=float, default=DEFAULT_OSL_FOOT_TORQUE_LIMIT)
    p.add_argument("--torque-slew-limit", type=float, default=DEFAULT_OSL_TORQUE_SLEW_LIMIT)
    p.add_argument("--loadcell-scale", type=float, default=DEFAULT_OSL_LOADCELL_SCALE)
    p.add_argument("--body-weight-n", type=float, default=None)
    p.add_argument(
        "--disabled-muscle-scale",
        type=float,
        default=0.0,
        help="Target scale for the 19 left prosthesis-boundary muscle controls; 0 = fully masked, 1 = unchanged.",
    )
    p.add_argument(
        "--disabled-muscle-scale-start",
        type=float,
        default=1.0,
        help="Initial muscle scale when --disabled-muscle-scale-ramp-steps > 0 (default 1.0 = healthy dynamics).",
    )
    p.add_argument(
        "--disabled-muscle-scale-ramp-steps",
        type=int,
        default=0,
        help="Debug-only: ramp muscle scale before reaching target. Do not combine partial scale with prosthesis torque.",
    )
    p.add_argument(
        "--prosthesis-controller",
        default="osl_fsm",
        choices=("osl_fsm", "reference_pd", "fsm_impedance", "jetson_limited", "jetson_full"),
        help="Pluggable prosthesis controller (4-DoF torque or Jetson ONNX knee).",
    )
    p.add_argument("--jetson-package-dir", default=None)
    p.add_argument("--jetson-ramp-incline", type=float, default=0.0)
    p.add_argument("--jetson-subject-height", type=float, default=1.75)
    p.add_argument("--jetson-subject-weight", type=float, default=70.0)
    p.add_argument("--jetson-contact-threshold-n", type=float, default=20.0)
    p.add_argument("--jetson-min-phase-time", type=float, default=0.08)
    p.add_argument("--jetson-knee-position-kp", type=float, default=35.0)
    p.add_argument("--jetson-knee-position-kd", type=float, default=4.0)
    p.add_argument("--jetson-knee-output-sign", type=float, default=1.0)
    p.add_argument("--keep-raw", action="store_true")
    return p.parse_args()


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

    env_pr_params = OmegaConf.to_container(_config.experiment.env_params, resolve=True)
    env_pr_params["env_name"] = "MyoFullBodyProsthesisEnv"
    env_pr = TaskFactory.get_factory_cls(_config.experiment.task_factory.name).make(
        **env_pr_params,
        **OmegaConf.to_container(_config.experiment.task_factory.params, resolve=True),
        prosthesis={"enabled": True},
    )
    mapping = build_distill_mapping(env, env_pr)
    env_pr.stop()

    jetson_mode = args.prosthesis_controller.startswith("jetson_")
    ankle_torque_limit = 0.0 if jetson_mode else args.ankle_torque_limit
    foot_torque_limit = 0.0 if jetson_mode else args.foot_torque_limit
    deploy_harness = build_deploy_harness(
        env,
        controller_name=args.prosthesis_controller,
        disabled_action_indices=np.asarray(mapping.teacher_disabled_action_indices, dtype=np.int32),
        disabled_muscle_scale=float(args.disabled_muscle_scale),
        knee_torque_limit=args.knee_torque_limit,
        ankle_torque_limit=ankle_torque_limit,
        foot_torque_limit=foot_torque_limit,
        torque_slew_limit=args.torque_slew_limit,
        loadcell_scale=args.loadcell_scale,
        body_weight_n=args.body_weight_n,
        jetson_package_dir=args.jetson_package_dir,
        jetson_ramp_incline=args.jetson_ramp_incline,
        jetson_subject_height=args.jetson_subject_height,
        jetson_subject_weight=args.jetson_subject_weight,
        jetson_contact_threshold_n=args.jetson_contact_threshold_n,
        jetson_min_phase_time=args.jetson_min_phase_time,
        jetson_knee_position_kp=args.jetson_knee_position_kp,
        jetson_knee_position_kd=args.jetson_knee_position_kd,
        jetson_knee_output_sign=args.jetson_knee_output_sign,
        control_dt=float(control_dt),
    )
    thresholds = None
    body_weight_n = None
    if args.prosthesis_controller == "osl_fsm":
        body_weight_n = deploy_harness.harness.body_weight_n
        thresholds = deploy_harness.harness.controller.thresholds
    elif args.prosthesis_controller.startswith("jetson_"):
        body_weight_n = None
        thresholds = None
        print(f"Jetson stance model: {deploy_harness.controller.stance_model_path}")
        print(f"Jetson swing model: {deploy_harness.controller.swing_model_path}")

    n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
    fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
    raw_path = motion_dir / f".{safe_motion}_masked_policy_osl_fsm_raw.mp4"
    video_path = web_mp4_path(motion_dir / f"{safe_motion}_masked_policy_osl_fsm.mp4")
    log_path = motion_dir / "masked_policy_osl_fsm.csv"
    meta_path = motion_dir / "masked_policy_osl_fsm_meta.json"

    print(f"Checkpoint: {checkpoint}")
    print(f"Motion: {args.motion_path} | steps={n_steps} | fps={fps}")
    if int(args.disabled_muscle_scale_ramp_steps) > 0:
        print(
            f"Disabled muscles: {disabled_actuators.size} | "
            f"scale ramp {args.disabled_muscle_scale_start} -> {args.disabled_muscle_scale} "
            f"over {args.disabled_muscle_scale_ramp_steps} steps"
        )
    else:
        print(f"Disabled muscles: {disabled_actuators.size} | disabled_muscle_scale={args.disabled_muscle_scale}")
    print(f"Prosthesis controller: {args.prosthesis_controller}")
    if thresholds is not None and body_weight_n is not None:
        print(
            "OSL calibration: "
            f"body_weight_n={body_weight_n:.1f} "
            f"thresholds=({thresholds.load_lstance_n:.1f}, {thresholds.load_eswing_n:.1f}, "
            f"{thresholds.load_estance_n:.1f}) loadcell_scale={args.loadcell_scale}"
        )
    print(f"Recording -> {video_path}")

    obs = env.reset()
    obs_policy = runner.reset_obs(apply_obs_mask(obs, spec))
    deploy_harness.reset()
    torque_limit = np.asarray(
        [args.knee_torque_limit, args.ankle_torque_limit, args.foot_torque_limit, args.foot_torque_limit],
        dtype=np.float64,
    )

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

    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(
        [
            "step",
            "fsm_state",
            "loadcell_fz_n",
            "knee_q",
            "ankle_q",
            "subtalar_q",
            "mtp_q",
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
                muscle_scale = scheduled_muscle_scale(
                    step,
                    end_scale=float(args.disabled_muscle_scale),
                    ramp_steps=int(args.disabled_muscle_scale_ramp_steps),
                    start_scale=float(args.disabled_muscle_scale_start),
                )
                deploy_harness.disabled_muscle_scale = float(muscle_scale)
                policy_action, _value = runner.act(obs_policy)
                obs, reward, done, diag = deploy_harness.step(policy_action)
                obs_policy = runner.update_obs(apply_obs_mask(obs, spec))
                episode_return += float(np.asarray(reward).item())
                done_count += int(bool(done))
                disabled_norm = float(diag.get("disabled_ctrl_norm", 0.0))
                disabled_ctrl_max = max(disabled_ctrl_max, disabled_norm)

                q = np.asarray(env.data.qpos[audit.qpos_indices], dtype=np.float64)
                prosthesis_tau = np.asarray(
                    diag.get("prosthesis_torque", diag.get("knee_torque", np.zeros(4))),
                    dtype=np.float64,
                ).reshape(-1)
                if prosthesis_tau.size < 4:
                    prosthesis_tau = np.pad(
                        prosthesis_tau,
                        (0, 4 - prosthesis_tau.size),
                        constant_values=float(diag.get("knee_torque", 0.0)),
                    )
                log_w.writerow(
                    [
                        step + 1,
                        str(diag.get("state", args.prosthesis_controller)),
                        f"{float(diag.get('loadcell_fz_n', 0.0)):.6f}",
                        f"{q[0]:.6f}",
                        f"{q[1]:.6f}",
                        f"{q[2]:.6f}",
                        f"{q[3]:.6f}",
                        f"{float(prosthesis_tau[0]):.6f}",
                        f"{float(prosthesis_tau[1]):.6f}",
                        f"{float(prosthesis_tau[2]):.6f}",
                        f"{float(prosthesis_tau[3]):.6f}",
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
                        f"step={step+1}/{n_steps} controller={args.prosthesis_controller} "
                        f"knee={q[0]:.3f} ankle={q[1]:.3f} disabled_norm={disabled_norm:.3e}",
                        flush=True,
                    )
    finally:
        renderer.close()
        env.stop()
        log_f.close()

    encode_web_mp4(raw_path, video_path, remove_src=not args.keep_raw)

    meta = {
        "replay_mode": "masked_full_muscle_policy_with_left_prosthesis_controller",
        "description": (
            "Masked-observation full-muscle policy drives remaining muscles; "
            "the 19 left prosthesis-boundary muscle outputs are zeroed at execution; "
            "a pluggable 4-DoF prosthesis controller injects joint torques."
        ),
        "prosthesis_controller": args.prosthesis_controller,
        "checkpoint": str(checkpoint),
        "motion_path": args.motion_path,
        "n_steps": n_steps,
        "recorded_steps": rows,
        "fps": fps,
        "duration_s": float(rows * control_dt),
        "episode_return": episode_return,
        "done_count": done_count,
        "disabled_muscle_count": int(disabled_actuators.size),
        "disabled_muscle_names": list(DEFAULT_DISABLED_MUSCLE_NAMES),
        "disabled_muscle_scale": float(args.disabled_muscle_scale),
        "disabled_muscle_scale_start": float(args.disabled_muscle_scale_start),
        "disabled_muscle_scale_ramp_steps": int(args.disabled_muscle_scale_ramp_steps),
        "disabled_ctrl_max_norm": disabled_ctrl_max,
        "masked_obs_dim": spec.masked_obs_dim,
        "raw_obs_dim": spec.raw_obs_dim,
        "policy_action_dim": spec.action_dim,
        "osl_inputs": ["knee_position_rad", "knee_velocity_rad_s", "ankle_position_rad", "loadcell_fz_n"],
        "osl_torque_dofs": [j.name for j in audit.joints],
        "fsm_body_weight_n": body_weight_n,
        "fsm_load_thresholds_n": (
            {
                "load_lstance": thresholds.load_lstance_n,
                "load_eswing": thresholds.load_eswing_n,
                "load_estance": thresholds.load_estance_n,
            }
            if thresholds is not None
            else None
        ),
        "loadcell_scale": args.loadcell_scale,
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
