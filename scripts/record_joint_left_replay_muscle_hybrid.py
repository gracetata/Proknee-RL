#!/usr/bin/env python
"""Hybrid rollout: lock left leg to reference joint angles; replay remaining muscles from same capture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.distill.config import repo_root
from musclemimic.evaluation.joint_replay import load_joint_trajectory, write_replay_meta
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.muscle_replay import (
    actuator_ids_for_muscle_names,
    aligned_hybrid_replay_steps_from_muscle_only,
    aligned_joint_left_muscle_ctrl_steps,
    color_disabled_muscle_tendons_blue,
    load_muscle_trajectory,
    record_left_joint_pinned_actuator_ctrl_masked_replay,
)
from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint, setup_headless


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", default="KIT/3/walk_6m_straight_line04_poses")
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument(
        "--checkpoint_path",
        default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2",
    )
    p.add_argument(
        "--muscle-trajectory",
        default=None,
        help="Default: outputs/replay/joint_muscle_hybrid/<controller>/<SAFE>/aligned_muscle_trajectory.npz",
    )
    p.add_argument(
        "--alignment",
        choices=["joint", "muscle"],
        default="joint",
        help="joint = official joint replay for left 4-DoF + muscle ctrl for rest; "
        "muscle = left reference from same muscle capture (debug only).",
    )
    p.add_argument(
        "--joint-trajectory",
        default=None,
        help="Official joint_mimic joint_trajectory.npz for left-leg replay. "
        "Default: outputs/replay/joint_mimic/<controller>/<motion>/joint_trajectory.npz",
    )
    p.add_argument(
        "--replay_root",
        default=None,
        help="Default: outputs/replay/joint_muscle_hybrid/<controller_type>",
    )
    p.add_argument("--mask-preset", choices=sorted(MUSCLE_MASK_PRESETS), default="knee15")
    p.add_argument("--output-tag", default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--cam-distance", type=float, default=6.0)
    p.add_argument("--cam-elevation", type=float, default=-20.0)
    p.add_argument("--cam-azimuth", type=float, default=90.0)
    p.add_argument("--left-joint-kp", default="240,180,60,40")
    p.add_argument("--left-joint-kd", default="24,18,6,4")
    p.add_argument("--left-torque-limit", default="140,120,60,60")
    p.add_argument("--torque-slew-limit", type=float, default=35.0)
    p.add_argument(
        "--left-joint-control-mode",
        choices=["lock", "pd"],
        default="lock",
    )
    p.add_argument(
        "--hybrid-mode",
        choices=["left_only", "scaffold"],
        default="left_only",
        help="left_only = muscle physics for non-left, hard-lock left 4-DoF from joint replay; "
        "scaffold = kinematic pin for non-left (debug / comparison only).",
    )
    p.add_argument(
        "--recapture",
        action="store_true",
        help="Run capture_hybrid_reference_trajectory.py before replay.",
    )
    return p.parse_args()


def parse_vec4(value: str) -> np.ndarray:
    vals = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if len(vals) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got {value!r}")
    return np.asarray(vals, dtype=np.float64)


def hybrid_dir(args: argparse.Namespace) -> Path:
    safe = safe_motion_name(args.motion_path)
    return repo_root() / "outputs" / "replay" / "joint_muscle_hybrid" / args.controller_type / safe


def build_env(config, motion_path: str):
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
    amass["rel_dataset_path"] = [motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    if args.recapture:
        import subprocess
        import sys

        cmd = [
            sys.executable,
            str(repo_root() / "scripts" / "capture_hybrid_reference_trajectory.py"),
            "--motion_path",
            args.motion_path,
            "--controller_type",
            args.controller_type,
            "--checkpoint_path",
            args.checkpoint_path,
        ]
        print("Recapturing aligned reference:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(repo_root()))

    base_dir = hybrid_dir(args)
    muscle_path = Path(args.muscle_trajectory) if args.muscle_trajectory else base_dir / "aligned_muscle_trajectory.npz"
    if not muscle_path.is_file():
        raise FileNotFoundError(
            f"Missing aligned muscle trajectory: {muscle_path}. "
            "Run scripts/capture_hybrid_reference_trajectory.py or pass --recapture."
        )

    joint_path = (
        Path(args.joint_trajectory)
        if args.joint_trajectory
        else repo_root()
        / "outputs"
        / "replay"
        / "joint_mimic"
        / args.controller_type
        / safe_motion_name(args.motion_path)
        / "joint_trajectory.npz"
    )
    if not joint_path.is_file():
        raise FileNotFoundError(
            f"Missing official joint trajectory: {joint_path}. "
            "Run scripts/replay_official_joint_trajectory.py first."
        )
    joint_traj = load_joint_trajectory(joint_path)

    muscle_traj = load_muscle_trajectory(muscle_path)
    n_steps = (
        aligned_hybrid_replay_steps_from_muscle_only(muscle_traj.n_frames)
        if args.alignment == "muscle"
        else aligned_joint_left_muscle_ctrl_steps(joint_traj.n_frames, muscle_traj.n_frames)
    )

    replay_root = Path(args.replay_root) if args.replay_root else repo_root() / "outputs" / "replay" / "joint_muscle_hybrid" / args.controller_type
    tag = f"_{args.output_tag}" if args.output_tag else ""
    out_dir = replay_root / args.mask_preset / safe_motion_name(args.motion_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"joint_left_muscle_{args.mask_preset}_replay{tag}"
    raw_path = out_dir / f".{stem}_raw.mp4"
    video_path = web_mp4_path(out_dir / f"{stem}.mp4")
    meta_path = out_dir / f"{stem}_meta.json"

    print(f"Muscle reference: {muscle_path} ({muscle_traj.n_frames} frames, dt={muscle_traj.dt})")
    if joint_path is not None:
        print(f"Joint reference: {joint_path} ({joint_traj.n_frames} frames)")
    print(f"Alignment: {args.alignment}; hybrid steps: {n_steps}")
    print(f"Record: {video_path}")

    config, _agent_state, _metadata = load_checkpoint(args.checkpoint_path)
    OmegaConf.set_struct(config, False)
    apply_temporal_params(config)

    env = build_env(config, args.motion_path)
    try:
        audit = audit_myofullbody_left_leg(env.model)
        left_qpos = np.asarray(audit.qpos_indices, dtype=np.int32)
        left_qvel = np.asarray(audit.qvel_indices, dtype=np.int32)
        all_qpos = np.arange(env.model.nq, dtype=np.int32)
        all_qvel = np.arange(env.model.nv, dtype=np.int32)
        non_left_qpos = np.setdiff1d(all_qpos, left_qpos)
        non_left_qvel = np.setdiff1d(all_qvel, left_qvel)
        disabled_names = MUSCLE_MASK_PRESETS[args.mask_preset]
        disabled_actuators = actuator_ids_for_muscle_names(env.model, disabled_names)
        highlighted = color_disabled_muscle_tendons_blue(env.model, disabled_names)

        episode_return, done_count, n_steps = record_left_joint_pinned_actuator_ctrl_masked_replay(
            env=env,
            muscle_traj=muscle_traj,
            joint_traj=joint_traj,
            alignment=args.alignment,
            left_qpos=left_qpos,
            left_qvel=left_qvel,
            disabled_actuators=disabled_actuators,
            record_path=raw_path,
            width=int(args.width),
            height=int(args.height),
            fps=args.fps,
            cam_distance=float(args.cam_distance),
            cam_elevation=float(args.cam_elevation),
            cam_azimuth=float(args.cam_azimuth),
            left_joint_kp=parse_vec4(args.left_joint_kp),
            left_joint_kd=parse_vec4(args.left_joint_kd),
            left_torque_limit=parse_vec4(args.left_torque_limit),
            torque_slew_limit=float(args.torque_slew_limit),
            left_joint_control_mode=str(args.left_joint_control_mode),
            hybrid_mode=str(args.hybrid_mode),
            non_left_qpos=non_left_qpos,
            non_left_qvel=non_left_qvel,
        )

        meta = {
            "motion_path": args.motion_path,
            "replay_mode": f"joint_left_{args.left_joint_control_mode}_muscle_{args.mask_preset}_hybrid",
            "alignment": args.alignment,
            "hybrid_mode": args.hybrid_mode,
            "description": (
                "Left 4-DoF hard-locked to official joint_mimic replay; non-left body driven only by "
                "recorded actuator_ctrl. Mask preset knee15 plus all left prosthesis-boundary muscles "
                "are zeroed during lock (knee15 alone leaves thigh muscles that fight joint lock). "
                "Complement to joint_replay_osl_fsm (pins non-left, FSM on left)."
            ),
            "left_joint_control_mode": str(args.left_joint_control_mode),
            "mask_preset": args.mask_preset,
            "disabled_muscle_names": list(disabled_names),
            "muscle_trajectory_npz": str(muscle_path.resolve()),
            "joint_trajectory_npz": str(joint_path.resolve()),
            "video": str(video_path),
            "n_frames": int(n_steps),
            "dt": float(muscle_traj.dt),
            "episode_return": float(episode_return),
            "done_count": int(done_count),
        }
        encode_web_mp4(raw_path, video_path, remove_src=True)
        write_replay_meta(meta_path, meta)
        print(f"  frames={n_steps} return={episode_return:.3f} done_count={done_count}")
        print(f"Saved video: {video_path}")
    finally:
        env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
