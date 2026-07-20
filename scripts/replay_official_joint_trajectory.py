#!/usr/bin/env python
"""Capture official-policy joint trajectory, then kinematically replay joints (no muscle ctrl)."""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from musclemimic.distill.config import load_fullbody_config, make_env, repo_root
from musclemimic.evaluation.adapters import build_adapter, resolve_n_steps
from musclemimic.evaluation.joint_replay import (
    joint_trajectory_from_buffer,
    load_joint_trajectory,
    record_kinematic_replay,
    save_joint_trajectory,
    write_replay_meta,
)
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.rollout_loop import run_rollout
from musclemimic.evaluation.types import EvalConfig
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--motion_path",
        default="KIT/3/walk_6m_straight_line04_poses",
        help="Reference motion used for official rollout (default: 7s straight walk).",
    )
    p.add_argument(
        "--checkpoint_path",
        default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2",
    )
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--n_steps", default="auto", help="Rollout length; auto = full trajectory.")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Default: outputs/replay/joint_mimic/<controller>/<safe_motion>/",
    )
    p.add_argument("--joint_trajectory", default=None, help="Existing joint_trajectory.npz; skip rollout if set.")
    p.add_argument("--skip_rollout", action="store_true", help="Require existing joint_trajectory.npz.")
    p.add_argument("--skip_video", action="store_true")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--cam-distance", type=float, default=3.5)
    p.add_argument("--cam-elevation", type=float, default=0.0)
    p.add_argument("--cam-azimuth", type=float, default=90.0)
    return p.parse_args()


def default_output_dir(controller_type: str, motion_path: str) -> Path:
    return repo_root() / "outputs" / "replay" / "joint_mimic" / controller_type / safe_motion_name(motion_path)


def build_replay_env(config, motion_path: str):
    cfg = deepcopy(config)
    OmegaConf.set_struct(cfg, False)
    env_params = OmegaConf.to_container(cfg.experiment.env_params, resolve=True)
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    apply_eval_terminal_defaults(env_params, cfg, strict_termination=False)
    cfg.experiment.env_params = env_params
    return make_env(cfg, env_name="MyoFullBody", motion_paths=[motion_path], use_mujoco=True, fixed_start=True)


def capture_official_joints(args: argparse.Namespace, out_dir: Path):
    eval_config = EvalConfig(
        config_name="conf_fullbody_gmr_resnet",
        checkpoint_path=args.checkpoint_path,
        controller_type=args.controller_type,
        env_type="official_fullbody",
        output_dir=out_dir,
        eval_seed=args.eval_seed,
        n_steps=args.n_steps,
        use_mujoco=True,
        save_video=False,
        save_plots=False,
        show_ghost=False,
        no_termination=True,
    )
    adapter = build_adapter(eval_config, args.motion_path)
    try:
        traj_len = adapter.get_traj_length()
        max_steps = resolve_n_steps(eval_config.n_steps, traj_len)
        buffer = run_rollout(adapter, max_steps=max_steps)
        joint_traj = joint_trajectory_from_buffer(
            buffer,
            source_controller=args.controller_type,
            source_checkpoint=args.checkpoint_path,
        )
        traj_path = out_dir / "joint_trajectory.npz"
        save_joint_trajectory(traj_path, joint_traj)
        return joint_traj, traj_path, buffer
    finally:
        adapter.close()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.controller_type, args.motion_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_npz = Path(args.joint_trajectory) if args.joint_trajectory else out_dir / "joint_trajectory.npz"
    buffer = None

    if args.skip_rollout or traj_npz.is_file():
        if not traj_npz.is_file():
            raise FileNotFoundError(f"Missing joint trajectory: {traj_npz}")
        joint_traj = load_joint_trajectory(traj_npz)
        print(f"Loaded joint trajectory: {traj_npz} ({joint_traj.n_frames} frames, {joint_traj.duration_s:.2f}s)")
    else:
        print(f"Capturing official rollout joints -> {traj_npz}")
        joint_traj, traj_npz, buffer = capture_official_joints(args, out_dir)
        print(
            f"Saved {joint_traj.n_frames} frames ({joint_traj.duration_s:.2f}s) "
            f"done_reason={joint_traj.metadata.get('done_reason') if joint_traj.metadata else '?'}"
        )

    video_path = None
    if not args.skip_video:
        config = load_fullbody_config("conf_fullbody_gmr_resnet")
        env = build_replay_env(config, args.motion_path)
        video_path = web_mp4_path(out_dir / "joint_replay.mp4")
        raw_path = out_dir / ".joint_replay_raw.mp4"
        print(f"Recording kinematic joint replay -> {video_path}")
        record_kinematic_replay(
            env=env,
            traj=joint_traj,
            record_path=raw_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            cam_distance=args.cam_distance,
            cam_elevation=args.cam_elevation,
            cam_azimuth=args.cam_azimuth,
        )
        encode_web_mp4(raw_path, video_path)
        env.stop()
        print(f"Saved joint replay video: {video_path}")

    meta = {
        "motion_path": args.motion_path,
        "controller_type": args.controller_type,
        "checkpoint_path": args.checkpoint_path,
        "replay_mode": "joint_kinematic",
        "description": (
            "Official policy rollout qpos/qvel replayed by directly setting joint state each frame; "
            "muscle controls are zero (mechanical mimic, not muscle tracking)."
        ),
        "joint_trajectory_npz": str(traj_npz),
        "joint_replay_video": str(video_path) if video_path else None,
        "n_frames": joint_traj.n_frames,
        "duration_s": joint_traj.duration_s,
        "dt": joint_traj.dt,
        "source_controller": joint_traj.source_controller,
        "source_checkpoint": joint_traj.source_checkpoint,
        "capture_metadata": joint_traj.metadata,
    }
    if buffer is not None:
        meta["episode_return"] = float(buffer.episode_return)
    write_replay_meta(out_dir / "replay_meta.json", meta)
    print(f"Wrote metadata: {out_dir / 'replay_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
