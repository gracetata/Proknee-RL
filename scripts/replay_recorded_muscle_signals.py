#!/usr/bin/env python
"""Replay recorded full-muscle action signals open-loop and record videos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.distill.config import repo_root
from musclemimic.evaluation.joint_replay import write_replay_meta
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.muscle_replay import (
    actuator_ids_for_muscle_names,
    color_disabled_muscle_tendons_blue,
    load_muscle_trajectory,
    record_open_loop_actuator_ctrl_masked_replay,
    record_open_loop_muscle_signal_replay,
)
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint, setup_headless

FOUR_REPLAY_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", default=None)
    p.add_argument("--all-four", action="store_true")
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument(
        "--checkpoint_path",
        default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2",
        help="Checkpoint used only to recover the matching env config.",
    )
    p.add_argument(
        "--replay_root",
        default=None,
        help="Output root. Default: outputs/replay/muscle_mimic/<controller_type>",
    )
    p.add_argument(
        "--source-root",
        default=None,
        help="Source muscle_trajectory.npz root. Default: outputs/replay/muscle_mimic/<controller_type>",
    )
    p.add_argument(
        "--output-tag",
        default=None,
        help="Optional suffix for masked-replay output filenames, e.g. closecam.",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--cam-distance", type=float, default=6.0)
    p.add_argument("--cam-elevation", type=float, default=-20.0)
    p.add_argument("--cam-azimuth", type=float, default=90.0)
    p.add_argument(
        "--mask-preset",
        default=None,
        choices=["full", "foot1", "foot5", "distal11", "knee15", "strict19"],
        help="Disabled-muscle preset for low-level actuator_ctrl replay; use 'full' to replay all muscles.",
    )
    return p.parse_args()


def _default_muscle_replay_root(args: argparse.Namespace) -> Path:
    return repo_root() / "outputs" / "replay" / "muscle_mimic" / str(args.controller_type)


def source_root(args: argparse.Namespace) -> Path:
    if args.source_root:
        root = Path(args.source_root)
        return root if root.is_absolute() else repo_root() / root
    return _default_muscle_replay_root(args)


def replay_root(args: argparse.Namespace) -> Path:
    if args.replay_root:
        root = Path(args.replay_root)
        return root if root.is_absolute() else repo_root() / root
    return _default_muscle_replay_root(args)


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


def replay_one(args: argparse.Namespace, config, motion_path: str) -> dict:
    safe = safe_motion_name(motion_path)
    src_dir = source_root(args) / safe
    traj_path = src_dir / "muscle_trajectory.npz"
    if not traj_path.is_file():
        raise FileNotFoundError(f"Missing recorded muscle trajectory: {traj_path}")
    traj = load_muscle_trajectory(traj_path)

    preset = args.mask_preset
    if preset is None:
        motion_dir = src_dir
        raw_path = motion_dir / ".muscle_signal_replay_raw.mp4"
        video_path = web_mp4_path(motion_dir / "muscle_signal_replay.mp4")
        meta_path = motion_dir / "muscle_signal_replay_meta.json"
    else:
        tag = f"_{args.output_tag}" if args.output_tag else ""
        motion_dir = replay_root(args) / preset / safe
        motion_dir.mkdir(parents=True, exist_ok=True)
        stem = f"muscle_ctrl_{preset}_replay{tag}"
        raw_path = motion_dir / f".{stem}_raw.mp4"
        video_path = web_mp4_path(motion_dir / f"{stem}.mp4")
        meta_path = motion_dir / f"{stem}_meta.json"

    env = build_env(config, motion_path)
    try:
        if preset is None:
            print(f"[muscle signal replay] {motion_path} -> {video_path}")
            episode_return, done_count, n_steps = record_open_loop_muscle_signal_replay(
                env=env,
                traj=traj,
                record_path=raw_path,
                width=int(args.width),
                height=int(args.height),
                fps=args.fps,
                cam_distance=float(args.cam_distance),
                cam_elevation=float(args.cam_elevation),
                cam_azimuth=float(args.cam_azimuth),
            )
            meta = {
                "motion_path": motion_path,
                "replay_mode": "recorded_muscle_signal_open_loop",
                "description": (
                    "Open-loop replay of saved full-muscle policy_action frames. "
                    "The policy is not queried during this replay."
                ),
                "source_muscle_trajectory_npz": str(traj_path),
                "muscle_signal_replay_video": str(video_path),
                "n_frames": int(n_steps),
                "duration_s": float(n_steps * traj.dt),
                "dt": float(traj.dt),
                "episode_return": float(episode_return),
                "done_count": int(done_count),
                "source_controller": traj.source_controller,
                "source_checkpoint": traj.source_checkpoint,
            }
        else:
            disabled_names = MUSCLE_MASK_PRESETS[preset]
            disabled_actuators = actuator_ids_for_muscle_names(env.model, disabled_names)
            highlighted = color_disabled_muscle_tendons_blue(env.model, disabled_names)
            print(f"[{preset} muscle ctrl replay] {motion_path} -> {video_path}")
            print(f"disabled_actuators={disabled_actuators.tolist()} highlighted_tendons={highlighted}")
            episode_return, done_count, n_steps = record_open_loop_actuator_ctrl_masked_replay(
                env=env,
                traj=traj,
                disabled_actuators=disabled_actuators,
                record_path=raw_path,
                width=int(args.width),
                height=int(args.height),
                fps=args.fps,
                cam_distance=float(args.cam_distance),
                cam_elevation=float(args.cam_elevation),
                cam_azimuth=float(args.cam_azimuth),
            )
            meta = {
                "motion_path": motion_path,
                "replay_mode": f"recorded_actuator_ctrl_{preset}_open_loop",
                "description": (
                    "Open-loop replay of recorded low-level actuator_ctrl frames with selected "
                    "disabled muscle actuators zeroed. No policy and no external prosthesis controller."
                ),
                "mask_preset": preset,
                "disabled_muscle_names": list(disabled_names),
                "disabled_actuator_ids": disabled_actuators.tolist(),
                "highlighted_disabled_muscle_tendons": int(highlighted),
                "source_muscle_trajectory_npz": str(traj_path),
                "video": str(video_path),
                "n_frames": int(n_steps),
                "duration_s": float(n_steps * traj.dt),
                "dt": float(traj.dt),
                "episode_return": float(episode_return),
                "done_count": int(done_count),
                "source_controller": traj.source_controller,
                "source_checkpoint": traj.source_checkpoint,
                "cam_distance": float(args.cam_distance),
                "cam_elevation": float(args.cam_elevation),
                "cam_azimuth": float(args.cam_azimuth),
            }

        encode_web_mp4(raw_path, video_path, remove_src=True)
        write_replay_meta(meta_path, meta)
        print(f"  frames={n_steps} return={episode_return:.3f} done_count={done_count}")
        return meta
    finally:
        env.stop()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    motions = list(FOUR_REPLAY_MOTIONS) if args.all_four else [args.motion_path]
    if not motions or motions == [None]:
        raise ValueError("Provide --motion_path or --all-four")

    config, _agent_state, _metadata = load_checkpoint(args.checkpoint_path)
    OmegaConf.set_struct(config, False)
    apply_temporal_params(config)

    records = [replay_one(args, config, motion) for motion in motions]
    if len(records) > 1:
        summary_dir = replay_root(args) / "replay_four"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_replay_meta(summary_dir / "muscle_signal_replay_summary.json", {"motions": records})
        print(f"Wrote summary: {summary_dir / 'muscle_signal_replay_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
