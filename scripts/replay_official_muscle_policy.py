#!/usr/bin/env python
"""Capture official-policy muscle rollout and record closed-loop muscle replay video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import repo_root
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.joint_replay import write_replay_meta
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.muscle_replay import record_muscle_closed_loop_rollout, save_muscle_trajectory
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless
from loco_mujoco.task_factories import TaskFactory

FOUR_REPLAY_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", default=None)
    p.add_argument("--all-four", action="store_true", help="Record all four replay_four motions.")
    p.add_argument(
        "--checkpoint_path",
        default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2",
    )
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=0, help="0 = full trajectory length")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Default: outputs/replay/muscle_mimic/<controller>/<safe_motion>/",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--cam-distance", type=float, default=6.0)
    p.add_argument("--cam-elevation", type=float, default=-20.0)
    p.add_argument("--cam-azimuth", type=float, default=90.0)
    p.add_argument("--skip_video", action="store_true")
    return p.parse_args()


def default_output_dir(controller_type: str, motion_path: str) -> Path:
    return repo_root() / "outputs" / "replay" / "muscle_mimic" / controller_type / safe_motion_name(motion_path)


def build_env(config, motion_path: str):
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


def record_one(args: argparse.Namespace, motion_path: str) -> dict:
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.controller_type, motion_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    config, agent_state, metadata = load_checkpoint(args.checkpoint_path)
    OmegaConf.set_struct(config, False)
    control_dt = apply_temporal_params(config)
    env = build_env(config, motion_path)
    try:
        agent_conf = PPOJax.init_agent_conf(env, config)
        agent_state = align_agent_state(agent_state, agent_conf)
        runner = PolicyRunner.from_agent_state(
            agent_conf,
            agent_state,
            env,
            deterministic=True,
            seed=int(args.eval_seed),
        )

        n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
        fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
        raw_path = out_dir / ".muscle_replay_raw.mp4"
        video_path = web_mp4_path(out_dir / "muscle_replay.mp4")
        traj_path = out_dir / "muscle_trajectory.npz"

        print(f"[muscle replay] {motion_path} -> {out_dir} ({n_steps} steps)")
        traj, episode_return, done_count = record_muscle_closed_loop_rollout(
            env=env,
            policy=runner,
            n_steps=n_steps,
            width=int(args.width),
            height=int(args.height),
            fps=fps,
            cam_distance=float(args.cam_distance),
            cam_elevation=float(args.cam_elevation),
            cam_azimuth=float(args.cam_azimuth),
            record_path=None if args.skip_video else raw_path,
        )
        traj.motion_path = motion_path
        traj.source_controller = str(args.controller_type)
        traj.source_checkpoint = str(args.checkpoint_path)
        save_muscle_trajectory(traj_path, traj)

        if not args.skip_video:
            encode_web_mp4(raw_path, video_path, remove_src=True)
            print(f"Saved muscle replay video: {video_path}")

        meta = {
            "motion_path": motion_path,
            "controller_type": args.controller_type,
            "checkpoint_path": args.checkpoint_path,
            "replay_mode": "muscle_closed_loop",
            "description": (
                "Official full-muscle policy closed-loop rollout: policy actions drive all muscle "
                "actuators in MuJoCo physics (not kinematic joint replay)."
            ),
            "muscle_trajectory_npz": str(traj_path),
            "muscle_replay_video": str(video_path) if not args.skip_video else None,
            "n_frames": int(traj.n_frames),
            "duration_s": float(traj.duration_s),
            "dt": float(traj.dt),
            "policy_action_dim": int(traj.policy_action.shape[-1]),
            "actuator_ctrl_dim": int(traj.actuator_ctrl.shape[-1]),
            "episode_return": float(episode_return),
            "done_count": int(done_count),
            "distill_metadata": metadata,
        }
        write_replay_meta(out_dir / "replay_meta.json", meta)
        print(
            f"  frames={traj.n_frames} return={episode_return:.3f} done_count={done_count} "
            f"npz={traj_path}"
        )
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

    records = []
    for motion in motions:
        records.append(record_one(args, motion))

    if len(records) > 1:
        summary_dir = repo_root() / "outputs" / "replay" / "muscle_mimic" / args.controller_type / "replay_four"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_replay_meta(
            summary_dir / "summary.json",
            {"controller_type": args.controller_type, "checkpoint_path": args.checkpoint_path, "motions": records},
        )
        print(f"Wrote summary: {summary_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
