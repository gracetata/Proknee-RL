#!/usr/bin/env python
"""Record split-action prosthesis policy replay."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless


DEFAULT_CHECKPOINT = (
    "/home/user/Workspace/musclemimic/musclemimic/outputs/split_action_prosthesis_distill/latest/checkpoints/"
    "checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "/home/user/Workspace/musclemimic/musclemimic/outputs/eval_split_action_prosthesis/replay_four"
DEFAULT_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--motion-path", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--n-steps", type=int, default=0, help="0 = full trajectory length")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--cam-distance", type=float, default=6.0)
    parser.add_argument("--cam-elevation", type=float, default=-20.0)
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def make_env_for_motion(config, motion_path: str):
    OmegaConf.set_struct(config, False)
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBodyProsthesisEnv"
    env_params["headless"] = True
    prosthesis = dict(env_params.get("prosthesis", {}) or {})
    prosthesis["enabled"] = True
    prosthesis["control_mode"] = "train_policy"
    env_params["prosthesis"] = prosthesis
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


def record_motion(args, config, agent_state, metadata, motion_path: str, out_dir: Path, control_dt: float) -> dict:
    env = make_env_for_motion(config, motion_path)
    try:
        n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
        agent_conf = PPOJax.init_agent_conf(env, config)
        aligned_state = align_agent_state(agent_state, agent_conf)
        runner = PolicyRunner.from_agent_state(
            agent_conf,
            aligned_state,
            env,
            deterministic=not args.stochastic,
            seed=args.seed,
        )
        fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
        safe_motion = motion_path.replace("/", "_")
        raw_path = out_dir / f".{safe_motion}_raw.mp4"
        video_path = web_mp4_path(out_dir / f"{safe_motion}_split_action_replay.mp4")

        renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = args.cam_distance
        cam.elevation = args.cam_elevation
        cam.azimuth = args.cam_azimuth

        obs = env.reset()
        obs_policy = runner.reset_obs(obs)
        episode_return = 0.0
        done_count = 0
        first_done_step = None
        with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
            for step in range(n_steps):
                action, _value = runner.act(obs_policy)
                obs, reward, _absorbing, done, _info = env.step(action)
                obs_policy = runner.update_obs(obs)
                episode_return += float(np.asarray(reward).item())
                if bool(done):
                    done_count += 1
                    if first_done_step is None:
                        first_done_step = step
                cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())
        renderer.close()
        encode_web_mp4(raw_path, video_path)
        meta = {
            "checkpoint": str(args.checkpoint),
            "motion_path": motion_path,
            "n_steps": n_steps,
            "fps": fps,
            "duration_s": float(n_steps * control_dt),
            "episode_return": episode_return,
            "done_count": done_count,
            "first_done_step": first_done_step,
            "video": str(video_path),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "distill_metadata": metadata,
        }
        (out_dir / f"{safe_motion}_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"{motion_path}: saved {video_path} return={episode_return:.3f} done_count={done_count}")
        return meta
    finally:
        env.stop()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))
    config, agent_state, metadata = load_checkpoint(args.checkpoint)
    control_dt = apply_temporal_params(config)
    motions = args.motion_path or DEFAULT_MOTIONS
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Recording {len(motions)} motion(s) -> {out_dir}")

    records = []
    for motion_path in motions:
        records.append(record_motion(args, config, agent_state, metadata, motion_path, out_dir, control_dt))
    summary = {
        "checkpoint": str(args.checkpoint),
        "run_name": run_name,
        "output_dir": str(out_dir),
        "motions": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
