#!/usr/bin/env python
"""Record clean full-body official policy videos without ghost or debug overlays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import jax
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults, apply_terminal_cli_overrides
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import (
    align_agent_state,
    apply_temporal_params,
    load_checkpoint,
    setup_headless,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--record-path", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--cam-distance", type=float, default=3.5)
    parser.add_argument("--cam-elevation", type=float, default=0.0)
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--train-state-seed", type=int, default=0)
    parser.add_argument("--strict-termination", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--mujoco-viewer", action="store_true", default=False)
    parser.add_argument("--viser-viewer", action="store_true", default=False)
    parser.add_argument("--no-termination", action="store_true", default=True)
    parser.add_argument("--terminal-state-type", default=None)
    parser.add_argument("--mean-site-deviation-threshold", type=float, default=None)
    parser.add_argument("--root-deviation-threshold", type=float, default=None)
    parser.add_argument("--root-orientation-threshold", type=float, default=None)
    parser.add_argument("--root-site", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_headless(args)

    config, agent_state, _metadata = load_checkpoint(args.path)
    OmegaConf.set_struct(config, False)

    config.experiment.env_params = config.experiment.env_params.copy()
    env_params = config.experiment.env_params
    env_params["headless"] = args.no_render

    if "Mjx" in env_params.get("env_name", ""):
        env_params["env_name"] = env_params["env_name"].replace("Mjx", "")

    if "goal_params" in env_params:
        env_params["goal_params"]["visualize_goal"] = False
        env_params["goal_params"]["n_visual_geoms"] = 0

    if "th_params" not in env_params:
        env_params["th_params"] = {}
    env_params["th_params"]["random_start"] = False
    env_params["th_params"]["fixed_start_conf"] = [0, 0]
    env_params["th_params"]["start_from_random_step"] = False

    control_dt = apply_temporal_params(config)
    play_env_params = OmegaConf.to_container(env_params, resolve=True)
    apply_eval_terminal_defaults(play_env_params, config, args.strict_termination)
    apply_terminal_cli_overrides(play_env_params, args)

    task_params = config.experiment.task_factory.params
    task_params.amass_dataset_conf.rel_dataset_path = [args.motion_path]
    task_params.amass_dataset_conf.dataset_group = None

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    merged_params = {
        **play_env_params,
        **OmegaConf.to_container(task_params, resolve=True),
    }
    env = factory.make(**merged_params)

    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    if config.experiment.n_seeds > 1:
        agent_state = agent_state.replace(
            train_state=jax.tree.map(lambda x: x[args.train_state_seed], agent_state.train_state)
        )
    runner = PolicyRunner.from_agent_state(
        agent_conf,
        agent_state,
        env,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.record_path)), exist_ok=True)
    video_path = web_mp4_path(args.record_path)
    raw_path = video_path.parent / f".{video_path.stem}_raw.mp4"
    fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
    obs = env.reset()
    obs_policy = runner.reset_obs(obs)

    print(f"Recording clean video -> {video_path}")
    print(f"Motion: {args.motion_path} | steps={args.n_steps} | fps={fps}")
    done_count = 0
    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.cam_distance
    cam.elevation = args.cam_elevation
    cam.azimuth = args.cam_azimuth
    with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
        for step in range(args.n_steps):
            action, _value = runner.act(obs_policy)
            obs, _reward, _absorbing, done, info = env.step(action)
            obs_policy = runner.update_obs(obs)
            if bool(done):
                done_count += 1
            cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            renderer.update_scene(env.data, camera=cam)
            frame = renderer.render()
            writer.append_data(frame)
            if (step + 1) % 1000 == 0:
                substep = info.get("subtraj_step_no", "?") if isinstance(info, dict) else "?"
                traj_len = info.get("traj_len", "?") if isinstance(info, dict) else "?"
                print(f"step={step + 1}/{args.n_steps} traj={substep}/{traj_len} done_count={done_count}", flush=True)

    renderer.close()
    env.stop()
    encode_web_mp4(raw_path, video_path)
    print(f"Saved clean video: {video_path}")
    print(f"Done flags observed but ignored: {done_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
