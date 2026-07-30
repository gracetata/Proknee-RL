#!/usr/bin/env python3
"""Train the prosthesis with official MJLAB ManagerBasedRlEnv and RSL-RL PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from mjlab_common import replay_paths
from torque_replay_training.mjlab_task import make_env_cfg, make_runner_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-xml", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--group", choices=("train", "validation"), default="train")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--episode-steps", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-root", default="torque_replay_training/outputs/mjlab")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--healthy-kp", type=float, default=50.0)
    parser.add_argument("--healthy-kd", type=float, default=5.0)
    parser.add_argument("--healthy-correction-limit", type=float, default=300.0)
    parser.add_argument("--root-position-kp", type=float, default=5000.0)
    parser.add_argument("--root-velocity-kd", type=float, default=1000.0)
    parser.add_argument("--root-force-limit", type=float, default=10000.0)
    parser.add_argument("--root-orientation-kp", type=float, default=1000.0)
    parser.add_argument("--root-angular-velocity-kd", type=float, default=100.0)
    parser.add_argument("--root-torque-limit", type=float, default=1000.0)
    args = parser.parse_args()

    env_cfg = make_env_cfg(
        replay_paths=replay_paths(args.split, args.data_dir, args.group),
        model_xml=args.model_xml,
        model_root=args.model_root,
        num_envs=args.num_envs,
        episode_steps=args.episode_steps,
        seed=args.seed,
        healthy_kp=args.healthy_kp,
        healthy_kd=args.healthy_kd,
        healthy_correction_limit=args.healthy_correction_limit,
        root_position_kp=args.root_position_kp,
        root_velocity_kd=args.root_velocity_kd,
        root_force_limit=args.root_force_limit,
        root_orientation_kp=args.root_orientation_kp,
        root_angular_velocity_kd=args.root_angular_velocity_kd,
        root_torque_limit=args.root_torque_limit,
    )
    runner_cfg = make_runner_cfg(max_iterations=args.max_iterations, seed=args.seed)
    if args.run_name:
        runner_cfg.run_name = args.run_name
    suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if runner_cfg.run_name:
        suffix += f"_{runner_cfg.run_name}"
    log_dir = (
        Path(args.log_root).resolve()
        / runner_cfg.experiment_name
        / suffix
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(runner_cfg))

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
        wrapped,
        asdict(runner_cfg),
        str(log_dir),
        args.device,
    )
    try:
        runner.learn(
            num_learning_iterations=runner_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
