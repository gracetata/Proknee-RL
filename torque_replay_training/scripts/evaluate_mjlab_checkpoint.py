#!/usr/bin/env python3
"""Evaluate an official RSL-RL checkpoint with replay beta and assistance at zero."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from mjlab_common import replay_paths
from torque_replay_training.mjlab_task import make_env_cfg, make_runner_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-xml", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--group", choices=("train", "validation"), default="validation")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--episode-steps", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()

    env_cfg = make_env_cfg(
        replay_paths=replay_paths(args.split, args.data_dir, args.group),
        model_xml=args.model_xml,
        model_root=args.model_root,
        num_envs=args.num_envs,
        episode_steps=args.episode_steps,
        fixed_replay_beta=0.0,
    )
    runner_cfg = make_runner_cfg(max_iterations=1)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg), device=args.device)
    runner.load(str(Path(args.checkpoint).resolve()), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    obs = wrapped.get_observations()
    completed = 0
    falls = 0
    steps = 0
    reward_sum = 0.0
    try:
        while completed < args.episodes:
            with torch.no_grad():
                actions = policy(obs)
            obs, reward, dones, _ = wrapped.step(actions)
            count = int(dones.count_nonzero())
            if count:
                fallen = env.termination_manager.get_term("fallen")
                falls += int((fallen & dones.bool()).count_nonzero())
                completed += count
            reward_sum += float(reward.mean())
            steps += 1
    finally:
        wrapped.close()
    report = {
        "backend": "official MJLAB / MuJoCo Warp",
        "mode": "checkpoint-policy-only",
        "replay_beta": 0.0,
        "assistance_scale": 0.0,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episodes": completed,
        "falls": falls,
        "fall_rate": falls / max(1, completed),
        "environment_steps": steps,
        "mean_step_reward": reward_sum / max(1, steps),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
