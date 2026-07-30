#!/usr/bin/env python3
"""Gate training on exact replay in the actual official MJLAB backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_common import replay_paths
from torque_replay_training.mjlab_task import make_env_cfg
from torque_replay_training.mjlab_task.mdp import replay_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-xml", required=True)
    parser.add_argument("--model-root")
    parser.add_argument("--group", choices=("train", "validation"), default="validation")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--healthy-kp", type=float, default=0.0)
    parser.add_argument("--healthy-kd", type=float, default=0.0)
    parser.add_argument("--healthy-correction-limit", type=float, default=0.0)
    parser.add_argument("--root-position-kp", type=float, default=0.0)
    parser.add_argument("--root-velocity-kd", type=float, default=0.0)
    parser.add_argument("--root-force-limit", type=float, default=0.0)
    parser.add_argument("--root-orientation-kp", type=float, default=0.0)
    parser.add_argument("--root-angular-velocity-kd", type=float, default=0.0)
    parser.add_argument("--root-torque-limit", type=float, default=0.0)
    parser.add_argument("--max-fall-rate", type=float, default=0.0)
    parser.add_argument("--max-tracking-error", type=float, default=0.10)
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = make_env_cfg(
        replay_paths=replay_paths(args.split, args.data_dir, args.group),
        model_xml=args.model_xml,
        model_root=args.model_root,
        num_envs=args.num_envs,
        episode_steps=args.episode_steps,
        fixed_replay_beta=1.0,
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
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env.reset()
    actions = torch.zeros((args.num_envs, 4), device=args.device)
    falls = 0
    timeouts = 0
    episodes = 0
    observed_tracking_error = 0.0
    minimum_height = float("inf")
    minimum_up_z = float("inf")
    try:
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(actions)
            falls += int(terminated.count_nonzero())
            timeouts += int(truncated.count_nonzero())
            episodes += int((terminated | truncated).count_nonzero())
            term = replay_action(env)
            observed_tracking_error = max(
                observed_tracking_error,
                float(term.max_tracking_error.max()),
            )
            minimum_height = min(minimum_height, float(env.sim.data.qpos[:, 2].min()))
            from torque_replay_training.mjlab_task.mdp import root_up_z

            minimum_up_z = min(minimum_up_z, float(root_up_z(env).min()))
    finally:
        env.close()
    report = {
        "backend": "official MJLAB / MuJoCo Warp",
        "mujoco_version": env.metadata["mujoco_version"],
        "warp_version": env.metadata["warp_version"],
        "mode": (
            "assisted-replay"
            if any(
                (
                    args.healthy_kp,
                    args.healthy_kd,
                    args.root_position_kp,
                    args.root_velocity_kd,
                    args.root_orientation_kp,
                    args.root_angular_velocity_kd,
                )
            )
            else "unassisted-replay"
        ),
        "replay_beta": 1.0,
        "assistance_scale": 1.0,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "episodes": episodes,
        "falls": falls,
        "timeouts": timeouts,
        "fall_rate": falls / max(1, episodes),
        "maximum_tracking_error": observed_tracking_error,
        "minimum_root_height": minimum_height,
        "minimum_root_up_z": minimum_up_z,
        "stabilizer": {
            "healthy_kp": args.healthy_kp,
            "healthy_kd": args.healthy_kd,
            "healthy_correction_limit": args.healthy_correction_limit,
            "root_position_kp": args.root_position_kp,
            "root_velocity_kd": args.root_velocity_kd,
            "root_force_limit": args.root_force_limit,
            "root_orientation_kp": args.root_orientation_kp,
            "root_angular_velocity_kd": args.root_angular_velocity_kd,
            "root_torque_limit": args.root_torque_limit,
        },
    }
    report["passed"] = (
        report["fall_rate"] <= args.max_fall_rate
        and observed_tracking_error <= args.max_tracking_error
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
