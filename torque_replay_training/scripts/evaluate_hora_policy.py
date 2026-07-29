#!/usr/bin/env python3
"""Evaluate a HORA checkpoint or zero-residual baseline on vector replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from torque_replay_training.hora.models import ActorCritic, RunningMeanStd
from torque_replay_training.hora.ppo import load_hora_config
from torque_replay_training.hora_env import (
    HoraReplayEnvConfig,
    HoraTorqueReplayVecEnv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--group", choices=("train", "validation"), default="validation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device")
    parser.add_argument("--limit-datasets", type=int, default=0)
    args = parser.parse_args()
    ppo, environment_raw, units = load_hora_config(args.config)
    device = args.device or ppo.device
    environment_raw["num_envs"] = ppo.num_actors
    environment = HoraReplayEnvConfig(
        **{
            key: tuple(value) if key in {"residual_limits", "torque_limits"} else value
            for key, value in environment_raw.items()
        }
    )
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir)
    paths = [data_dir / row["dataset_basename"] for row in split[args.group]]
    if args.limit_datasets:
        paths = paths[: args.limit_datasets]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {args.group} datasets: {missing[:5]}")

    completed = 0
    falls = 0
    lengths = np.zeros(ppo.num_actors, dtype=np.int32)
    completed_lengths: list[int] = []
    metric_values: dict[str, list[float]] = {}
    with HoraTorqueReplayVecEnv(
        args.model,
        paths,
        environment,
        device=device,
        seed=ppo.seed + 10_000,
    ) as env:
        policy_model = normalizer = None
        if args.policy:
            checkpoint = torch.load(
                args.policy,
                map_location=device,
                weights_only=False,
            )
            policy_model = ActorCritic(
                env.observation_size,
                env.action_size,
                units,
                ppo.initial_log_std,
            ).to(device)
            policy_model.load_state_dict(checkpoint["model"])
            policy_model.eval()
            normalizer = RunningMeanStd((env.observation_size,)).to(device)
            normalizer.load_state_dict(checkpoint["running_mean_std"])
            normalizer.eval()
        observations = env.reset()
        while completed < args.episodes:
            if policy_model is None:
                actions = torch.zeros(
                    (env.num_envs, env.action_size),
                    dtype=torch.float32,
                    device=env.device,
                )
            else:
                normalized = normalizer(observations)
                actions = policy_model.act_inference(normalized).clamp(-1.0, 1.0)
            observations, _rewards, dones, info = env.step(actions)
            lengths += 1
            for name, value in info.items():
                if name in {"fell", "time_outs", "end_of_data"}:
                    continue
                metric_values.setdefault(name, []).extend(
                    value.detach().cpu().float().tolist()
                )
            for actor in dones.nonzero(as_tuple=False).reshape(-1).tolist():
                if completed >= args.episodes:
                    break
                falls += int(info["fell"][actor])
                completed_lengths.append(int(lengths[actor]))
                lengths[actor] = 0
                completed += 1
    report = {
        "mode": "policy" if args.policy else "healthy_baseline",
        "episodes": completed,
        "falls": falls,
        "fall_rate": falls / completed,
        "mean_episode_length": float(np.mean(completed_lengths)),
        "policy": args.policy,
        **{
            f"mean_{name}": float(np.mean(values))
            for name, values in metric_values.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
