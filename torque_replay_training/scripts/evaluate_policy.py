#!/usr/bin/env python3
"""Evaluate a zero-residual baseline or trained policy on deterministic episodes."""

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import numpy as np

import _bootstrap  # noqa: F401

from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.ppo import load_policy_checkpoint, load_training_config
from torque_replay_training.replay_env import TorqueReplayEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, nargs="+")
    parser.add_argument("--policy", help="Omit to evaluate the zero-residual healthy baseline")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    _ppo_config, replay_config = load_training_config(args.config)
    model = params = metadata = None
    if args.policy:
        model, params, metadata = load_policy_checkpoint(args.policy)
    returns = []
    lengths = []
    falls = 0
    timeouts = 0
    end_of_data = 0
    metric_names = (
        "healthy_pos_rms",
        "healthy_vel_rms",
        "prosthesis_pos_rms",
        "prosthesis_vel_rms",
        "root_height",
        "root_up_z",
        "residual_norm",
        "action_norm",
        "action_rate_norm",
        "command_norm",
    )
    metrics = {name: [] for name in metric_names}
    with TorqueReplayEnv(args.dataset, args.checkpoint, replay_config) as env:
        if metadata and int(metadata["observation_size"]) != env.observation_size:
            raise ValueError("policy/environment observation sizes differ")
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode)
            episode_return = 0.0
            length = 0
            while True:
                if model is None:
                    action = np.zeros(env.action_size, dtype=np.float32)
                else:
                    mean, _log_std, _value = model.apply(
                        {"params": params}, jnp.asarray(obs)[None, :]
                    )
                    action = np.asarray(jnp.tanh(mean[0]), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                length += 1
                for name in metric_names:
                    metrics[name].append(float(info[name]))
                if terminated or truncated:
                    falls += int(info["fell"])
                    timeouts += int(info["timeout"])
                    end_of_data += int(info["end_of_data"])
                    break
            returns.append(episode_return)
            lengths.append(length)
    print(
        json.dumps(
            {
                "mode": "policy" if args.policy else "healthy_baseline",
                "episodes": args.episodes,
                "mean_return": float(np.mean(returns)),
                "mean_length": float(np.mean(lengths)),
                "falls": falls,
                "fall_rate": float(falls / args.episodes),
                "timeouts": timeouts,
                "end_of_data": end_of_data,
                **{
                    f"mean_{name}": float(np.mean(values))
                    for name, values in metrics.items()
                },
                "policy": args.policy,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
