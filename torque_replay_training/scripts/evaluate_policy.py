#!/usr/bin/env python3
"""Load a trained residual checkpoint and run deterministic MuJoCo episodes."""

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
    parser.add_argument("--policy", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    _ppo_config, replay_config = load_training_config(args.config)
    model, params, metadata = load_policy_checkpoint(args.policy)
    returns = []
    lengths = []
    falls = 0
    with TorqueReplayEnv(args.dataset, args.checkpoint, replay_config) as env:
        if int(metadata["observation_size"]) != env.observation_size:
            raise ValueError("policy/environment observation sizes differ")
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode)
            episode_return = 0.0
            length = 0
            while True:
                mean, _log_std, _value = model.apply({"params": params}, jnp.asarray(obs)[None, :])
                action = np.asarray(jnp.tanh(mean[0]), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                length += 1
                if terminated or truncated:
                    falls += int(info["fell"])
                    break
            returns.append(episode_return)
            lengths.append(length)
    print(
        json.dumps(
            {
                "episodes": args.episodes,
                "mean_return": float(np.mean(returns)),
                "mean_length": float(np.mean(lengths)),
                "falls": falls,
                "policy": args.policy,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
