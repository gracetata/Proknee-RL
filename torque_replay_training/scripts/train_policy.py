#!/usr/bin/env python3
"""Train the left prosthesis residual policy."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

import _bootstrap  # noqa: F401

from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.ppo import ResidualPPOTrainer, load_training_config
from torque_replay_training.replay_env import TorqueReplayEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, nargs="+", help="One or more qualified replay .npz files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--seed", type=int, help="Override ppo.seed from the YAML config")
    args = parser.parse_args()
    ppo_config, replay_config = load_training_config(args.config)
    if args.seed is not None:
        ppo_config = replace(ppo_config, seed=args.seed)
    with TorqueReplayEnv(
        args.dataset,
        args.checkpoint,
        replay_config,
        seed=ppo_config.seed,
    ) as env:
        print(
            json.dumps(
                {
                    "observation_size": env.observation_size,
                    "action_size": env.action_size,
                    "datasets": len(env.datasets),
                    "dataset_steps": [dataset.n_steps for dataset in env.datasets],
                    "physics_substeps": env.dataset.n_substeps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        trainer = ResidualPPOTrainer(env, ppo_config, args.output)
        result = trainer.train()
    print(json.dumps({"status": "complete", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
