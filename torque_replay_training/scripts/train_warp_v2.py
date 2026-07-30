#!/usr/bin/env python3
"""Train the independent prosthesis policy in MuJoCo Warp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from torque_replay_training.warp_v2.config import load_training_config
from torque_replay_training.warp_v2.env import WarpReplayEnv
from torque_replay_training.warp_v2.ppo import WarpHoraPPO
from torque_replay_training.warp_v2.replay import PackedReplay


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _datasets(split_path: Path, data_dir: Path, group: str) -> list[Path]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    paths = [data_dir / row["dataset_basename"] for row in split[group]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {group} compact data: {missing[:5]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group", choices=("train", "validation"), default="train")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-agent-steps", type=int)
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    split_path = Path(args.split).resolve()
    manifest_path = Path(args.manifest).resolve()
    model_path = Path(args.model).resolve()
    config = load_training_config(config_path)
    environment = config.environment
    ppo = config.ppo
    if args.num_envs is not None:
        environment = replace(environment, num_envs=args.num_envs)
    if args.seed is not None:
        environment = replace(environment, seed=args.seed)
    if args.max_agent_steps is not None:
        ppo = replace(ppo, max_agent_steps=args.max_agent_steps)
    ppo.validate(environment.num_envs)

    paths = _datasets(
        split_path,
        Path(args.data_dir).resolve(),
        args.group,
    )
    if args.limit_datasets:
        paths = paths[: args.limit_datasets]
    if not paths:
        raise ValueError("no replay trajectories selected")
    repo_root = Path(__file__).resolve().parents[2]
    git_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "architecture": "warp-v2-independent-torque",
        "framework": "MuJoCo Warp + HORA-derived PyTorch PPO",
        "behavior_cloning": False,
        "imitation_reward": False,
        "final_replay_beta": 0.0,
        "git_head": git_head,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "split": str(split_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "split_group": args.group,
        "datasets": [str(path) for path in paths],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_policy": "only physical GPU 5",
    }
    replay = PackedReplay.load(paths, device=ppo.device)
    with WarpReplayEnv(
        model_path,
        replay,
        environment,
        config.replay_annealing,
        config.reward,
        device=ppo.device,
    ) as env:
        print(
            json.dumps(
                {
                    "status": "initialized",
                    "device": str(env.device),
                    "num_envs": env.num_envs,
                    "motions": replay.n_motions,
                    "actor_observation_size": env.actor_observation_size,
                    "critic_observation_size": env.critic_observation_size,
                    "action_size": env.action_size,
                    "replay_beta": env.beta,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        trainer = WarpHoraPPO(
            env,
            args.output,
            ppo,
            run_metadata=metadata,
        )
        result = trainer.train()
    print(json.dumps({"status": "complete", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
