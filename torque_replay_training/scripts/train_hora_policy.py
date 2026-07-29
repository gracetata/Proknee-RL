#!/usr/bin/env python3
"""Train the prosthesis residual with the vendored HORA PyTorch PPO core."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import _bootstrap  # noqa: F401

from torque_replay_training.hora import HoraPPO, load_hora_config
from torque_replay_training.hora_env import (
    HoraReplayEnvConfig,
    HoraTorqueReplayVecEnv,
)


def _datasets(split_path: Path, data_dir: Path, group: str) -> list[Path]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    paths = [data_dir / row["dataset_basename"] for row in split[group]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {group} datasets: {missing[:5]}")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group", choices=("train", "validation"), default="train")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-agent-steps", type=int)
    parser.add_argument("--limit-datasets", type=int, default=0)
    args = parser.parse_args()
    ppo_config, environment_raw, units = load_hora_config(args.config)
    if args.seed is not None:
        ppo_config = replace(ppo_config, seed=args.seed)
    if args.device:
        ppo_config = replace(ppo_config, device=args.device)
    if args.max_agent_steps is not None:
        ppo_config = replace(ppo_config, max_agent_steps=args.max_agent_steps)
    environment_raw["num_envs"] = ppo_config.num_actors
    environment = HoraReplayEnvConfig(
        **{
            key: tuple(value) if key in {"residual_limits", "torque_limits"} else value
            for key, value in environment_raw.items()
        }
    )
    split_path = Path(args.split).resolve()
    paths = _datasets(split_path, Path(args.data_dir).resolve(), args.group)
    if args.limit_datasets:
        paths = paths[: args.limit_datasets]
    git_head = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "framework": "HORA-derived PyTorch PPO",
        "hora_source_commit": "410d95824dd28b198f6d6910cb4c09330a6303be",
        "git_head": git_head,
        "config": str(Path(args.config).resolve()),
        "split": str(split_path),
        "compact_manifest": (
            str(Path(args.manifest).resolve()) if args.manifest else None
        ),
        "compact_manifest_sha256": (
            _sha256(Path(args.manifest).resolve()) if args.manifest else None
        ),
        "split_group": args.group,
        "datasets": [str(path) for path in paths],
        "model": str(Path(args.model).resolve()),
    }
    with HoraTorqueReplayVecEnv(
        args.model,
        paths,
        environment,
        device=ppo_config.device,
        seed=ppo_config.seed,
    ) as env:
        print(
            json.dumps(
                {
                    "framework": metadata["framework"],
                    "device": str(env.device),
                    "num_envs": env.num_envs,
                    "num_threads": env.num_threads,
                    "datasets": len(env.datasets),
                    "observation_size": env.observation_size,
                    "action_size": env.action_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        trainer = HoraPPO(
            env,
            args.output,
            ppo_config,
            units,
            run_metadata=metadata,
        )
        result = trainer.train()
    print(json.dumps({"status": "complete", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
