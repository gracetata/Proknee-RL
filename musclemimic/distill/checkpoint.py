"""Checkpoint export helpers for prosthesis distillation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import csv
import json

import jax
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import TrainState
from musclemimic.algorithms.common.checkpoint_manager import CheckpointMetadata, UnifiedCheckpointManager


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def save_loss_curve(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_param_npz(path: str | Path, params: Any, run_stats: Any) -> None:
    flat: dict[str, np.ndarray] = {}
    for prefix, tree in (("params", params), ("run_stats", run_stats)):
        leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
        flat[f"{prefix}.__treedef__"] = np.asarray(str(treedef))
        for path_items, value in leaves:
            key = ".".join(str(getattr(item, "key", item)) for item in path_items)
            flat[f"{prefix}.{key}"] = np.asarray(value)
    np.savez_compressed(path, **flat)


def save_torch_sidecar(path: str | Path, params: Any, run_stats: Any, metadata: dict) -> bool:
    try:
        import torch
    except Exception:
        return False
    payload = {
        "params": tree_to_numpy(params),
        "run_stats": tree_to_numpy(run_stats),
        "metadata": metadata,
    }
    torch.save(payload, path)
    return True


def save_distilled_checkpoint(
    *,
    checkpoint_dir: str | Path,
    agent_conf,
    params,
    run_stats,
    opt_state,
    step: int,
    metadata: dict,
) -> str:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_state = TrainState(
        apply_fn=agent_conf.network.apply,
        params=params,
        tx=agent_conf.tx,
        opt_state=opt_state,
        # PPO fine-tune should start a fresh optimizer/update schedule from the distilled weights.
        step=0,
        run_stats=run_stats,
    )
    agent_state = SimpleNamespace(train_state=train_state)
    md = CheckpointMetadata(
        step=int(step),
        update_number=0,
        global_timestep=0,
        # Finetune should set its own budget from experiment.total_timesteps.
        target_global_timestep=0,
        learning_rate=float(metadata.get("learning_rate", 0.0)),
        num_envs=int(getattr(agent_conf.config.experiment, "num_envs", 1)),
        num_steps=int(getattr(agent_conf.config.experiment, "num_steps", 1)),
        num_minibatches=int(getattr(agent_conf.config.experiment, "num_minibatches", 1)),
        update_epochs=int(getattr(agent_conf.config.experiment, "update_epochs", 1)),
        backend="distill",
        env_name=str(agent_conf.config.experiment.env_params.get("env_name", "")),
    )
    manager = UnifiedCheckpointManager(str(checkpoint_dir), max_to_keep=2, async_save=False)
    try:
        path = manager.save_checkpoint(0, agent_conf, agent_state, md)
        manager.wait_until_finished()
    finally:
        manager.close()

    sidecar_meta = {
        **metadata,
        "orbax_checkpoint": path,
        "config": OmegaConf.to_container(agent_conf.config, resolve=True, throw_on_missing=False),
    }
    (checkpoint_dir / "distilled_metadata.json").write_text(json.dumps(sidecar_meta, indent=2, default=str), encoding="utf-8")
    save_param_npz(checkpoint_dir / "distilled_policy.npz", params, run_stats)
    save_torch_sidecar(checkpoint_dir / "distilled_policy.pt", params, run_stats, sidecar_meta)
    latest_alias = checkpoint_dir / "checkpoint_distilled"
    if not latest_alias.exists():
        try:
            latest_alias.symlink_to(Path(path).name, target_is_directory=True)
        except OSError:
            pass
    return str(latest_alias if latest_alias.exists() else path)
