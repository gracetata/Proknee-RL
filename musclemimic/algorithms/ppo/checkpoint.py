"""
Checkpoint loading utilities for PPO.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import open_dict

from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager
from musclemimic.algorithms.common.dataclasses import TrainState


def create_agent_state_from_orbax(orbax_data: dict[str, Any]) -> SimpleNamespace:
    """
    Create agent state from orbax checkpoint data.

    Args:
        orbax_data: dict with params, opt_state, step, run_stats

    Returns:
        SimpleNamespace with train_state attribute
    """
    params = orbax_data.get("params", {})
    opt_state = orbax_data.get("opt_state", {})
    step = orbax_data.get("step", 0)
    run_stats = orbax_data.get("run_stats", {})

    class _DummyModule:
        def apply(self, *args, **kwargs):
            raise NotImplementedError("dummy module for checkpoint loading")

    ts = TrainState(
        apply_fn=_DummyModule().apply,
        params=params,
        tx=None,
        opt_state=opt_state,
        step=step,
        run_stats=run_stats or {},
    )

    agent_state = SimpleNamespace()
    agent_state.train_state = ts
    return agent_state


def load_checkpoint_for_resume(
    checkpoint_path: str,
    agent_conf: Any,
) -> tuple[SimpleNamespace, dict[str, int]]:
    """
    Load checkpoint and prepare resume info.

    Args:
        checkpoint_path: path to checkpoint
        agent_conf: agent configuration (config will be modified for lr override)

    Returns:
        (loaded_state, resume_info dict)
    """
    checkpoint_path_obj = Path(checkpoint_path)

    manager = UnifiedCheckpointManager(
        checkpoint_dir=str(checkpoint_path_obj.parent),
        max_to_keep=5,
    )

    try:
        (_, loaded_state_data), metadata = manager.load_checkpoint(checkpoint_path)
        loaded_state = create_agent_state_from_orbax(loaded_state_data)

        # inject lr override if available
        cur_lr = getattr(metadata, "learning_rate", None)
        if cur_lr is not None and cur_lr > 0:
            try:
                with open_dict(agent_conf.config):
                    agent_conf.config.experiment.resume_lr_override = float(cur_lr)
            except Exception:
                pass

        def _meta_int(value, default: int) -> int:
            if value is None:
                return default
            return int(value)

        resume_info = {
            "update_number": _meta_int(getattr(metadata, "update_number", None), -1),
            "global_timestep": _meta_int(getattr(metadata, "global_timestep", None), -1),
            "target_global_timestep": _meta_int(getattr(metadata, "target_global_timestep", None), 0),
            "num_envs": int(getattr(metadata, "num_envs", -1) or -1),
            "num_steps": int(getattr(metadata, "num_steps", -1) or -1),
            "num_minibatches": int(getattr(metadata, "num_minibatches", -1) or -1),
            "update_epochs": int(getattr(metadata, "update_epochs", -1) or -1),
        }

        return loaded_state, resume_info
    finally:
        manager.close()
