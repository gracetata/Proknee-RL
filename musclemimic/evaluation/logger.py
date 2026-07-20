"""Persistence helpers for locomotion evaluation."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from musclemimic.evaluation.types import EvalConfig, MotionResult, RolloutBuffer


def safe_motion_name(motion_path: str) -> str:
    return motion_path.replace("/", "_").replace("\\", "_")


def make_timestamped_eval_dir(base: Path) -> Path:
    now = datetime.now()
    out = base / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_config_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def save_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


def save_rollout_npz(path: Path, buffer: RolloutBuffer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **buffer.to_npz_dict())


def write_motion_artifacts(
    *,
    eval_config: EvalConfig,
    motion_dir: Path,
    buffer: RolloutBuffer,
    metrics: dict[str, Any],
    video_src: Path | None = None,
) -> None:
    motion_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(motion_dir / "metrics.json", metrics)
    save_rollout_npz(motion_dir / "rollout_data.npz", buffer)
    if video_src is not None and video_src.is_file():
        dest = motion_dir / video_src.name
        if video_src.resolve() != dest.resolve():
            shutil.copy2(video_src, dest)


def build_run_config_payload(eval_config: EvalConfig, cli_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "cli": cli_args,
        "eval_config": {
            "config_name": eval_config.config_name,
            "checkpoint_path": eval_config.checkpoint_path,
            "controller_type": eval_config.controller_type,
            "env_type": eval_config.env_type,
            "eval_seed": eval_config.eval_seed,
            "n_steps": eval_config.n_steps,
            "prosthesis_control_mode": eval_config.prosthesis_control_mode,
            "prosthesis_controller": eval_config.prosthesis_controller,
            "eval_force": eval_config.eval_force,
            "git_commit": eval_config.git_commit,
        },
        "extra": eval_config.extra,
    }


def motion_result_paths(eval_dir: Path, motion_path: str) -> Path:
    return eval_dir / "per_motion" / safe_motion_name(motion_path)
