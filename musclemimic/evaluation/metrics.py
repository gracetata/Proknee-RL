"""Metric orchestration."""

from __future__ import annotations

from typing import Any

from musclemimic.evaluation.force_metrics import (
    compute_grf_metrics,
    compute_reference_force_metrics,
    compute_torque_metrics,
)
from musclemimic.evaluation.kinematic_metrics import compute_kinematic_metrics
from musclemimic.evaluation.symmetry_metrics import compute_symmetry_metrics
from musclemimic.evaluation.types import RolloutBuffer


def compute_all_metrics(
    buffer: RolloutBuffer,
    *,
    meta: dict[str, Any] | None = None,
    force_reference_path: str | None = None,
) -> dict[str, Any]:
    meta = dict(meta or {})
    metrics: dict[str, Any] = {
        "motion_path": buffer.motion_path,
        "official_imitation": compute_kinematic_metrics(buffer, meta),
        "force": compute_grf_metrics(buffer, meta),
        "torque": compute_torque_metrics(buffer, meta),
        "symmetry": compute_symmetry_metrics(buffer, meta),
    }
    ref_metrics = compute_reference_force_metrics(buffer, force_reference_path)
    if ref_metrics:
        metrics["reference_force"] = ref_metrics
    return metrics


def flatten_metrics_for_csv(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dict for CSV rows."""

    flat: dict[str, Any] = {}

    def walk(obj, path: str):
        if isinstance(obj, dict) and "value" in obj and len(obj) <= 3:
            flat[path.rstrip(".")] = obj.get("value")
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}{k}.")
        else:
            flat[path.rstrip(".")] = obj

    for key, val in metrics.items():
        if key in {"motion_path", "motion_type", "controller_name", "env_type", "checkpoint_path", "success", "done_reason"}:
            flat[key] = val
            continue
        walk(val, f"{key}.")
    return flat
