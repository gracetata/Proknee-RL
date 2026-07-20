"""CSV report generation for locomotion evaluation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from musclemimic.evaluation.metrics import flatten_metrics_for_csv
from musclemimic.evaluation.types import MotionResult


def _collect_columns(results: list[MotionResult]) -> list[str]:
    keys: set[str] = set()
    for r in results:
        flat = flatten_metrics_for_csv(r.metrics)
        keys.update(flat.keys())
    base = ["motion_path", "motion_type", "success", "done_reason", "error_message"]
    rest = sorted(k for k in keys if k not in base)
    return base + rest


def write_metrics_per_motion_csv(path: Path, results: list[MotionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _collect_columns(results)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = flatten_metrics_for_csv(r.metrics)
            row.update(
                {
                    "motion_path": r.motion_path,
                    "motion_type": r.motion_type,
                    "success": r.success,
                    "done_reason": r.metrics.get("done_reason", r.metrics.get("official_imitation", {}).get("done_reason")),
                    "error_message": r.error_message or "",
                }
            )
            if isinstance(row.get("done_reason"), dict):
                row["done_reason"] = row["done_reason"].get("value")
            writer.writerow(row)


def write_failed_motions_csv(path: Path, results: list[MotionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = [r for r in results if r.error_message or not r.success]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["motion_path", "motion_type", "success", "done_reason", "error_message"])
        writer.writeheader()
        for r in failed:
            done = r.metrics.get("done_reason")
            if isinstance(done, dict):
                done = done.get("value")
            writer.writerow(
                {
                    "motion_path": r.motion_path,
                    "motion_type": r.motion_type,
                    "success": r.success,
                    "done_reason": done,
                    "error_message": r.error_message or "",
                }
            )


def _numeric_values(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        v = row.get(key)
        if v is None or v == "":
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def write_metrics_summary_csv(path: Path, per_motion_rows: list[dict[str, Any]]) -> None:
    """Aggregate mean/std/median/success_rate by motion_type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in per_motion_rows:
        by_type[str(row.get("motion_type", "unknown"))].append(row)

    metric_keys = set()
    for row in per_motion_rows:
        metric_keys.update(k for k in row if k not in {"motion_path", "motion_type", "error_message"})

    fieldnames = ["motion_type", "count", "success_rate"]
    stat_cols = []
    for key in sorted(metric_keys):
        if key in {"motion_path", "motion_type", "success", "done_reason", "error_message"}:
            continue
        for stat in ("mean", "std", "median"):
            col = f"{key}.{stat}"
            fieldnames.append(col)
            stat_cols.append((key, stat))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for motion_type, rows in sorted(by_type.items()):
            out: dict[str, Any] = {
                "motion_type": motion_type,
                "count": len(rows),
                "success_rate": float(np.mean([1.0 if r.get("success") in (1, True, "1") else 0.0 for r in rows])),
            }
            for key, stat in stat_cols:
                vals = _numeric_values(rows, key)
                if not vals:
                    out[f"{key}.{stat}"] = ""
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                if stat == "mean":
                    out[f"{key}.{stat}"] = float(np.mean(arr))
                elif stat == "std":
                    out[f"{key}.{stat}"] = float(np.std(arr))
                else:
                    out[f"{key}.{stat}"] = float(np.median(arr))
            writer.writerow(out)
