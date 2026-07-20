#!/usr/bin/env python
"""Recompute CSV summaries and dataset plots from an existing eval directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musclemimic.evaluation.metrics import flatten_metrics_for_csv
from musclemimic.evaluation.motion_taxonomy import classify_motion_type
from musclemimic.evaluation.plotter import plot_dataset_summary
from musclemimic.evaluation.report import (
    write_failed_motions_csv,
    write_metrics_per_motion_csv,
    write_metrics_summary_csv,
)
from musclemimic.evaluation.types import MotionResult


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--save_plots", action="store_true", default=True)
    return p.parse_args()


def load_results(eval_dir: Path) -> list[MotionResult]:
    per_motion = eval_dir / "per_motion"
    results: list[MotionResult] = []
    if not per_motion.is_dir():
        raise FileNotFoundError(f"Missing per_motion directory: {per_motion}")
    for motion_dir in sorted(per_motion.iterdir()):
        if not motion_dir.is_dir():
            continue
        metrics_path = motion_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        motion_path = str(metrics.get("motion_path", motion_dir.name))
        success = bool(metrics.get("official_imitation", {}).get("success", {}).get("value", 0))
        results.append(
            MotionResult(
                motion_path=motion_path,
                motion_type=metrics.get("motion_type") or classify_motion_type(motion_path),
                safe_name=motion_dir.name,
                success=success,
                metrics=metrics,
                output_dir=motion_dir,
            )
        )
    return results


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    results = load_results(eval_dir)
    write_metrics_per_motion_csv(eval_dir / "metrics_per_motion.csv", results)
    write_failed_motions_csv(eval_dir / "failed_motions.csv", results)
    rows = []
    for r in results:
        row = flatten_metrics_for_csv(r.metrics)
        row.update({"motion_path": r.motion_path, "motion_type": r.motion_type, "success": int(r.success)})
        rows.append(row)
    write_metrics_summary_csv(eval_dir / "metrics_summary.csv", rows)
    if args.save_plots:
        plot_dataset_summary(rows, eval_dir / "summary_plots")
    print(f"Summarized {len(results)} motions -> {eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
