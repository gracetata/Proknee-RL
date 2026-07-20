#!/usr/bin/env python
"""Compute 6-dimension composite score and radar plot from evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from musclemimic.evaluation.composite_score import (
    build_composite_report,
    save_composite_report,
)
from musclemimic.evaluation.radar_plot import plot_composite_radar


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics_json", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--symmetry_summary", type=Path, default=None)
    p.add_argument("--torque_summary", type=Path, default=None)
    p.add_argument("--config-name", default="conf_eval_composite_score")
    p.add_argument("--radar_name", default="composite_score_radar.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = build_composite_report(
        args.metrics_json,
        config_name=args.config_name,
        symmetry_summary_path=args.symmetry_summary,
        torque_summary_path=args.torque_summary,
    )
    json_path, out_dir = save_composite_report(report, args.output_dir)
    radar_path = plot_composite_radar(report, out_dir / args.radar_name)

    print("=== Composite locomotion score ===")
    for dim_id, block in report["dimensions"].items():
        zh = block.get("label_zh", dim_id)
        print(f"  {zh} ({dim_id}): {block['score']:.1f}")
    print(f"  TOTAL: {report['total_score']:.1f} / 100")
    print(f"Wrote {json_path}")
    print(f"Wrote {radar_path}")


if __name__ == "__main__":
    main()
