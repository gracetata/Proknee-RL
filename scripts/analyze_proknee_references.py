#!/usr/bin/env python
"""Summarize phase balance for cached retargeted references."""

from __future__ import annotations

import argparse
from pathlib import Path

from musclemimic.proknee.reference_dataset import summarize_reference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        default="/home/user/Workspace/musclemimic/data/caches/AMASS/MyoFullBody/gmr",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(Path(args.cache_root).glob("**/*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"No .npz files under {args.cache_root}")
        return 1

    totals = {"frames": 0, "static": 0.0, "dynamic": 0.0, "start_stop": 0.0, "high_dynamic": 0.0}
    for path in paths:
        stats = summarize_reference(path)
        totals["frames"] += stats.n_frames
        totals["static"] += stats.static_frac * stats.n_frames
        totals["dynamic"] += stats.dynamic_frac * stats.n_frames
        totals["start_stop"] += stats.start_stop_frac * stats.n_frames
        totals["high_dynamic"] += stats.high_dynamic_frac * stats.n_frames
        print(
            f"{stats.n_frames:5d} "
            f"static={stats.static_frac:5.2%} dynamic={stats.dynamic_frac:5.2%} "
            f"start_stop={stats.start_stop_frac:5.2%} high={stats.high_dynamic_frac:5.2%} "
            f"{path}"
        )

    n = max(totals["frames"], 1)
    print("\nAggregate:")
    print(f"  files={len(paths)} frames={totals['frames']}")
    print(f"  static={totals['static']/n:5.2%}")
    print(f"  dynamic={totals['dynamic']/n:5.2%}")
    print(f"  start_stop={totals['start_stop']/n:5.2%}")
    print(f"  high_dynamic={totals['high_dynamic']/n:5.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
