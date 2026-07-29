#!/usr/bin/env python3
"""Verify that every dataset referenced by a flat-walking split is present."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from torque_replay_training.paths import NEW_PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default=str(NEW_PROJECT_ROOT / "configs/flat_walk_split.json"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(NEW_PROJECT_ROOT / "data/fullbody_all_v3"),
    )
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir)
    rows = [row for group in ("train", "validation") for row in split[group]]
    missing = []
    wrong_size = []
    for row in rows:
        path = data_dir / row["dataset_basename"]
        if not path.is_file():
            missing.append(path.name)
        elif path.stat().st_size != int(row["bytes"]):
            wrong_size.append(path.name)
    report = {
        "requested": len(rows),
        "present": len(rows) - len(missing),
        "valid_size": len(rows) - len(missing) - len(wrong_size),
        "bytes": sum(
            (data_dir / row["dataset_basename"]).stat().st_size
            for row in rows
            if (data_dir / row["dataset_basename"]).is_file()
        ),
        "missing": missing,
        "wrong_size": wrong_size,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing or wrong_size:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
