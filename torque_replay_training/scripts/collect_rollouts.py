#!/usr/bin/env python3
"""Export and qualify a batch of full-body motions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import _bootstrap  # noqa: F401

from torque_replay_training.exporter import export_fullbody_rollout
from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.schema import SCHEMA_VERSION


def _safe_name(motion: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", motion).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", action="append", required=True, help="Repeat for every motion")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=0, help="0 requires the complete reference motion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true", help="Smoke/debug only")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for motion in args.motion:
        output = output_dir / f"{_safe_name(motion)}.npz"
        row = {"motion": motion, "output": str(output), "passed": False}
        try:
            dataset = export_fullbody_rollout(
                checkpoint_path=args.checkpoint,
                motion_path=motion,
                output_path=output,
                n_steps=args.steps,
                seed=args.seed,
                require_complete=not args.allow_incomplete,
            )
            row.update(
                {
                    "passed": bool(dataset.metadata["qualified_full_motion"] or args.allow_incomplete),
                    "qualified_full_motion": bool(dataset.metadata["qualified_full_motion"]),
                    "steps": dataset.n_steps,
                    "root_height_min": dataset.metadata["root_height_min"],
                    "root_up_min": dataset.metadata["root_up_min"],
                }
            )
        except Exception as error:  # keep the manifest useful when one clip fails
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": args.checkpoint,
        "production": not args.allow_incomplete and args.steps == 0,
        "all_passed": all(row["passed"] for row in rows),
        "rollouts": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "all_passed": manifest["all_passed"]}))
    if not manifest["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
