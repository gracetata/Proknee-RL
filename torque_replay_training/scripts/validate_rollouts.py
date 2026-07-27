#!/usr/bin/env python3
"""Incrementally validate every dataset listed in a rollout manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.validation import validate_replay, validate_replay_batch


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    source_path = Path(args.manifest)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidates = [row for row in source.get("rollouts", []) if row.get("passed") is True]
    output_path = (
        Path(args.output)
        if args.output
        else source_path.with_name("validation_manifest.json")
    )
    previous: dict[str, Any] = {}
    if args.resume and output_path.is_file():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        previous = {
            row["motion"]: row
            for row in loaded.get("rollouts", [])
            if row.get("passed") is True
        }

    rows_by_motion = dict(previous)

    def write_progress() -> None:
        rows = [
            rows_by_motion[row["motion"]]
            for row in candidates
            if row["motion"] in rows_by_motion
        ]
        payload = {
            "source_manifest": str(source_path),
            "requested": len(candidates),
            "completed": sum(item.get("passed") is True for item in rows),
            "failed": sum(item.get("passed") is False for item in rows),
            "pending": len(candidates) - len(rows),
            "all_passed": len(rows) == len(candidates)
            and all(item.get("passed") is True for item in rows),
            "rollouts": rows,
        }
        _atomic_json(output_path, payload)

    write_progress()
    pending = [row for row in candidates if row["motion"] not in rows_by_motion]
    for chunk_start in range(0, len(pending), args.chunk_size):
        chunk = pending[chunk_start : chunk_start + args.chunk_size]
        try:
            results = validate_replay_batch(
                [row["output"] for row in chunk],
                args.checkpoint,
                steps=args.steps,
            )
            outcomes: list[dict[str, Any] | Exception] = results
        except Exception:
            # Isolate a corrupt or incompatible file instead of failing every
            # other dataset in the validation chunk.
            outcomes = []
            for source_row in chunk:
                try:
                    outcomes.append(
                        validate_replay(
                            source_row["output"],
                            args.checkpoint,
                            steps=args.steps,
                        )
                    )
                except Exception as error:
                    outcomes.append(error)

        for source_row, outcome in zip(chunk, outcomes, strict=True):
            motion = source_row["motion"]
            row = {
                "motion": motion,
                "dataset": source_row["output"],
                "passed": False,
            }
            if isinstance(outcome, Exception):
                row["error"] = f"{type(outcome).__name__}: {outcome}"
            else:
                row.update(outcome)
            rows_by_motion[motion] = row
            write_progress()
            print(json.dumps(row, sort_keys=True), flush=True)

    final = json.loads(output_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "validation_manifest": str(output_path),
                "all_passed": final["all_passed"],
                "completed": final["completed"],
                "failed": final["failed"],
            }
        )
    )
    if not final["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
