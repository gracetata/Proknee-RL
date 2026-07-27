#!/usr/bin/env python3
"""Export and qualify a resumable batch of full-body motions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import _bootstrap  # noqa: F401

from torque_replay_training.exporter import FullbodyRolloutSession
from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.schema import SCHEMA_VERSION, TorqueReplayDataset


def _safe_name(motion: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", motion).strip("_")


def _read_motion_file(path: str | Path) -> list[str]:
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            rows.append(value)
    return rows


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_row(motion: str, output: Path) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    try:
        dataset = TorqueReplayDataset.load(output)
    except Exception:
        return None
    if dataset.motion_path != motion or not dataset.metadata.get("qualified_full_motion", False):
        return None
    return {
        "motion": motion,
        "output": str(output),
        "passed": True,
        "qualified_full_motion": True,
        "steps": dataset.n_steps,
        "root_height_min": dataset.metadata["root_height_min"],
        "root_up_min": dataset.metadata["root_up_min"],
        "resumed": True,
    }


def _rejected_row(motion: str, output: Path) -> dict[str, Any] | None:
    rejected = output.with_name(f"{output.stem}.rejected{output.suffix or '.npz'}")
    if not rejected.is_file():
        return None
    try:
        dataset = TorqueReplayDataset.load(rejected)
    except Exception:
        return None
    if dataset.motion_path != motion or dataset.metadata.get("qualified_full_motion", True):
        return None
    return {
        "motion": motion,
        "output": str(output),
        "rejected_output": str(rejected),
        "passed": False,
        "qualified_full_motion": False,
        "steps": dataset.n_steps,
        "root_height_min": dataset.metadata["root_height_min"],
        "root_up_min": dataset.metadata["root_up_min"],
        "error": "resumed previously rejected full-motion rollout",
        "resumed": True,
    }


def _manifest(
    *,
    checkpoint: str,
    rows_by_motion: dict[str, dict[str, Any]],
    motions: list[str],
    production: bool,
) -> dict[str, Any]:
    rows = [
        rows_by_motion.get(
            motion,
            {
                "motion": motion,
                "passed": False,
                "status": "pending",
            },
        )
        for motion in motions
    ]
    completed = sum(row.get("passed") is True for row in rows)
    failed = sum("error" in row for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint,
        "production": production,
        "all_passed": completed == len(motions),
        "requested": len(motions),
        "completed": completed,
        "failed": failed,
        "pending": len(motions) - completed - failed,
        "rollouts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", action="append", default=[], help="Repeat for every motion")
    parser.add_argument("--motion-file", action="append", default=[], help="One motion per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=0, help="0 requires the complete reference motion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Keep every existing qualified v3 file")
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="With --resume, recompute existing .rejected.npz audit files",
    )
    parser.add_argument("--allow-incomplete", action="store_true", help="Smoke/debug only")
    args = parser.parse_args()

    motions = list(args.motion)
    for motion_file in args.motion_file:
        motions.extend(_read_motion_file(motion_file))
    motions = _deduplicate(motions)
    if not motions:
        parser.error("provide at least one --motion or --motion-file")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    rows_by_motion: dict[str, dict[str, Any]] = {}
    production = not args.allow_incomplete and args.steps == 0

    pending = []
    for motion in motions:
        output = output_dir / f"{_safe_name(motion)}.npz"
        resumed = None
        if args.resume:
            resumed = _completed_row(motion, output)
            if resumed is None and not args.retry_rejected:
                resumed = _rejected_row(motion, output)
        if resumed is not None:
            rows_by_motion[motion] = resumed
            print(json.dumps(resumed, sort_keys=True), flush=True)
        else:
            pending.append(motion)

    _atomic_json(
        manifest_path,
        _manifest(
            checkpoint=args.checkpoint,
            rows_by_motion=rows_by_motion,
            motions=motions,
            production=production,
        ),
    )

    for chunk_start in range(0, len(pending), args.chunk_size):
        chunk = pending[chunk_start : chunk_start + args.chunk_size]
        try:
            session = FullbodyRolloutSession(
                checkpoint_path=args.checkpoint,
                motion_paths=chunk,
                seed=args.seed,
            )
        except Exception as error:
            for motion in chunk:
                row = {
                    "motion": motion,
                    "output": str(output_dir / f"{_safe_name(motion)}.npz"),
                    "passed": False,
                    "error": f"session setup failed: {type(error).__name__}: {error}",
                }
                rows_by_motion[motion] = row
                print(json.dumps(row, sort_keys=True), flush=True)
            _atomic_json(
                manifest_path,
                _manifest(
                    checkpoint=args.checkpoint,
                    rows_by_motion=rows_by_motion,
                    motions=motions,
                    production=production,
                ),
            )
            continue

        with session:
            for trajectory_index, motion in enumerate(chunk):
                output = output_dir / f"{_safe_name(motion)}.npz"
                row: dict[str, Any] = {
                    "motion": motion,
                    "output": str(output),
                    "passed": False,
                }
                try:
                    dataset = session.export(
                        motion_path=motion,
                        trajectory_index=trajectory_index,
                        output_path=output,
                        n_steps=args.steps,
                        require_complete=not args.allow_incomplete,
                    )
                    row.update(
                        {
                            "passed": bool(
                                dataset.metadata["qualified_full_motion"] or args.allow_incomplete
                            ),
                            "qualified_full_motion": bool(
                                dataset.metadata["qualified_full_motion"]
                            ),
                            "steps": dataset.n_steps,
                            "root_height_min": dataset.metadata["root_height_min"],
                            "root_up_min": dataset.metadata["root_up_min"],
                        }
                    )
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                rows_by_motion[motion] = row
                print(json.dumps(row, sort_keys=True), flush=True)
                _atomic_json(
                    manifest_path,
                    _manifest(
                        checkpoint=args.checkpoint,
                        rows_by_motion=rows_by_motion,
                        motions=motions,
                        production=production,
                    ),
                )

    manifest = _manifest(
        checkpoint=args.checkpoint,
        rows_by_motion=rows_by_motion,
        motions=motions,
        production=production,
    )
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "all_passed": manifest["all_passed"],
                "completed": manifest["completed"],
                "failed": manifest["failed"],
            }
        )
    )
    if not manifest["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
