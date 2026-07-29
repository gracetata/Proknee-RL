#!/usr/bin/env python3
"""Build a subject-disjoint flat-walking split from validated schema-v3 replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from torque_replay_training.paths import NEW_PROJECT_ROOT


FLAT_CATEGORIES = (
    "generic_walking",
    "straight_forward",
    "walking_slow",
    "walking_medium",
    "walking_fast",
)
DEFAULT_HOLDOUT_SUBJECTS = ("9", "425")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "motions": len(rows),
        "control_steps": sum(int(row["control_steps"]) for row in rows),
        "duration_seconds": round(sum(float(row["duration_seconds"]) for row in rows), 2),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "subjects": dict(sorted(Counter(row["subject"] for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        default=str(NEW_PROJECT_ROOT / "configs/motion_catalog.csv"),
    )
    parser.add_argument(
        "--validation-manifest",
        default=str(NEW_PROJECT_ROOT / "data/fullbody_all_v3/validation_manifest.json"),
    )
    parser.add_argument(
        "--output",
        default=str(NEW_PROJECT_ROOT / "configs/flat_walk_split.json"),
    )
    parser.add_argument(
        "--holdout-subject",
        action="append",
        dest="holdout_subjects",
        help="Repeat to override the default subject-disjoint validation set",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog).resolve()
    validation_path = Path(args.validation_manifest).resolve()
    output_path = Path(args.output).resolve()
    holdout_subjects = tuple(args.holdout_subjects or DEFAULT_HOLDOUT_SUBJECTS)

    with catalog_path.open(encoding="utf-8", newline="") as stream:
        catalog = {row["motion_path"]: row for row in csv.DictReader(stream)}
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("all_passed") is not True:
        raise SystemExit("validation manifest must have all_passed=true")

    selected: list[dict[str, Any]] = []
    for replay in validation.get("rollouts", []):
        motion = str(replay["motion"])
        metadata = catalog.get(motion)
        if replay.get("passed") is not True or metadata is None:
            continue
        if metadata["action_category"] not in FLAT_CATEGORIES:
            continue
        dataset = Path(str(replay["dataset"]))
        if not dataset.is_file():
            relocated = validation_path.parent / dataset.name
            if not relocated.is_file():
                raise FileNotFoundError(f"missing validated dataset: {dataset}")
            dataset = relocated
        selected.append(
            {
                "motion": motion,
                "dataset_basename": dataset.name,
                "subject": metadata["subject"],
                "category": metadata["action_category"],
                "action_zh": metadata["action_zh"],
                "control_steps": int(metadata["control_steps"]),
                "duration_seconds": float(metadata["duration_seconds"]),
                "bytes": dataset.stat().st_size,
            }
        )

    selected.sort(key=lambda row: row["motion"])
    train = [row for row in selected if row["subject"] not in holdout_subjects]
    validation_rows = [row for row in selected if row["subject"] in holdout_subjects]
    for category in FLAT_CATEGORIES:
        if not any(row["category"] == category for row in train):
            raise SystemExit(f"training split has no {category}")
        if not any(row["category"] == category for row in validation_rows):
            raise SystemExit(f"validation split has no {category}")
    if {row["subject"] for row in train} & {row["subject"] for row in validation_rows}:
        raise SystemExit("subject leakage detected")

    payload = {
        "schema_version": 1,
        "source_catalog": str(catalog_path),
        "source_catalog_sha256": _sha256(catalog_path),
        "source_validation_manifest": str(validation_path),
        "source_validation_manifest_sha256": _sha256(validation_path),
        "flat_categories": list(FLAT_CATEGORIES),
        "excluded_categories": [
            "walking_run",
            "straight_backward",
            "turn_left",
            "turn_right",
            "clockwise_circle",
            "counterclockwise_circle",
            "handrail_walking",
            "supported_walking",
            "special_walking",
            "custom",
        ],
        "holdout_subjects": list(holdout_subjects),
        "summary": {
            "all": _summary(selected),
            "train": _summary(train),
            "validation": _summary(validation_rows),
        },
        "train": train,
        "validation": validation_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
