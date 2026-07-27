#!/usr/bin/env python3
"""Report cache, export, rejection, and validation progress for the all-motion set."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from torque_replay_training.paths import DEFAULT_DATA_ROOT, NEW_PROJECT_ROOT


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config_root = NEW_PROJECT_ROOT / "configs"
    motion_list = [
        row.strip()
        for row in (config_root / "all_available_motions.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    category = {}
    with (config_root / "motion_catalog.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            category[row["motion_path"]] = row["action_category"]

    cache_root = (
        Path.home() / ".musclemimic" / "caches" / "AMASS" / "MyoFullBody" / "gmr"
    )
    cache_ready = [motion for motion in motion_list if (cache_root / f"{motion}.npz").is_file()]
    output = DEFAULT_DATA_ROOT / "fullbody_all_v3"
    manifest = _load_json(output / "manifest.json")
    validation = _load_json(output / "validation_manifest.json")
    accepted = [
        row["motion"]
        for row in manifest.get("rollouts", [])
        if row.get("passed") is True
    ]
    rejected = [
        row["motion"]
        for row in manifest.get("rollouts", [])
        if "error" in row
    ]
    validated = [
        row["motion"]
        for row in validation.get("rollouts", [])
        if row.get("passed") is True
    ]
    payload = {
        "project": str(NEW_PROJECT_ROOT.parent),
        "motion_list": len(motion_list),
        "cache_ready": len(cache_ready),
        "cache_missing": len(motion_list) - len(cache_ready),
        "export": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "pending": len(motion_list) - len(accepted) - len(rejected),
        },
        "validation": {
            "passed": len(validated),
            "pending": max(len(accepted) - len(validated), 0),
            "failed": sum(
                row.get("passed") is False
                for row in validation.get("rollouts", [])
            ),
        },
        "accepted_categories": dict(
            sorted(Counter(category.get(motion, "unknown") for motion in accepted).items())
        ),
        "rejected_categories": dict(
            sorted(Counter(category.get(motion, "unknown") for motion in rejected).items())
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
