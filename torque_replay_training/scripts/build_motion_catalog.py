#!/usr/bin/env python3
"""Inventory GMR caches and build the tracked all-motion replay candidate list."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTION_LABELS = {
    "walking_slow": "慢速直行",
    "walking_medium": "中速直行",
    "walking_fast": "快速直行",
    "walking_run": "跑步",
    "straight_forward": "直线前进",
    "straight_backward": "直线后退",
    "turn_left": "左转",
    "turn_right": "右转",
    "clockwise_circle": "顺时针绕圈",
    "counterclockwise_circle": "逆时针绕圈",
    "supported_walking": "有支撑行走",
    "handrail_walking": "扶栏行走",
    "generic_walking": "普通行走",
    "special_walking": "特殊风格行走",
    "custom": "自制长序列",
}


def _value(item: Any) -> Any:
    return item.get("value") if isinstance(item, dict) and "value" in item else item


def classify_motion(motion: str) -> str:
    name = motion.lower()
    if "/custom/" in name:
        return "custom"
    if "counterclockwisecircle" in name:
        return "counterclockwise_circle"
    if "clockwisecircle" in name:
        return "clockwise_circle"
    if "straightbackward" in name:
        return "straight_backward"
    if "straightforward" in name or "straight_line" in name or "walking_forward_" in name:
        return "straight_forward"
    if "turn_left" in name or "leftturn" in name:
        return "turn_left"
    if "turn_right" in name or "rightturn" in name:
        return "turn_right"
    if "walking_slow" in name:
        return "walking_slow"
    if "walking_medium" in name:
        return "walking_medium"
    if "walking_fast" in name:
        return "walking_fast"
    if "walking_run" in name:
        return "walking_run"
    if "handrail" in name:
        return "handrail_walking"
    if "support" in name:
        return "supported_walking"
    if any(token in name for token in ("nordic", "egyptian")):
        return "special_walking"
    return "generic_walking"


def _historical_evaluation(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return rows
    for path in root.rglob("metrics.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        motion = payload.get("motion_path")
        if not motion:
            continue
        metrics = payload.get("official_imitation", {})
        row = {
            "success": _value(metrics.get("success")) in (True, 1, 1.0),
            "frame_coverage": float(_value(metrics.get("frame_coverage")) or 0.0),
            "done_reason": str(_value(metrics.get("done_reason")) or "unknown"),
        }
        if motion not in rows or row["frame_coverage"] > rows[motion]["frame_coverage"]:
            rows[motion] = row
    return rows


def _cache_metadata(path: Path) -> tuple[int, float]:
    with np.load(path, allow_pickle=False) as data:
        frames = int(data["qpos"].shape[0])
        frequency = float(data["frequency"].item())
    return frames, frequency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--catalog-output", required=True)
    parser.add_argument("--motion-list-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    evaluation = _historical_evaluation(Path(args.evaluation_root))
    rows = []
    for path in sorted(cache_root.rglob("*.npz")):
        relative = path.relative_to(cache_root).with_suffix("")
        motion = relative.as_posix()
        frames, frequency = _cache_metadata(path)
        historical = evaluation.get(motion)
        official = not motion.startswith("KIT/custom/")
        # Every locally present GMR cache is attempted. Historical evaluation is
        # descriptive only; current schema-v3 export applies its own upright and
        # exact-replay qualification.
        replay_candidate = True
        category = classify_motion(motion)
        rows.append(
            {
                "motion_path": motion,
                "source_set": "official_kit" if official else "custom",
                "subject": relative.parts[1] if len(relative.parts) > 2 else "custom",
                "action_category": category,
                "action_zh": ACTION_LABELS[category],
                "reference_frames": frames,
                "control_steps": frames - 1,
                "duration_seconds": round((frames - 1) / frequency, 3),
                "frequency_hz": frequency,
                "historical_tracker_success": (
                    historical["success"] if historical is not None else None
                ),
                "historical_frame_coverage": (
                    historical["frame_coverage"] if historical is not None else None
                ),
                "historical_done_reason": (
                    historical["done_reason"] if historical is not None else "not_evaluated"
                ),
                "replay_candidate": replay_candidate,
            }
        )

    catalog_output = Path(args.catalog_output)
    catalog_output.parent.mkdir(parents=True, exist_ok=True)
    with catalog_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    candidates = [row["motion_path"] for row in rows if row["replay_candidate"]]
    motion_list_output = Path(args.motion_list_output)
    motion_list_output.parent.mkdir(parents=True, exist_ok=True)
    motion_list_output.write_text("\n".join(candidates) + "\n", encoding="utf-8")

    category_counts = Counter(
        row["action_category"] for row in rows if row["replay_candidate"]
    )
    summary = {
        "cache_files": len(rows),
        "official_kit": sum(row["source_set"] == "official_kit" for row in rows),
        "custom": sum(row["source_set"] == "custom" for row in rows),
        "historically_evaluated": sum(
            row["historical_tracker_success"] is not None for row in rows
        ),
        "replay_candidates": len(candidates),
        "historical_failures": [
            row["motion_path"]
            for row in rows
            if row["source_set"] == "official_kit"
            and row["historical_tracker_success"] is False
        ],
        "not_evaluated": [
            row["motion_path"]
            for row in rows
            if row["historical_tracker_success"] is None
        ],
        "candidate_control_steps": sum(
            int(row["control_steps"]) for row in rows if row["replay_candidate"]
        ),
        "candidate_duration_seconds": round(
            sum(float(row["duration_seconds"]) for row in rows if row["replay_candidate"]),
            3,
        ),
        "candidate_categories": dict(sorted(category_counts.items())),
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
