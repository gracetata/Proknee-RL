"""Motion path categorization for aggregated evaluation summaries."""

from __future__ import annotations


def is_straight_locomotion(motion_type: str) -> bool:
    return motion_type in {"walk", "run"}


def classify_motion_type(motion_path: str) -> str:
    path = motion_path.lower().replace("\\", "/")
    name = path.split("/")[-1]

    if any(tok in path for tok in ("transition",)):
        return "transition"
    if any(tok in name for tok in ("jump", "hop")):
        return "jump"
    if "circle" in path or "circle" in name:
        return "circle_walk"
    if "turn_left" in path or "leftturn" in name:
        return "turn_left"
    if "turn_right" in path or "rightturn" in name:
        return "turn_right"
    if any(tok in name for tok in ("run", "running", "jog")):
        return "run"
    if any(tok in name for tok in ("walk", "walking")) or "walk" in path:
        return "walk"
    return "other"


def safe_motion_name(motion_path: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", motion_path).strip("_")
