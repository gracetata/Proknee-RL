"""Shared CLI helpers for the ProKnee MJLAB scripts."""

from __future__ import annotations

import json
from pathlib import Path


def replay_paths(
    split_path: str | Path,
    data_dir: str | Path,
    group: str,
) -> tuple[str, ...]:
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if group not in split:
        raise KeyError(f"split has no group {group!r}")
    root = Path(data_dir).resolve()
    paths = tuple(str(root / row["dataset_basename"]) for row in split[group])
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} compact replay files are missing")
    return paths
