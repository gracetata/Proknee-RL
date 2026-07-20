#!/usr/bin/env python
"""Move motion eval dirs under 6d / 5d / 4d scoring buckets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from musclemimic.distill.config import repo_root
from musclemimic.evaluation.scoring_profile import eval_run_bucket

BUCKETS: tuple[str, ...] = (
    "6d_symmetric_gait",
    "5d_asymmetric_gait",
    "4d_other",
)


def safe_to_motion_path(safe: str) -> str:
    if safe.startswith("KIT_"):
        parts = safe.split("_")
        if len(parts) >= 3:
            return f"{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}"
    return safe.replace("_", "/")


def motion_path_from_dir(item: Path) -> str:
    cs = item / "analysis" / "composite_score.json"
    if cs.is_file():
        data = json.loads(cs.read_text(encoding="utf-8"))
        return str(data.get("motion_path") or safe_to_motion_path(item.name))
    return safe_to_motion_path(item.name)


def iter_motion_dirs(controller_root: Path) -> list[tuple[Path, str]]:
    """Return (dir_path, safe_name) for every scored motion under controller_root."""
    found: list[tuple[Path, str]] = []
    seen: set[str] = set()

    def add(item: Path) -> None:
        if not item.is_dir():
            return
        if item.name in BUCKETS:
            return
        if item.name.startswith("_"):
            return
        if item.name in seen:
            return
        seen.add(item.name)
        found.append((item, item.name))

    for item in sorted(controller_root.iterdir()):
        if item.is_dir() and item.name in BUCKETS:
            for child in sorted(item.iterdir()):
                add(child)
        else:
            add(item)
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument("--runs_root", default="outputs/eval/runs")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = root / runs_root
    controller_root = runs_root / args.controller_type

    if not controller_root.is_dir():
        raise SystemExit(f"Missing {controller_root}")

    for bucket in BUCKETS:
        (controller_root / bucket).mkdir(parents=True, exist_ok=True)

    log_src = controller_root / "_batch_progress.jsonl"
    log_dst = runs_root / f"{args.controller_type}_batch_progress.jsonl"
    if log_src.is_file():
        if args.dry_run:
            print(f"would move log {log_src} -> {log_dst}")
        elif not log_dst.exists() or log_src.stat().st_mtime > log_dst.stat().st_mtime:
            shutil.move(str(log_src), str(log_dst))
            print(f"moved log -> {log_dst}")

    moved = {b: 0 for b in BUCKETS}
    skipped = 0
    for item, safe in iter_motion_dirs(controller_root):
        motion_path = motion_path_from_dir(item)
        bucket = eval_run_bucket(motion_path)
        dest = controller_root / bucket / safe
        if item.resolve() == dest.resolve():
            continue
        if dest.exists():
            print(f"skip (dest exists): {safe} -> {bucket}/")
            skipped += 1
            continue
        rel_from = item.relative_to(controller_root)
        if args.dry_run:
            print(f"would move {rel_from} -> {bucket}/{safe}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(dest))
        moved[bucket] += 1

    counts = {b: len(list((controller_root / b).iterdir())) for b in BUCKETS}
    print("Done.", {"moved": moved, "counts": counts, "skipped": skipped})
    remaining = sorted(p.name for p in controller_root.iterdir() if p.is_dir())
    print("top-level now:", remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
