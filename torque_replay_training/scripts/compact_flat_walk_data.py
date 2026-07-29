#!/usr/bin/env python3
"""Build the lossless compact flat-walk replay catalog in parallel."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from torque_replay_training.compact_schema import CompactTorqueReplayDataset
from torque_replay_training.paths import NEW_PROJECT_ROOT
from torque_replay_training.schema import TorqueReplayDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(NEW_PROJECT_ROOT))
    except ValueError:
        return str(path)


def _convert(source_value: str, target_value: str) -> dict[str, Any]:
    source = Path(source_value)
    target = Path(target_value)
    compact = CompactTorqueReplayDataset.from_full(TorqueReplayDataset.load(source))
    compact.save(target)
    reloaded = CompactTorqueReplayDataset.load(target)
    if not (
        reloaded.motion_path == compact.motion_path
        and reloaded.dt_control == compact.dt_control
        and reloaded.dt_physics == compact.dt_physics
        and reloaded.n_steps == compact.n_steps
        and reloaded.n_substeps == compact.n_substeps
        and reloaded.nq == compact.nq
        and reloaded.nv == compact.nv
    ):
        raise RuntimeError(f"compact reload mismatch: {target}")
    return {
        "dataset_basename": target.name,
        "motion": compact.motion_path,
        "control_steps": compact.n_steps,
        "physics_substeps": compact.n_substeps,
        "nq": compact.nq,
        "nv": compact.nv,
        "source_bytes": source.stat().st_size,
        "compact_bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default=str(NEW_PROJECT_ROOT / "configs/flat_walk_split.json"),
    )
    parser.add_argument(
        "--source-dir",
        default=str(NEW_PROJECT_ROOT / "data/fullbody_all_v3"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(NEW_PROJECT_ROOT / "data/flat_walk_compact_v1"),
    )
    parser.add_argument(
        "--manifest",
        default=str(NEW_PROJECT_ROOT / "configs/flat_walk_compact_manifest.json"),
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    split_path = Path(args.split).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_group = {
        row["dataset_basename"]: group
        for group in ("train", "validation")
        for row in split[group]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    futures = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for name in sorted(expected_group):
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            future = executor.submit(_convert, str(source), str(output_dir / name))
            futures[future] = name
        rows = []
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            row["group"] = expected_group[row["dataset_basename"]]
            rows.append(row)
            print(
                f"[{index:03d}/{len(futures):03d}] {row['dataset_basename']} "
                f"{row['source_bytes'] / 2**20:.2f} -> "
                f"{row['compact_bytes'] / 2**20:.2f} MiB",
                flush=True,
            )
    rows.sort(key=lambda row: row["dataset_basename"])
    source_bytes = sum(int(row["source_bytes"]) for row in rows)
    compact_bytes = sum(int(row["compact_bytes"]) for row in rows)
    payload = {
        "compact_schema_version": 1,
        "split": _portable_path(split_path),
        "source_dir": _portable_path(source_dir),
        "output_dir": _portable_path(output_dir),
        "summary": {
            "datasets": len(rows),
            "control_steps": sum(int(row["control_steps"]) for row in rows),
            "source_bytes": source_bytes,
            "compact_bytes": compact_bytes,
            "compression_ratio": compact_bytes / source_bytes,
            "train": sum(row["group"] == "train" for row in rows),
            "validation": sum(row["group"] == "validation" for row in rows),
        },
        "datasets": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
