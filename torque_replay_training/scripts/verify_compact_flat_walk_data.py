#!/usr/bin/env python3
"""Verify compact flat-walk files by size, SHA-256 and schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from torque_replay_training.compact_schema import CompactTorqueReplayDataset
from torque_replay_training.paths import NEW_PROJECT_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(NEW_PROJECT_ROOT / "configs/flat_walk_compact_manifest.json"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(NEW_PROJECT_ROOT / "data/flat_walk_compact_v1"),
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Only check file size and SHA-256",
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir)
    missing = []
    wrong_size = []
    wrong_sha256 = []
    invalid_schema = []
    total_bytes = 0
    for row in manifest["datasets"]:
        path = data_dir / row["dataset_basename"]
        if not path.is_file():
            missing.append(path.name)
            continue
        total_bytes += path.stat().st_size
        if path.stat().st_size != int(row["compact_bytes"]):
            wrong_size.append(path.name)
            continue
        if _sha256(path) != row["sha256"]:
            wrong_sha256.append(path.name)
            continue
        if not args.skip_schema:
            try:
                dataset = CompactTorqueReplayDataset.load(path)
                if (
                    dataset.n_steps != int(row["control_steps"])
                    or dataset.n_substeps != int(row["physics_substeps"])
                    or dataset.nq != int(row["nq"])
                    or dataset.nv != int(row["nv"])
                ):
                    raise ValueError("manifest dimensions differ")
            except Exception as error:
                invalid_schema.append({"dataset": path.name, "error": str(error)})
    requested = len(manifest["datasets"])
    failed = len(missing) + len(wrong_size) + len(wrong_sha256) + len(invalid_schema)
    report = {
        "requested": requested,
        "valid": requested - failed,
        "bytes": total_bytes,
        "missing": missing,
        "wrong_size": wrong_size,
        "wrong_sha256": wrong_sha256,
        "invalid_schema": invalid_schema,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
