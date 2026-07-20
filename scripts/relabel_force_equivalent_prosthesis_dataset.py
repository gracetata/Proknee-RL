#!/usr/bin/env python
"""Relabel prosthesis DAgger rollouts with force-equivalent remaining-muscle actions.

For each frame, scale the teacher remaining-muscle action so that the implied
actuator force at the visited student state better matches the teacher force.
Prosthesis 4-DOF labels are kept as teacher_qfrc torque targets already stored
in the rollout npz.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from musclemimic.distill.config import repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", nargs="+", required=True, help="One or more rollout directories")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force_eps", type=float, default=1.0, help="Min |force| for ratio scaling")
    parser.add_argument("--ratio_min", type=float, default=0.25)
    parser.add_argument("--ratio_max", type=float, default=4.0)
    parser.add_argument("--copy_mapping", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def force_equivalent_action(
    action: np.ndarray,
    target_force: np.ndarray,
    student_force: np.ndarray,
    *,
    force_eps: float,
    ratio_min: float,
    ratio_max: float,
) -> tuple[np.ndarray, dict[str, float]]:
    action = np.asarray(action, dtype=np.float32)
    target_force = np.asarray(target_force, dtype=np.float32)
    student_force = np.asarray(student_force, dtype=np.float32)
    denom = np.where(np.abs(student_force) >= force_eps, student_force, np.nan)
    ratio = np.where(np.isfinite(denom), target_force / denom, 1.0).astype(np.float32)
    ratio = np.clip(ratio, ratio_min, ratio_max)
    equiv = np.clip(action * ratio, -1.0, 1.0).astype(np.float32)
    rel_before = float(np.mean(np.abs(target_force - student_force)) / (np.mean(np.abs(target_force)) + 1e-6))
    # Approximate post-correction force as scaled student force.
    approx_force = student_force * ratio
    rel_after = float(np.mean(np.abs(target_force - approx_force)) / (np.mean(np.abs(target_force)) + 1e-6))
    action_delta = float(np.mean(np.abs(equiv - action)))
    return equiv, {
        "relative_force_mismatch_before": rel_before,
        "relative_force_mismatch_after_approx": rel_after,
        "mean_action_delta": action_delta,
    }


def relabel_file(
    in_path: Path,
    out_path: Path,
    *,
    force_eps: float,
    ratio_min: float,
    ratio_max: float,
) -> dict:
    with np.load(in_path, allow_pickle=True) as data:
        required = {
            "obs_student",
            "target_remaining_muscle_action",
            "target_remaining_muscle_force",
            "student_remaining_muscle_force_from_teacher_action",
            "target_prosthesis_action",
            "target_prosthesis_tau",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{in_path}: missing required arrays {missing}")

        payload = {key: data[key] for key in data.files}
        action = np.asarray(payload["target_remaining_muscle_action"], dtype=np.float32)
        target_force = np.asarray(payload["target_remaining_muscle_force"], dtype=np.float32)
        student_force = np.asarray(payload["student_remaining_muscle_force_from_teacher_action"], dtype=np.float32)
        equiv, stats = force_equivalent_action(
            action,
            target_force,
            student_force,
            force_eps=force_eps,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
        )
        payload["target_remaining_muscle_action_raw"] = action
        payload["target_remaining_muscle_action_force_equiv"] = equiv
        payload["target_remaining_muscle_action"] = equiv
        payload["relabel_mode"] = np.asarray("force_equivalent_v1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    stats.update(
        {
            "input": str(in_path),
            "output": str(out_path),
            "frames": int(action.shape[0]),
        }
    )
    return stats


def main() -> int:
    args = parse_args()
    input_dirs = [resolve_path(p) for p in args.input_dir]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for input_dir in input_dirs:
        for in_path in sorted(input_dir.glob("*.npz")):
            out_name = f"{input_dir.name}__{in_path.name}" if len(input_dirs) > 1 else in_path.name
            out_path = output_dir / out_name
            records.append(
                relabel_file(
                    in_path,
                    out_path,
                    force_eps=float(args.force_eps),
                    ratio_min=float(args.ratio_min),
                    ratio_max=float(args.ratio_max),
                )
            )

    if args.copy_mapping:
        for input_dir in input_dirs:
            mapping = input_dir / "mapping.json"
            if mapping.exists():
                shutil.copy2(mapping, output_dir / "mapping.json")
                break

    summary = {
        "files": len(records),
        "frames": int(sum(r["frames"] for r in records)),
        "mean_relative_force_mismatch_before": float(np.mean([r["relative_force_mismatch_before"] for r in records])),
        "mean_relative_force_mismatch_after_approx": float(
            np.mean([r["relative_force_mismatch_after_approx"] for r in records])
        ),
        "mean_action_delta": float(np.mean([r["mean_action_delta"] for r in records])),
        "records": records,
        "output_dir": str(output_dir),
    }
    (output_dir / "relabel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
