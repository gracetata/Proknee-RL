#!/usr/bin/env python
"""Batch locomotion eval: rollout + adaptive composite score + radar (no section3/4/5 PNGs)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from musclemimic.distill.config import motion_list_from_group, repo_root
from musclemimic.evaluation.composite_score import (
    build_composite_report,
    save_composite_report,
)
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.light_summaries import (
    build_symmetry_summary_from_rollout,
    build_torque_summary_from_rollout,
)
from musclemimic.evaluation.motion_taxonomy import classify_motion_type
from musclemimic.evaluation.radar_plot import plot_composite_radar
from musclemimic.evaluation.scoring_profile import eval_run_bucket, resolve_scoring_profile


def _resolve_run_dir(bundle_dir: Path) -> Path:
    if (bundle_dir / "config.yaml").is_file():
        return bundle_dir
    dated = sorted(bundle_dir.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in dated:
        if (candidate / "config.yaml").is_file():
            return candidate
    return bundle_dir


def _flatten_timestamp_subdir(bundle_dir: Path) -> None:
    if (bundle_dir / "per_motion").is_dir() and (bundle_dir / "config.yaml").is_file():
        return
    dated = sorted(bundle_dir.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ts_dir in dated:
        if not (ts_dir / "config.yaml").is_file():
            continue
        for name in ("config.yaml", "per_motion", "metrics_per_motion.csv", "metrics_summary.csv", "failed_motions.csv"):
            src = ts_dir / name
            if src.exists():
                dest = bundle_dir / name
                if dest.exists():
                    continue
                src.rename(dest)
        try:
            ts_dir.rmdir()
            if ts_dir.parent.is_dir() and not any(ts_dir.parent.iterdir()):
                ts_dir.parent.rmdir()
        except OSError:
            pass
        break


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset_group",
        default="KIT_KINESIS_TESTING_MOTIONS",
        help="AMASS group name (108 test motions). Use KIT_KINESIS_TRAINING_MOTIONS for 972.",
    )
    p.add_argument("--motion_list", default=None, help="Optional text file, one motion per line")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument("--env_type", default="official_fullbody")
    p.add_argument("--runs_root", default="outputs/eval/runs")
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    p.add_argument("--eval-config-name", default="conf_eval_locomotion")
    p.add_argument("--save_video", type=_str2bool, default=False)
    p.add_argument("--resume", action="store_true", help="Skip motions with composite_score.json")
    p.add_argument("--start", type=int, default=0, help="Start index in motion list")
    p.add_argument("--limit", type=int, default=0, help="Max motions (0 = all)")
    return p.parse_args()


def _load_motions(args) -> list[str]:
    if args.motion_list:
        lines = Path(args.motion_list).read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return motion_list_from_group(args.dataset_group)


def _run_rollout(
    *,
    root: Path,
    motion_path: str,
    eval_dir: Path,
    args,
) -> int:
    placeholder = eval_dir / "config.yaml"
    if not placeholder.exists():
        placeholder.write_text("# reserved\n", encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/evaluate_locomotion.py",
        "--env_type",
        args.env_type,
        "--checkpoint_path",
        args.checkpoint_path,
        "--motion_path",
        motion_path,
        "--output_dir",
        str(eval_dir),
        "--controller_type",
        args.controller_type,
        "--config-name",
        args.config_name,
        "--eval-config-name",
        args.eval_config_name,
        "--save_video",
        "false" if not args.save_video else "true",
        "--save_plots",
        "false",
        "--no_ghost",
        "--no_termination",
    ]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(root)).returncode


def _score_motion(
    *,
    motion_path: str,
    eval_dir: Path,
    config_name: str,
) -> dict:
    safe = safe_motion_name(motion_path)
    motion_dir = eval_dir / "per_motion" / safe
    metrics_json = motion_dir / "metrics.json"
    rollout_npz = motion_dir / "rollout_data.npz"
    if not metrics_json.is_file() or not rollout_npz.is_file():
        raise FileNotFoundError(f"Missing artifacts in {motion_dir}")

    motion_type = classify_motion_type(motion_path)
    profile = resolve_scoring_profile(motion_type)

    torque_summary = build_torque_summary_from_rollout(
        rollout_npz, motion_path, config_name=config_name
    )
    symmetry_summary = None
    if "bilateral_symmetry" in profile.active_dimensions:
        symmetry_summary = build_symmetry_summary_from_rollout(
            rollout_npz, motion_path, config_name=config_name
        )

    analysis_dir = eval_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report = build_composite_report(
        metrics_json,
        symmetry_summary=symmetry_summary,
        torque_summary=torque_summary,
        motion_path=motion_path,
    )
    save_composite_report(report, analysis_dir)
    plot_composite_radar(report, analysis_dir / "composite_score_radar.png")
    return report


def main() -> int:
    args = parse_args()
    root = repo_root()
    motions = _load_motions(args)
    if args.limit > 0:
        motions = motions[args.start : args.start + args.limit]
    elif args.start > 0:
        motions = motions[args.start :]

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = root / runs_root
    runs_root = runs_root / args.controller_type

    log_path = runs_root.parent / f"{args.controller_type}_batch_progress.jsonl"
    runs_root.mkdir(parents=True, exist_ok=True)

    ok, skip, fail = 0, 0, 0
    t0 = time.time()
    print(f"Batch: {len(motions)} motions -> {runs_root}")

    for i, motion_path in enumerate(motions):
        safe = safe_motion_name(motion_path)
        bucket = eval_run_bucket(motion_path)
        eval_dir = runs_root / bucket / safe
        score_json = eval_dir / "analysis" / "composite_score.json"
        legacy_score = runs_root / safe / "analysis" / "composite_score.json"

        print(f"\n[{i+1}/{len(motions)}] {motion_path}", flush=True)
        if args.resume and (score_json.is_file() or legacy_score.is_file()):
            print("  resume: skip (composite_score.json exists)")
            skip += 1
            continue

        try:
            eval_dir.mkdir(parents=True, exist_ok=True)
            rc = _run_rollout(root=root, motion_path=motion_path, eval_dir=eval_dir, args=args)
            if rc != 0:
                raise RuntimeError(f"evaluate_locomotion exited {rc}")
            _flatten_timestamp_subdir(eval_dir)
            eval_dir = _resolve_run_dir(eval_dir)

            report = _score_motion(
                motion_path=motion_path,
                eval_dir=eval_dir,
                config_name=args.config_name,
            )
            entry = {
                "motion_path": motion_path,
                "motion_type": report.get("motion_type"),
                "total_score": report.get("total_score"),
                "n_dimensions": report.get("scoring_profile", {}).get("n_dimensions"),
                "status": "ok",
            }
            ok += 1
            print(
                f"  OK total={report['total_score']:.1f} "
                f"({report['scoring_profile']['n_dimensions']}D)",
                flush=True,
            )
        except Exception as exc:
            entry = {
                "motion_path": motion_path,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            fail += 1
            print(f"  FAIL: {exc}", flush=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed/60:.1f} min | ok={ok} skip={skip} fail={fail} | log={log_path}"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
