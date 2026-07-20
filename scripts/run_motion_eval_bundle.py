#!/usr/bin/env python
"""Evaluate one motion end-to-end: rollout → analysis plots → adaptive composite score."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from musclemimic.distill.config import repo_root
from musclemimic.evaluation.composite_score import (
    build_composite_report,
    save_composite_report,
)
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.motion_taxonomy import classify_motion_type
from musclemimic.evaluation.radar_plot import plot_composite_radar
from musclemimic.evaluation.light_summaries import persist_rollout_analysis_summaries
from musclemimic.evaluation.scoring_profile import eval_run_bucket, resolve_scoring_profile


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", required=True)
    p.add_argument("--env_type", default="official_fullbody")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument("--output_dir", default=None, help="Default: outputs/eval/runs/<controller>/<safe_motion>")
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    p.add_argument("--eval-config-name", default="conf_eval_locomotion")
    p.add_argument("--save_video", type=_str2bool, default=False)
    p.add_argument("--save_plots", type=_str2bool, default=False)
    p.add_argument("--skip_rollout", action="store_true")
    p.add_argument(
        "--with_section_plots",
        action="store_true",
        help="Also write analysis/section3|4|5 PNG plots (off by default).",
    )
    return p.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _flatten_timestamp_subdir(bundle_dir: Path) -> None:
    """Hoist YYYY-MM-DD/HH-MM-SS run artifacts to bundle_dir root."""
    if (bundle_dir / "config.yaml").is_file() and (bundle_dir / "per_motion").is_dir():
        return
    dated = sorted(bundle_dir.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ts_dir in dated:
        if not (ts_dir / "config.yaml").is_file():
            continue
        for name in (
            "config.yaml",
            "metrics.json",
            "metrics_per_motion.csv",
            "metrics_summary.csv",
            "failed_motions.csv",
            "per_motion",
            "analysis",
        ):
            src = ts_dir / name
            if src.exists():
                dest = bundle_dir / name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                src.rename(dest)
        parent = ts_dir.parent
        ts_dir.rmdir()
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
        break


def _resolve_run_dir(bundle_dir: Path) -> Path:
    """Runner may create YYYY-MM-DD/HH-MM-SS under bundle_dir on first run."""
    if (bundle_dir / "config.yaml").is_file():
        return bundle_dir
    dated = sorted(bundle_dir.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in dated:
        if (candidate / "config.yaml").is_file():
            return candidate
    return bundle_dir


def main() -> int:
    args = parse_args()
    root = repo_root()
    safe = safe_motion_name(args.motion_path)
    if args.output_dir:
        eval_dir = Path(args.output_dir)
        if not eval_dir.is_absolute():
            eval_dir = root / eval_dir
    else:
        bucket = eval_run_bucket(args.motion_path)
        eval_dir = root / "outputs" / "eval" / "runs" / args.controller_type / bucket / safe
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Prevent runner from nesting another timestamp dir under bundle_dir.
    placeholder_cfg = eval_dir / "config.yaml"
    if not placeholder_cfg.exists():
        placeholder_cfg.write_text("# reserved; overwritten by evaluate_locomotion.py\n", encoding="utf-8")

    motion_type = classify_motion_type(args.motion_path)
    profile = resolve_scoring_profile(motion_type)
    print(f"Motion type: {motion_type}")
    print(f"Scoring: {profile.summary_zh}")
    print(f"Active dimensions ({len(profile.active_dimensions)}): {', '.join(profile.active_dimensions)}")

    if not args.skip_rollout:
        _run(
            [
                sys.executable,
                "scripts/evaluate_locomotion.py",
                "--env_type",
                args.env_type,
                "--checkpoint_path",
                args.checkpoint_path,
                "--motion_path",
                args.motion_path,
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
            ],
            cwd=root,
        )

    eval_dir = _resolve_run_dir(eval_dir)
    _flatten_timestamp_subdir(eval_dir)
    eval_dir = _resolve_run_dir(eval_dir)
    motion_dir = eval_dir / "per_motion" / safe
    rollout_npz = motion_dir / "rollout_data.npz"
    metrics_json = motion_dir / "metrics.json"
    if not rollout_npz.is_file() or not metrics_json.is_file():
        print(f"Missing rollout artifacts under {motion_dir}", file=sys.stderr)
        return 1

    analysis_dir = eval_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    torque_summary, symmetry_summary = persist_rollout_analysis_summaries(
        analysis_dir,
        rollout_npz,
        args.motion_path,
        config_name=args.config_name,
        include_symmetry="bilateral_symmetry" in profile.active_dimensions,
    )

    if args.with_section_plots:
        plot_cmds: list[tuple[str, list[str]]] = [
            (
                "scripts/plot_locomotion_leg_dynamics.py",
                ["--rollout_npz", str(rollout_npz), "--output_dir", str(analysis_dir)],
            ),
            (
                "scripts/plot_locomotion_force_symmetry.py",
                ["--rollout_npz", str(rollout_npz), "--output_dir", str(analysis_dir)],
            ),
            (
                "scripts/plot_locomotion_torque_metrics.py",
                [
                    "--rollout_npz",
                    str(rollout_npz),
                    "--output_dir",
                    str(analysis_dir),
                    "--motion_path",
                    args.motion_path,
                ],
            ),
            (
                "scripts/plot_locomotion_symmetry_metrics.py",
                [
                    "--rollout_npz",
                    str(rollout_npz),
                    "--output_dir",
                    str(analysis_dir),
                    "--motion_path",
                    args.motion_path,
                ],
            ),
        ]
        for script, extra in plot_cmds:
            _run([sys.executable, script, *extra], cwd=root)

    if args.with_section_plots:
        import json as _json

        sym_p = analysis_dir / "section5_symmetry" / "symmetry_metrics_summary.json"
        torq_p = analysis_dir / "section4_torque" / "torque_metrics_summary.json"
        if sym_p.is_file():
            symmetry_summary = _json.loads(sym_p.read_text(encoding="utf-8"))
        if torq_p.is_file():
            torque_summary = _json.loads(torq_p.read_text(encoding="utf-8"))

    report = build_composite_report(
        metrics_json,
        symmetry_summary=symmetry_summary,
        torque_summary=torque_summary,
        motion_path=args.motion_path,
    )
    save_composite_report(report, analysis_dir)
    radar_path = plot_composite_radar(report, analysis_dir / "composite_score_radar.png")

    print("\n=== Composite score ===")
    for dim_id, block in report["dimensions"].items():
        print(f"  {block.get('label_zh', dim_id)}: {block['score']:.1f}")
    print(f"  TOTAL ({report['scoring_profile']['n_dimensions']}D): {report['total_score']:.1f}")
    print(f"Bundle dir: {eval_dir}")
    print(f"Radar: {radar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
