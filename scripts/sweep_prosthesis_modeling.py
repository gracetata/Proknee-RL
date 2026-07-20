#!/usr/bin/env python
"""Run the prosthesis modeling repair sweeps from the plan."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from musclemimic.distill.config import repo_root


DEFAULT_MOTION = "KIT/3/walk_6m_straight_line04_poses"
DEFAULT_CKPT = "outputs/prosthesis_distill/latest/checkpoints/checkpoint_distilled"

KNEE_THIGH_MUSCLES = (
    "bflh_l",
    "bfsh_l",
    "semimem_l",
    "semiten_l",
    "recfem_l",
    "vasint_l",
    "vaslat_l",
    "vasmed_l",
)
BIARTICULAR_STABILIZERS = (
    "bflh_l",
    "semimem_l",
    "semiten_l",
    "recfem_l",
)


@dataclass(frozen=True)
class Variant:
    name: str
    phase: str
    torque_limits: str = "120,120,30,20"
    disable_exclude: tuple[str, ...] = ()
    action_type: str = "torque"
    label_mode: str = "inverse_dynamics"
    residual_pd_kp: str | None = None
    residual_pd_kd: str | None = None
    notes: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-path", default=DEFAULT_MOTION)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--output-dir", default="outputs/prosthesis_modeling_sweeps")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-scale-motions", type=int, default=10)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--phase",
        choices=["all", "mask", "torque", "pd", "scale"],
        default="all",
        help="Run all phases or only one phase.",
    )
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def run_cmd(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode == 0, proc.stdout[-4000:]


def collect_cmd(args: argparse.Namespace, variant: Variant, dataset_dir: Path, motion_paths: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/collect_teacher_rollouts.py",
        "--motion_path",
        *motion_paths,
        "--output_dir",
        str(dataset_dir),
        "--n_steps_per_motion",
        str(args.steps),
        "--num_workers",
        str(args.num_workers),
        "--continue_on_error",
        "--label_mode",
        variant.label_mode,
        "--pin_student_state",
        "--prosthesis_torque_limits",
        variant.torque_limits,
        "--prosthesis_action_type",
        variant.action_type,
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    if variant.disable_exclude:
        cmd.extend(["--disable_muscles_exclude", ",".join(variant.disable_exclude)])
    if variant.residual_pd_kp:
        cmd.extend(["--residual_pd_kp", variant.residual_pd_kp])
    if variant.residual_pd_kd:
        cmd.extend(["--residual_pd_kd", variant.residual_pd_kd])
    return cmd


def diagnose_cmd(args: argparse.Namespace, variant: Variant, dataset_dir: Path, diag_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/diagnose_prosthesis_distill.py",
        "--motion-path",
        args.motion_path,
        "--dataset-dir",
        str(dataset_dir),
        "--checkpoint",
        str(resolve(args.checkpoint)),
        "--output-dir",
        str(diag_dir),
        "--steps",
        str(args.steps),
        "--skip-policy",
        "--prosthesis_torque_limits",
        variant.torque_limits,
        "--prosthesis_action_type",
        variant.action_type,
    ]
    if variant.disable_exclude:
        cmd.extend(["--disable_muscles_exclude", ",".join(variant.disable_exclude)])
    if variant.residual_pd_kp:
        cmd.extend(["--residual_pd_kp", variant.residual_pd_kp])
    if variant.residual_pd_kd:
        cmd.extend(["--residual_pd_kd", variant.residual_pd_kd])
    return cmd


def variants() -> list[Variant]:
    mask = [
        Variant("strict_19", "mask", notes="current default 19-muscle hard mask"),
        Variant(
            "ankle_foot_only",
            "mask",
            disable_exclude=KNEE_THIGH_MUSCLES,
            notes="keep thigh/knee muscles active; disable ankle/foot set",
        ),
        Variant(
            "knee_ankle_primary",
            "mask",
            disable_exclude=BIARTICULAR_STABILIZERS,
            notes="keep selected biarticular stabilizers active",
        ),
    ]
    torque = [
        Variant(f"strict19_k{knee}_a{ankle}", "torque", torque_limits=f"{knee},{ankle},30,20")
        for knee in (120, 180, 240, 300)
        for ankle in (120, 160, 200)
    ]
    pd = [
        Variant(
            "strict19_pd_residual_zero",
            "pd",
            torque_limits="240,160,30,20",
            action_type="pd_residual_torque",
            label_mode="inverse_dynamics",
            residual_pd_kp="240,180,60,40",
            residual_pd_kd="24,18,6,4",
            notes="tests reference PD plus residual labels; zero_residual_pd diagnostic is written",
        )
    ]
    scale = [
        Variant(
            "scale_1motion_pd_residual",
            "scale",
            torque_limits="240,160,30,20",
            action_type="pd_residual_torque",
            residual_pd_kp="240,180,60,40",
            residual_pd_kd="24,18,6,4",
            notes="one-motion v2 data smoke test",
        ),
        Variant(
            "scale_10motion_pd_residual",
            "scale",
            torque_limits="240,160,30,20",
            action_type="pd_residual_torque",
            residual_pd_kp="240,180,60,40",
            residual_pd_kd="24,18,6,4",
            notes="ten-motion v2 data smoke test",
        ),
    ]
    return mask + torque + pd + scale


def read_diag_summary(path: Path) -> dict:
    summary_path = path / "diagnostic_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def read_dataset_summary(path: Path) -> dict:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload.get("summary", payload)


def row_from_summary(variant: Variant, dataset_dir: Path, diag_dir: Path, ok_collect: bool, ok_diag: bool, error: str) -> dict:
    summary = read_diag_summary(diag_dir)
    dataset_summary = read_dataset_summary(dataset_dir)
    replays = summary.get("replay_summaries", {})
    normalized = replays.get("normalized") or {}
    zero_pd = replays.get("zero_residual_pd") or {}
    return {
        "variant": variant.name,
        "phase": variant.phase,
        "collect_ok": ok_collect,
        "diagnose_ok": ok_diag,
        "torque_limits": variant.torque_limits,
        "action_type": variant.action_type,
        "disable_exclude": ",".join(variant.disable_exclude),
        "collected_motions": dataset_summary.get("collected_motions"),
        "total_frames": dataset_summary.get("total_frames"),
        "average_rollout_length": dataset_summary.get("average_rollout_length"),
        "target_action_clipping_ratio": dataset_summary.get("target_prosthesis_action_clipping_ratio"),
        "normalized_steps": normalized.get("steps"),
        "normalized_first_done_step": normalized.get("first_done_step"),
        "normalized_root_height_min": normalized.get("root_height_min"),
        "normalized_joint_error_mean": normalized.get("joint_error_mean"),
        "zero_pd_steps": zero_pd.get("steps"),
        "zero_pd_first_done_step": zero_pd.get("first_done_step"),
        "zero_pd_root_height_min": zero_pd.get("root_height_min"),
        "dataset_dir": str(dataset_dir),
        "diagnostic_dir": str(diag_dir),
        "notes": variant.notes,
        "error_tail": error,
    }


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = resolve(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cwd = repo_root()
    rows: list[dict] = []
    for variant in variants():
        if args.phase != "all" and variant.phase != args.phase:
            continue
        dataset_dir = root / "datasets" / variant.name
        diag_dir = root / "diagnostics" / variant.name
        if variant.name == "scale_10motion_pd_residual":
            # Use the configured training group for this smoke test; collect script handles max selection by explicit phase.
            collect = [
                sys.executable,
                "scripts/collect_teacher_rollouts.py",
                "--dataset_group",
                "KIT_KINESIS_TRAINING_MOTIONS",
                "--max_motions",
                str(args.max_scale_motions),
                "--output_dir",
                str(dataset_dir),
                "--n_steps_per_motion",
                str(args.steps),
                "--num_workers",
                str(args.num_workers),
                "--continue_on_error",
                "--label_mode",
                variant.label_mode,
                "--pin_student_state",
                "--prosthesis_torque_limits",
                variant.torque_limits,
                "--prosthesis_action_type",
                variant.action_type,
            ]
            if args.skip_existing:
                collect.append("--skip_existing")
            if variant.residual_pd_kp:
                collect.extend(["--residual_pd_kp", variant.residual_pd_kp])
            if variant.residual_pd_kd:
                collect.extend(["--residual_pd_kd", variant.residual_pd_kd])
        else:
            collect = collect_cmd(args, variant, dataset_dir, [args.motion_path])
        ok_collect, collect_tail = run_cmd(collect, cwd=cwd, log_path=root / "logs" / f"{variant.name}.collect.log")
        ok_diag = False
        diag_tail = ""
        if ok_collect and variant.name != "scale_10motion_pd_residual":
            ok_diag, diag_tail = run_cmd(
                diagnose_cmd(args, variant, dataset_dir, diag_dir),
                cwd=cwd,
                log_path=root / "logs" / f"{variant.name}.diagnose.log",
            )
        rows.append(
            row_from_summary(
                variant,
                dataset_dir,
                diag_dir,
                ok_collect,
                ok_diag,
                "" if ok_collect and (ok_diag or variant.name == "scale_10motion_pd_residual") else (collect_tail + diag_tail),
            )
        )
        write_summary(root / "sweep_summary.csv", rows)
    print(f"Wrote {root / 'sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
