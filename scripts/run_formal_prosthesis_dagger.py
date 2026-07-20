#!/usr/bin/env python3
"""Run formal beta-decay DAgger for 343-dim prosthesis distillation (remaining + 4DOF).

Pipeline per round i=1..N:
  1. Collect student-closed-loop rollouts with beta_i expert mix.
  2. Train on BC dataset + all accumulated DAgger rounds (dataset aggregation).
  3. Use the new checkpoint as the student for round i+1.

Round 0 (optional): BC-only supervised pretrain on --bc_dataset_dir.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from musclemimic.distill.config import repo_root
from musclemimic.distill.dagger_schedule import beta_for_round, schedule_preview


DEFAULT_BC_DIR = "outputs/_nonformal_runs/knee15_bc_kit3_only"
DEFAULT_MOTION = "KIT/3/walk_6m_straight_line04_poses"
DEFAULT_CONFIG = "conf_fullbody_prosthesis_gmr_resnet_knee15_smooth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", default="outputs/_nonformal_runs/formal_prosthesis_dagger_knee15")
    parser.add_argument("--bc_dataset_dir", default=DEFAULT_BC_DIR)
    parser.add_argument("--motion_path", nargs="*", default=[DEFAULT_MOTION])
    parser.add_argument("--num_dagger_rounds", type=int, default=3)
    parser.add_argument("--skip_bc_pretrain", action="store_true")
    parser.add_argument("--init_checkpoint", default=None, help="Skip BC pretrain and start DAgger from this checkpoint")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--mask-preset", default="knee15", choices=["strict19", "knee15", "distal11", "foot5", "foot1"])
    parser.add_argument("--prosthesis_action_type", default="pd_residual_torque")
    parser.add_argument(
        "--label_mode",
        choices=["reference_pd", "teacher_qfrc", "inverse_dynamics"],
        default="teacher_qfrc",
    )
    parser.add_argument(
        "--beta_schedule",
        choices=["constant", "inverse", "inverse_high_first", "exponential", "polynomial", "linear"],
        default="inverse_high_first",
    )
    parser.add_argument("--beta_constant", type=float, default=0.3)
    parser.add_argument("--beta_decay", type=float, default=0.7)
    parser.add_argument("--beta_min", type=float, default=0.2, help="Floor for auto beta schedules (avoid rollout collapse)")
    parser.add_argument("--bc_epochs", type=int, default=80)
    parser.add_argument("--dagger_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bc_lr", type=float, default=1e-4)
    parser.add_argument("--dagger_lr", type=float, default=5e-5)
    parser.add_argument("--lambda_muscle", type=float, default=0.1)
    parser.add_argument("--lambda_prosthesis", type=float, default=5.0)
    parser.add_argument("--lambda_tau", type=float, default=8.0)
    parser.add_argument("--teacher_checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue in work_dir: skip BC, load prior dagger dirs, start at last completed round + 1.",
    )
    parser.add_argument("--start_round", type=int, default=0, help="Resume from this round (0=BC, 1+=DAgger)")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def latest_checkpoint(train_output_dir: Path) -> Path:
    latest = train_output_dir / "latest" / "checkpoints" / "checkpoint_distilled"
    if latest.exists():
        return latest
    runs = sorted((train_output_dir).glob("*/checkpoints/checkpoint_distilled"))
    if not runs:
        raise FileNotFoundError(f"No checkpoint found under {train_output_dir}")
    return runs[-1]


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n>>> {printable}\n", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    subprocess.run(cmd, cwd=str(repo_root()), env=env, check=True)


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def infer_resume_state(work_dir: Path) -> tuple[int, Path | None, list[Path]]:
    """Return (next_round, student_checkpoint, accumulated_dagger_dirs) from an existing work_dir."""
    dagger_dirs: list[Path] = []
    last_train_round = 0
    student_ckpt: Path | None = None
    for round_idx in range(1, 100):
        dagger_dir = work_dir / f"dagger_round_{round_idx:02d}"
        train_dir = work_dir / f"train_round_{round_idx:02d}"
        if not dagger_dir.exists() or not train_dir.exists():
            break
        dagger_dirs.append(dagger_dir)
        last_train_round = round_idx
        student_ckpt = latest_checkpoint(train_dir)
    if last_train_round == 0:
        bc_dir = work_dir / "train_round_00_bc"
        if bc_dir.exists():
            return 1, latest_checkpoint(bc_dir), []
    return last_train_round + 1, student_ckpt, dagger_dirs


def main() -> int:
    args = parse_args()
    work_dir = resolve(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    bc_dir = resolve(args.bc_dataset_dir)
    if not bc_dir.exists():
        raise FileNotFoundError(f"BC dataset dir not found: {bc_dir}")

    student_ckpt: Path | None = resolve(args.init_checkpoint) if args.init_checkpoint else None
    dagger_dirs: list[Path] = []

    if args.resume:
        next_round, resumed_ckpt, resumed_dagger = infer_resume_state(work_dir)
        if resumed_ckpt is None:
            raise FileNotFoundError(f"--resume requested but no prior checkpoints found under {work_dir}")
        args.skip_bc_pretrain = True
        args.start_round = max(int(args.start_round), next_round)
        if student_ckpt is None:
            student_ckpt = resumed_ckpt
        dagger_dirs = list(resumed_dagger)
        print(
            f"Resume: next_dagger_round={args.start_round}, "
            f"accumulated_dagger={len(dagger_dirs)}, student={student_ckpt}"
        )

    preview = schedule_preview(
        args.num_dagger_rounds,
        schedule=args.beta_schedule,
        constant=args.beta_constant,
        decay=args.beta_decay,
        beta_min=args.beta_min,
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "work_dir": str(work_dir),
        "bc_dataset_dir": str(bc_dir),
        "motion_paths": list(args.motion_path),
        "num_dagger_rounds": int(args.num_dagger_rounds),
        "beta_schedule": args.beta_schedule,
        "beta_preview": [{"round": r, "beta": b} for r, b in preview],
        "rounds": [],
    }
    print("Formal DAgger beta schedule:")
    for round_idx, beta in preview:
        print(f"  round {round_idx}: beta={beta:.4f}")

    if not args.skip_bc_pretrain and args.start_round <= 0 and student_ckpt is None:
        train_bc_dir = work_dir / "train_round_00_bc"
        train_cmd = [
            sys.executable,
            "scripts/train_prosthesis_distillation.py",
            "--dataset_dir",
            str(bc_dir),
            "--config-name",
            args.config_name,
            "--mask-preset",
            args.mask_preset,
            "--prosthesis_action_type",
            args.prosthesis_action_type,
            "--output_dir",
            str(train_bc_dir),
            "--epochs",
            str(args.bc_epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.bc_lr),
            "--lambda_muscle",
            str(args.lambda_muscle),
            "--lambda_prosthesis",
            str(args.lambda_prosthesis),
            "--lambda_tau",
            str(args.lambda_tau),
            "--init",
            "official_trunk_init",
            "--teacher_checkpoint",
            args.teacher_checkpoint,
            "--seed",
            str(args.seed),
        ]
        run_cmd(train_cmd, dry_run=args.dry_run)
        student_ckpt = latest_checkpoint(train_bc_dir)
        manifest["rounds"].append(
            {
                "round": 0,
                "kind": "bc_pretrain",
                "checkpoint": str(student_ckpt),
                "train_output_dir": str(train_bc_dir),
            }
        )
    elif student_ckpt is None:
        student_ckpt = latest_checkpoint(work_dir / "train_round_00_bc")

    for round_idx in range(1, int(args.num_dagger_rounds) + 1):
        if round_idx < int(args.start_round):
            train_dir = work_dir / f"train_round_{round_idx:02d}"
            if train_dir.exists():
                student_ckpt = latest_checkpoint(train_dir)
            if not args.resume:
                dagger_dir = work_dir / f"dagger_round_{round_idx:02d}"
                if dagger_dir.exists() and dagger_dir not in dagger_dirs:
                    dagger_dirs.append(dagger_dir)
            continue

        beta = beta_for_round(
            round_idx,
            schedule=args.beta_schedule,
            constant=args.beta_constant,
            decay=args.beta_decay,
            beta_min=args.beta_min,
            max_rounds=args.num_dagger_rounds,
        )
        dagger_dir = work_dir / f"dagger_round_{round_idx:02d}"
        collect_cmd = [
            sys.executable,
            "scripts/collect_split_action_prosthesis_dagger_rollouts.py",
            "--student_checkpoint",
            str(student_ckpt),
            "--teacher_checkpoint",
            args.teacher_checkpoint,
            "--motion_path",
            *args.motion_path,
            "--output_dir",
            str(dagger_dir),
            "--config-name",
            args.config_name,
            "--mask-preset",
            args.mask_preset,
            "--prosthesis_action_type",
            args.prosthesis_action_type,
            "--label_mode",
            args.label_mode,
            "--dagger_round",
            str(round_idx),
            "--beta_schedule",
            args.beta_schedule,
            "--beta_constant",
            str(args.beta_constant),
            "--beta_decay",
            str(args.beta_decay),
            "--beta_min",
            str(args.beta_min),
            "--beta_max_rounds",
            str(args.num_dagger_rounds),
            "--seed",
            str(args.seed + round_idx),
        ]
        run_cmd(collect_cmd, dry_run=args.dry_run)
        dagger_dirs.append(dagger_dir)

        train_dir = work_dir / f"train_round_{round_idx:02d}"
        train_cmd = [
            sys.executable,
            "scripts/train_prosthesis_distillation.py",
            "--dataset_dir",
            str(bc_dir),
            *[item for d in dagger_dirs for item in ("--extra_dataset_dir", str(d))],
            "--val_dataset_dir",
            str(dagger_dir),
            "--config-name",
            args.config_name,
            "--mask-preset",
            args.mask_preset,
            "--prosthesis_action_type",
            args.prosthesis_action_type,
            "--output_dir",
            str(train_dir),
            "--epochs",
            str(args.dagger_epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.dagger_lr),
            "--lambda_muscle",
            str(args.lambda_muscle),
            "--lambda_prosthesis",
            str(args.lambda_prosthesis),
            "--lambda_tau",
            str(args.lambda_tau),
            "--init",
            "checkpoint",
            "--init_checkpoint",
            str(student_ckpt),
            "--seed",
            str(args.seed + 100 + round_idx),
        ]
        run_cmd(train_cmd, dry_run=args.dry_run)
        if not args.dry_run:
            student_ckpt = latest_checkpoint(train_dir)
        manifest["rounds"].append(
            {
                "round": round_idx,
                "kind": "dagger",
                "beta": beta,
                "dagger_dir": str(dagger_dir),
                "train_output_dir": str(train_dir),
                "checkpoint": str(student_ckpt),
                "accumulated_dagger_dirs": [str(p) for p in dagger_dirs],
            }
        )
        write_manifest(work_dir / "manifest.json", manifest)

    manifest["final_checkpoint"] = str(student_ckpt)
    write_manifest(work_dir / "manifest.json", manifest)
    print(f"\nDone. manifest: {work_dir / 'manifest.json'}")
    print(f"Final checkpoint: {student_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
