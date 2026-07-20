#!/usr/bin/env python
"""Plot left/right GRF and knee angles from locomotion eval rollout_data.npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from musclemimic.distill.config import load_fullbody_config, make_env, repo_root

KNEE_JOINTS = ("knee_angle_l", "knee_angle_r")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--motion_path", default=None)
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    return p.parse_args()


def knee_qpos_indices(motion_path: str | None, config_name: str) -> dict[str, int]:
    motion = motion_path or "KIT/3/walk_6m_straight_line04_poses"
    cfg = load_fullbody_config(config_name)
    env = make_env(cfg, motion_paths=[motion], use_mujoco=True)
    try:
        out = {}
        for name in KNEE_JOINTS:
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(name)
            out[name] = int(env.model.jnt_qposadr[jid])
        return out
    finally:
        env.stop()


def load_rollout(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_grf(time, left, right, out_path: Path, raw_l=None, raw_r=None, ref_l=None, ref_r=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, left, color="#2563eb", lw=1.4, label="filtered left vGRF")
    if raw_l is not None:
        axes[0].plot(time, raw_l, color="#2563eb", alpha=0.35, ls="--", lw=0.9, label="raw left vGRF")
    if ref_l is not None:
        axes[0].plot(time[: len(ref_l)], ref_l, color="#93c5fd", ls=":", lw=1.0, label="reference")
    axes[0].set_ylabel("N")
    axes[0].set_title("Left foot vertical GRF (world z)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, right, color="#dc2626", lw=1.4, label="filtered right vGRF")
    if raw_r is not None:
        axes[1].plot(time, raw_r, color="#dc2626", alpha=0.35, ls="--", lw=0.9, label="raw right vGRF")
    if ref_r is not None:
        axes[1].plot(time[: len(ref_r)], ref_r, color="#fca5a5", ls=":", lw=1.0, label="reference")
    axes[1].set_ylabel("N")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Right foot vertical GRF (world z)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("GRF — filtered default; raw dashed overlay", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_knee(time, knee_l, knee_r, ref_l=None, ref_r=None, out_path: Path | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, np.rad2deg(knee_l), color="#2563eb", lw=1.2, label="policy knee_angle_l")
    if ref_l is not None:
        axes[0].plot(time[: len(ref_l)], np.rad2deg(ref_l), color="#93c5fd", ls="--", lw=1.0, label="reference")
    axes[0].set_ylabel("deg")
    axes[0].set_title("Left knee (prosthesis side in prosthesis env)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, np.rad2deg(knee_r), color="#dc2626", lw=1.2, label="policy knee_angle_r")
    if ref_r is not None:
        axes[1].plot(time[: len(ref_r)], np.rad2deg(ref_r), color="#fca5a5", ls="--", lw=1.0, label="reference")
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Right knee (contralateral / symmetry leg)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Knee flexion — check periodic gait cycles", fontsize=11)
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grf_overlay(time, left, right, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, left, color="#2563eb", lw=1.2, label="left vGRF")
    ax.plot(time, right, color="#dc2626", lw=1.2, label="right vGRF")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N")
    ax.set_title("Left vs right GRF overlay (symmetry / phase)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rollout_path = Path(args.rollout_npz)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_rollout(rollout_path)
    motion_path = str(data.get("motion_path", args.motion_path or ""))
    if hasattr(motion_path, "item"):
        motion_path = motion_path.item()
    dt = float(np.asarray(data["dt"]).reshape(-1)[0])
    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    left_grf = np.asarray(data["left_GRF"], dtype=np.float64).reshape(-1)
    right_grf = np.asarray(data["right_GRF"], dtype=np.float64).reshape(-1)
    left_raw = np.asarray(data["left_GRF_raw"], dtype=np.float64).reshape(-1) if "left_GRF_raw" in data else None
    right_raw = np.asarray(data["right_GRF_raw"], dtype=np.float64).reshape(-1) if "right_GRF_raw" in data else None
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    ref_qpos = np.asarray(data["reference_qpos"], dtype=np.float64) if "reference_qpos" in data else None

    knee_idx = knee_qpos_indices(motion_path or None, args.config_name)
    knee_l = qpos[:, knee_idx["knee_angle_l"]]
    knee_r = qpos[:, knee_idx["knee_angle_r"]]
    ref_knee_l = ref_knee_r = None
    if ref_qpos is not None and ref_qpos.size:
        ref_knee_l = ref_qpos[:, knee_idx["knee_angle_l"]]
        ref_knee_r = ref_qpos[:, knee_idx["knee_angle_r"]]

    plot_grf(time, left_grf, right_grf, out_dir / "grf_left_right.png", raw_l=left_raw, raw_r=right_raw)
    plot_grf(time, left_grf, right_grf, out_dir / "grf_vertical_panels.png", raw_l=left_raw, raw_r=right_raw)
    plot_grf_overlay(time, left_grf, right_grf, out_dir / "grf_overlay_symmetry.png")
    plot_knee(time, knee_l, knee_r, ref_knee_l, ref_knee_r, out_dir / "knee_angle_left_right.png")

    # Separate single-panel exports for quick inspection
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, left_grf, color="#2563eb", lw=1.4, label="filtered")
    if left_raw is not None:
        ax.plot(time, left_raw, color="#2563eb", alpha=0.35, ls="--", lw=0.9, label="raw")
    ax.set_title("Left vertical GRF")
    ax.legend(fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "grf_left_only.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, right_grf, color="#dc2626", lw=1.4, label="filtered")
    if right_raw is not None:
        ax.plot(time, right_raw, color="#dc2626", alpha=0.35, ls="--", lw=0.9, label="raw")
    ax.set_title("Right vertical GRF")
    ax.legend(fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "grf_right_only.png", dpi=150)
    plt.close(fig)

    summary = {
        "motion_path": motion_path,
        "duration_s": float(time[-1]) if time.size else 0.0,
        "n_steps": int(time.size),
        "dt": dt,
        "peak_vGRF_left": float(np.max(left_grf)),
        "peak_vGRF_right": float(np.max(right_grf)),
        "knee_l_range_deg": [float(np.rad2deg(knee_l.min())), float(np.rad2deg(knee_l.max()))],
        "knee_r_range_deg": [float(np.rad2deg(knee_r.min())), float(np.rad2deg(knee_r.max()))],
        "contact_fraction_left": float(np.mean(left_grf > 1.0)),
        "contact_fraction_right": float(np.mean(right_grf > 1.0)),
    }
    with (out_dir / "leg_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
