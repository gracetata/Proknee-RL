#!/usr/bin/env python
"""Plot left vs right leg angles and joint moments from locomotion rollout."""

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

# Primary hinge DOFs for comparison (left / right qposadr, dofadr).
LEG_SIGNALS = {
    "knee_angle": ("knee_angle_l", "knee_angle_r"),
    "hip_flexion": ("hip_flexion_l", "hip_flexion_r"),
    "ankle_angle": ("ankle_angle_l", "ankle_angle_r"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--motion_path", default="KIT/3/walk_6m_straight_line04_poses")
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    p.add_argument(
        "--only-dashboard",
        action="store_true",
        help="Only regenerate leg_angles_dashboard.png",
    )
    return p.parse_args()


def resolve_leg_indices(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, (jl, jr) in LEG_SIGNALS.items():
        out[key] = {}
        for side, jname in (("left", jl), ("right", jr)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"Joint not found: {jname}")
            out[key][side] = {
                "qpos": int(model.jnt_qposadr[jid]),
                "dof": int(model.jnt_dofadr[jid]),
                "name": jname,
            }
    return out


def load_series(npz_path: Path, indices: dict) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    ref_qpos = np.asarray(data["reference_qpos"], dtype=np.float64) if "reference_qpos" in data else None
    tau = np.asarray(data["joint_torque"], dtype=np.float64)

    series: dict[str, dict[str, np.ndarray]] = {}
    for signal, sides in indices.items():
        series[signal] = {}
        for side, idx in sides.items():
            qp = idx["qpos"]
            df = idx["dof"]
            angle_rad = qpos[:, qp]
            series[signal][f"{side}_angle_rad"] = angle_rad
            series[signal][f"{side}_angle_deg"] = np.rad2deg(angle_rad)
            series[signal][f"{side}_moment"] = tau[:, df]
            if ref_qpos is not None and ref_qpos.size:
                series[signal][f"{side}_ref_angle_deg"] = np.rad2deg(ref_qpos[:, qp])
    return {"time": time, "signals": series, "motion_path": str(data.get("motion_path", ""))}


def plot_angle_comparison(
    time: np.ndarray,
    left_deg: np.ndarray,
    right_deg: np.ndarray,
    ref_l: np.ndarray | None,
    ref_r: np.ndarray | None,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, left_deg, color="#2563eb", lw=1.3, label="left policy")
    if ref_l is not None:
        axes[0].plot(time, ref_l, color="#93c5fd", ls="--", lw=1.0, label="left reference")
    axes[0].set_ylabel("deg")
    axes[0].set_title(f"{title} — left")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, right_deg, color="#dc2626", lw=1.3, label="right policy")
    if ref_r is not None:
        axes[1].plot(time, ref_r, color="#fca5a5", ls="--", lw=1.0, label="right reference")
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title(f"{title} — right")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{title} (left vs right panels)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overlay_comparison(
    time: np.ndarray,
    left_y: np.ndarray,
    right_y: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, left_y, color="#2563eb", lw=1.3, label="left")
    ax.plot(time, right_y, color="#dc2626", lw=1.3, label="right")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_fullbody_config(args.config_name)
    env = make_env(cfg, motion_paths=[args.motion_path], use_mujoco=True)
    try:
        indices = resolve_leg_indices(env.model)
        loaded = load_series(Path(args.rollout_npz), indices)
    finally:
        env.stop()

    time = loaded["time"]
    signals = loaded["signals"]

    if args.only_dashboard:
        # Combined angle dashboard only
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        panels = [
            ("knee_angle", "Knee angle (deg)"),
            ("hip_flexion", "Hip flexion (deg)"),
            ("ankle_angle", "Ankle angle (deg)"),
        ]
        for row, (key, ylab) in enumerate(panels):
            s = signals[key]
            ref_l = s.get("left_ref_angle_deg")
            ref_r = s.get("right_ref_angle_deg")
            axes[row, 0].plot(time, s["left_angle_deg"], color="#2563eb", lw=1.2, label="policy")
            if ref_l is not None:
                axes[row, 0].plot(time, ref_l, color="#93c5fd", ls="--", lw=1.0, label="reference")
            axes[row, 0].set_ylabel(ylab)
            axes[row, 0].set_title(f"{ylab} — left")
            axes[row, 0].legend(loc="upper right", fontsize=7)
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 1].plot(time, s["right_angle_deg"], color="#dc2626", lw=1.2, label="policy")
            if ref_r is not None:
                axes[row, 1].plot(time, ref_r, color="#fca5a5", ls="--", lw=1.0, label="reference")
            axes[row, 1].set_title(f"{ylab} — right")
            axes[row, 1].legend(loc="upper right", fontsize=7)
            axes[row, 1].grid(True, alpha=0.3)
        axes[2, 0].set_xlabel("time (s)")
        axes[2, 1].set_xlabel("time (s)")
        fig.suptitle("Leg angles — policy vs reference motion", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "leg_angles_dashboard.png", dpi=150)
        plt.close(fig)
        print(f"Dashboard saved to {out_dir / 'leg_angles_dashboard.png'}")
        return 0

    summary = {"motion_path": loaded["motion_path"], "signals": {}}

    angle_plots = [
        ("knee_angle", "Knee flexion angle", "knee_angle_left_right.png"),
        ("hip_flexion", "Hip flexion angle", "hip_flexion_left_right.png"),
        ("ankle_angle", "Ankle angle", "ankle_angle_left_right.png"),
    ]
    moment_plots = [
        ("ankle_angle", "Ankle joint moment (qfrc_actuator)", "ankle_moment_left_right.png"),
        ("hip_flexion", "Hip flexion moment (qfrc_actuator)", "hip_flexion_moment_left_right.png"),
        ("knee_angle", "Knee moment (qfrc_actuator)", "knee_moment_left_right.png"),
    ]

    for key, title, fname in angle_plots:
        s = signals[key]
        plot_angle_comparison(
            time,
            s["left_angle_deg"],
            s["right_angle_deg"],
            s.get("left_ref_angle_deg"),
            s.get("right_ref_angle_deg"),
            title,
            out_dir / fname,
        )
        summary["signals"][key] = {
            "left_range_deg": [float(np.min(s["left_angle_deg"])), float(np.max(s["left_angle_deg"]))],
            "right_range_deg": [float(np.min(s["right_angle_deg"])), float(np.max(s["right_angle_deg"]))],
        }

    for key, title, fname in moment_plots:
        s = signals[key]
        plot_angle_comparison(
            time,
            s["left_moment"],
            s["right_moment"],
            None,
            None,
            title,
            out_dir / fname,
        )
        summary["signals"][f"{key}_moment"] = {
            "left_peak_Nm": float(np.max(np.abs(s["left_moment"]))),
            "right_peak_Nm": float(np.max(np.abs(s["right_moment"]))),
        }

    # Combined angle dashboard (policy solid + reference dashed)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    panels = [
        ("knee_angle", "Knee angle (deg)"),
        ("hip_flexion", "Hip flexion (deg)"),
        ("ankle_angle", "Ankle angle (deg)"),
    ]
    for row, (key, ylab) in enumerate(panels):
        s = signals[key]
        ref_l = s.get("left_ref_angle_deg")
        ref_r = s.get("right_ref_angle_deg")
        axes[row, 0].plot(time, s["left_angle_deg"], color="#2563eb", lw=1.2, label="policy")
        if ref_l is not None:
            axes[row, 0].plot(time, ref_l, color="#93c5fd", ls="--", lw=1.0, label="reference")
        axes[row, 0].set_ylabel(ylab)
        axes[row, 0].set_title(f"{ylab} — left")
        axes[row, 0].legend(loc="upper right", fontsize=7)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(time, s["right_angle_deg"], color="#dc2626", lw=1.2, label="policy")
        if ref_r is not None:
            axes[row, 1].plot(time, ref_r, color="#fca5a5", ls="--", lw=1.0, label="reference")
        axes[row, 1].set_title(f"{ylab} — right")
        axes[row, 1].legend(loc="upper right", fontsize=7)
        axes[row, 1].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 1].set_xlabel("time (s)")
    fig.suptitle("Leg angles — policy vs reference motion", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "leg_angles_dashboard.png", dpi=150)
    plt.close(fig)

    with (out_dir / "leg_dynamics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Leg dynamics plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
