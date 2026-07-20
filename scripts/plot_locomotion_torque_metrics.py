#!/usr/bin/env python
"""Plot joint torque / prosthesis control metrics (simulator.md §4) — left vs right panels."""

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

SECTION_SUBDIR = "section4_torque"


def resolve_out_dir(base: Path) -> Path:
    base = Path(base)
    if not base.is_absolute():
        base = repo_root() / base
    if base.name == SECTION_SUBDIR:
        return base
    return base / SECTION_SUBDIR


LEG_SIGNALS = {
    "knee": ("knee_angle_l", "knee_angle_r"),
    "hip": ("hip_flexion_l", "hip_flexion_r"),
    "ankle": ("ankle_angle_l", "ankle_angle_r"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--motion_path", default="KIT/3/walk_6m_straight_line04_poses")
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    p.add_argument(
        "--torque-limits",
        type=float,
        nargs="*",
        default=None,
        help="Optional per-joint |tau| limits (knee hip ankle) for violation shading",
    )
    return p.parse_args()


def resolve_leg_dof(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, (jl, jr) in LEG_SIGNALS.items():
        out[key] = {}
        for side, jname in (("left", jl), ("right", jr)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"Joint not found: {jname}")
            out[key][side] = int(model.jnt_dofadr[jid])
    return out


def plot_left_right_panels(
    time: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    limit: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, y, side, color in (
        (axes[0], left, "left", "#2563eb"),
        (axes[1], right, "right", "#dc2626"),
    ):
        ax.plot(time, y, color=color, lw=1.3)
        if limit is not None and limit > 0:
            ax.axhline(limit, color="gray", ls=":", lw=0.8, alpha=0.7)
            ax.axhline(-limit, color="gray", ls=":", lw=0.8, alpha=0.7)
            viol = np.abs(y) > limit
            if np.any(viol):
                ax.fill_between(time, np.min(y), np.max(y), where=viol, color="red", alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {side}")
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"{title} (left / right panels)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_work_panels(
    time: np.ndarray,
    pos_l: np.ndarray,
    neg_l: np.ndarray,
    pos_r: np.ndarray,
    neg_r: np.ndarray,
    *,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, pos_l, color="#16a34a", lw=1.2, label="positive work")
    axes[0].plot(time, neg_l, color="#ca8a04", lw=1.2, label="negative work")
    axes[0].set_ylabel("J (cumulative)")
    axes[0].set_title(f"{title} — left")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time, pos_r, color="#16a34a", lw=1.2, label="positive work")
    axes[1].plot(time, neg_r, color="#ca8a04", lw=1.2, label="negative work")
    axes[1].set_ylabel("J (cumulative)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title(f"{title} — right")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{title} (left / right panels)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def lowpass_torque(tau: np.ndarray, dt: float, cutoff_hz: float = 6.0) -> np.ndarray:
    """Causal-ish low-pass for display metrics (muscle qfrc_actuator is noisy at 100 Hz)."""
    x = np.asarray(tau, dtype=np.float64).reshape(-1)
    if x.size < 8 or cutoff_hz <= 0:
        return x
    try:
        from scipy.signal import butter, filtfilt

        b, a = butter(2, min(cutoff_hz, 0.49 / max(dt, 1e-9)), fs=1.0 / max(dt, 1e-9), btype="low")
        return filtfilt(b, a, x)
    except Exception:
        return x


def torque_smoothness(tau: np.ndarray) -> np.ndarray:
    """simulator.md §4.3: |τ_t - τ_{t-1}| in Nm per step (not divided by dt)."""
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau.size < 2:
        return np.zeros_like(tau)
    return np.abs(np.diff(tau, prepend=tau[0]))


def torque_jerk(tau: np.ndarray) -> np.ndarray:
    """simulator.md §4.4: |τ_{t+1} - 2τ_t + τ_{t-1}| in Nm (not divided by dt²)."""
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau.size < 3:
        return np.zeros_like(tau)
    j = np.zeros_like(tau)
    j[1:-1] = np.abs(tau[2:] - 2 * tau[1:-1] + tau[:-2])
    j[0] = j[1]
    j[-1] = j[-2]
    return j


def mechanical_power(tau: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    return np.asarray(tau, dtype=np.float64) * np.asarray(qvel, dtype=np.float64)


def cumulative_work(power: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(power, dtype=np.float64)
    pos_inc = np.maximum(p, 0.0) * dt
    neg_inc = np.minimum(p, 0.0) * dt
    return np.cumsum(pos_inc), np.cumsum(neg_inc)


def joint_summary(tau: np.ndarray, qvel: np.ndarray, dt: float, limit: float | None) -> dict:
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    tau_f = lowpass_torque(tau, dt)
    qvel = np.asarray(qvel, dtype=np.float64).reshape(-1)
    power = mechanical_power(tau_f, qvel)
    pos_w, neg_w = cumulative_work(power, dt)
    out = {
        "torque_source": "qfrc_actuator (muscle actuator torque at DOF)",
        "torque_RMS_Nm": float(np.sqrt(np.mean(tau**2))),
        "peak_torque_Nm": float(np.max(np.abs(tau))),
        "torque_smoothness_mean_Nm_per_step": float(np.mean(torque_smoothness(tau_f))),
        "torque_jerk_mean_Nm": float(np.mean(torque_jerk(tau_f))),
        "positive_work_J_filtered_tau": float(pos_w[-1]) if pos_w.size else 0.0,
        "negative_work_J_filtered_tau": float(neg_w[-1]) if neg_w.size else 0.0,
        "net_work_J_filtered_tau": float(np.sum(power) * dt),
    }
    if limit is not None and limit > 0:
        out["torque_limit_violation_rate"] = float(np.mean(np.abs(tau) > limit))
    return out


def symmetry_index(a: float, b: float) -> float:
    denom = 0.5 * (abs(a) + abs(b))
    return float(abs(a - b) / denom * 100.0) if denom > 1e-6 else 0.0


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_fullbody_config(args.config_name)
    env = make_env(cfg, motion_paths=[args.motion_path], use_mujoco=True)
    try:
        dof_idx = resolve_leg_dof(env.model)
    finally:
        env.stop()

    data = np.load(Path(args.rollout_npz), allow_pickle=True)
    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    dt = float(np.asarray(data["dt"]).reshape(-1)[0]) if "dt" in data else float(np.median(np.diff(time)) if time.size > 1 else 0.01)
    tau_all = np.asarray(data["joint_torque"], dtype=np.float64)
    qvel_all = np.asarray(data["qvel"], dtype=np.float64)

    limits_by_joint: dict[str, float | None] = {k: None for k in LEG_SIGNALS}
    if args.torque_limits and len(args.torque_limits) == 3:
        for key, lim in zip(LEG_SIGNALS, args.torque_limits):
            limits_by_joint[key] = float(lim)

    summary: dict = {"joints": {}}

    summary_meta = {
        "note": (
            "τ from rollout joint_torque = MuJoCo qfrc_actuator (muscle torque at hinge DOF). "
            "Smoothness/jerk follow simulator.md per-step definitions; work/power use 6 Hz filtered τ."
        ),
    }

    for joint, sides in dof_idx.items():
        tau_l = lowpass_torque(tau_all[:, sides["left"]], dt)
        tau_r = lowpass_torque(tau_all[:, sides["right"]], dt)
        qv_l = qvel_all[:, sides["left"]]
        qv_r = qvel_all[:, sides["right"]]
        lim = limits_by_joint[joint]

        # 4.3 Torque smoothness
        plot_left_right_panels(
            time,
            torque_smoothness(tau_l),
            torque_smoothness(tau_r),
            title=f"{joint.capitalize()} torque smoothness (|Δτ|, 6 Hz filtered)",
            ylabel="Nm/step",
            out_path=out_dir / f"{joint}_torque_smoothness_left_right.png",
        )

        # 4.4 Torque jerk
        plot_left_right_panels(
            time,
            torque_jerk(tau_l),
            torque_jerk(tau_r),
            title=f"{joint.capitalize()} torque jerk (|Δ²τ|, 6 Hz filtered)",
            ylabel="Nm",
            out_path=out_dir / f"{joint}_torque_jerk_left_right.png",
        )

        # 4.6 Mechanical power
        pow_l = mechanical_power(tau_l, qv_l)
        pow_r = mechanical_power(tau_r, qv_r)
        plot_left_right_panels(
            time,
            pow_l,
            pow_r,
            title=f"{joint.capitalize()} mechanical power (filtered τ·q̇)",
            ylabel="W",
            out_path=out_dir / f"{joint}_mechanical_power_left_right.png",
        )

        # 4.7 Positive / negative work (cumulative)
        pos_l, neg_l = cumulative_work(pow_l, dt)
        pos_r, neg_r = cumulative_work(pow_r, dt)
        plot_work_panels(
            time,
            pos_l,
            neg_l,
            pos_r,
            neg_r,
            title=f"{joint.capitalize()} work (cumulative)",
            out_path=out_dir / f"{joint}_work_cumulative_left_right.png",
        )

        sl = joint_summary(tau_l, qv_l, dt, lim)
        sr = joint_summary(tau_r, qv_r, dt, lim)
        summary["joints"][joint] = {
            "left": sl,
            "right": sr,
            "symmetry_index_peak_torque_pct": symmetry_index(sl["peak_torque_Nm"], sr["peak_torque_Nm"]),
            "symmetry_index_torque_RMS_pct": symmetry_index(sl["torque_RMS_Nm"], sr["torque_RMS_Nm"]),
            "symmetry_index_positive_work_pct": symmetry_index(
                sl["positive_work_J_filtered_tau"], sr["positive_work_J_filtered_tau"]
            ),
        }

    summary = {**summary_meta, **summary}
    with (out_dir / "torque_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Torque metric plots saved to {out_dir}")
    for joint, block in summary["joints"].items():
        print(f"  [{joint}] peak L={block['left']['peak_torque_Nm']:.1f} R={block['right']['peak_torque_Nm']:.1f} Nm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
