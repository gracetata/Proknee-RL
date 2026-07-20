"""Per-motion and dataset summary plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from musclemimic.evaluation.types import RolloutBuffer


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_per_motion(buffer: RolloutBuffer, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    t = np.asarray(buffer.time, dtype=np.float64)
    if t.size == 0:
        return paths

    lv = np.asarray(buffer.left_grf, dtype=np.float64)
    rv = np.asarray(buffer.right_grf, dtype=np.float64)
    lv_raw = np.asarray(buffer.left_grf_raw, dtype=np.float64) if buffer.left_grf_raw else None
    rv_raw = np.asarray(buffer.right_grf_raw, dtype=np.float64) if buffer.right_grf_raw else None

    def _plot_grf_panel(ax, t_arr, filt, raw, title, color):
        ax.plot(t_arr, filt, color=color, lw=1.4, label="filtered vGRF")
        if raw is not None and raw.size == filt.size:
            ax.plot(t_arr, raw, color=color, alpha=0.35, lw=0.9, ls="--", label="raw vGRF")
        ax.set_ylabel("N")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _plot_grf_panel(axes[0], t, lv, lv_raw, "Left vertical GRF (filtered + raw)", "#2563eb")
    _plot_grf_panel(axes[1], t, rv, rv_raw, "Right vertical GRF (filtered + raw)", "#dc2626")
    axes[1].set_xlabel("time (s)")
    fig.suptitle("Vertical GRF — stance peaks from filtered curve", fontsize=11)
    p = out_dir / "grf_vertical.png"
    _savefig(p)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, lv, color="#2563eb", lw=1.4, label="left filtered")
    ax.plot(t, rv, color="#dc2626", lw=1.4, label="right filtered")
    if lv_raw is not None:
        ax.plot(t, lv_raw, color="#2563eb", alpha=0.3, ls="--", lw=0.8, label="left raw")
    if rv_raw is not None:
        ax.plot(t, rv_raw, color="#dc2626", alpha=0.3, ls="--", lw=0.8, label="right raw")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N")
    ax.legend()
    ax.set_title("GRF overlay (symmetry)")
    ax.grid(True, alpha=0.3)
    p = out_dir / "grf_overlay.png"
    _savefig(p)
    paths.append(p)

    if buffer.root_pos:
        root = np.stack([np.asarray(x) for x in buffer.root_pos], axis=0)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, root[:, 0], label="x")
        ax.plot(t, root[:, 1], label="y")
        ax.plot(t, root[:, 2], label="z")
        ax.set_title("Root position")
        ax.legend()
        p = out_dir / "root_position.png"
        _savefig(p)
        paths.append(p)

    if buffer.info_steps:
        keys = ["err_joint_pos", "err_rpos", "err_site_abs"]
        for key in keys:
            vals = [float(s[key]) for s in buffer.info_steps if key in s]
            if not vals:
                continue
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(np.arange(len(vals)) * buffer.dt, vals)
            ax.set_title(key)
            ax.set_xlabel("time (s)")
            p = out_dir / f"{key}.png"
            _savefig(p)
            paths.append(p)

    if buffer.prosthesis_tau and np.asarray(buffer.prosthesis_tau[0]).size:
        tau = np.stack([np.asarray(x) for x in buffer.prosthesis_tau], axis=0)
        fig, ax = plt.subplots(figsize=(8, 3))
        for j in range(tau.shape[1]):
            ax.plot(t[: tau.shape[0]], tau[:, j], label=f"joint {j}")
        ax.set_title("Prosthesis torque")
        ax.legend(fontsize=8)
        p = out_dir / "prosthesis_torque.png"
        _savefig(p)
        paths.append(p)

    return paths


def plot_dataset_summary(rows: list[dict[str, Any]], out_dir: Path, metric_keys: list[str] | None = None) -> list[Path]:
    """Boxplots over flattened metric columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    keys = metric_keys or [
        "official_imitation.joint_angle_error_deg.mean.value",
        "official_imitation.root_position_error_cm.mean.value",
        "force.peak_vGRF_left.value",
        "force.GRF_symmetry_index_peak.value",
    ]
    paths: list[Path] = []
    for key in keys:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(vals)
        ax.set_title(key.split(".")[-2] if "." in key else key)
        ax.set_ylabel("value")
        p = out_dir / f"summary_{key.replace('.', '_')}.png"
        _savefig(p)
        paths.append(p)
    return paths
