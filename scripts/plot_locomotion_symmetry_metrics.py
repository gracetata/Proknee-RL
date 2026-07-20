#!/usr/bin/env python
"""Plot left-right symmetry metrics (simulator.md §5) — panel comparison style."""

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
from musclemimic.evaluation.contact_extractor import segment_stance_phases

LEG_SIGNALS = {
    "knee": ("knee_angle_l", "knee_angle_r"),
    "hip": ("hip_flexion_l", "hip_flexion_r"),
    "ankle": ("ankle_angle_l", "ankle_angle_r"),
}

SECTION_SUBDIR = "section5_symmetry"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_npz", required=True)
    p.add_argument("--output_dir", required=True, help="Parent analysis dir or final section dir")
    p.add_argument("--motion_path", default="KIT/3/walk_6m_straight_line04_poses")
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    return p.parse_args()


def resolve_out_dir(base: Path) -> Path:
    base = Path(base)
    if not base.is_absolute():
        base = repo_root() / base
    if base.name == SECTION_SUBDIR:
        return base
    return base / SECTION_SUBDIR


def resolve_foot_site_indices(sites_for_mimic: list[str]) -> dict[str, int]:
    names = {s: i for i, s in enumerate(sites_for_mimic)}
    out: dict[str, int] = {}
    for side, candidates in (
        ("left", ("left_toes_mimic", "left_ankle_mimic", "left_foot_mimic")),
        ("right", ("right_toes_mimic", "right_ankle_mimic", "right_foot_mimic")),
    ):
        for c in candidates:
            if c in names:
                out[side] = names[c]
                break
        else:
            raise KeyError(f"No foot mimic site for {side}; have {sites_for_mimic}")
    return out


def resolve_leg_qpos(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, (jl, jr) in LEG_SIGNALS.items():
        out[key] = {}
        for side, jname in (("left", jl), ("right", jr)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(jname)
            out[key][side] = int(model.jnt_qposadr[jid])
    return out


def symmetry_index(left: float, right: float) -> float:
    denom = 0.5 * (abs(left) + abs(right))
    return float(abs(left - right) / denom * 100.0) if denom > 1e-6 else 0.0


def forward_axis(pos_xy: np.ndarray) -> np.ndarray:
    """Unit vector along mean horizontal progression."""
    pos = np.asarray(pos_xy, dtype=np.float64)
    if pos.shape[0] < 2:
        return np.array([1.0, 0.0])
    d = pos[1:] - pos[:-1]
    mean_d = np.mean(d, axis=0)
    n = float(np.linalg.norm(mean_d))
    if n < 1e-6:
        return np.array([1.0, 0.0])
    return mean_d / n


def step_lengths_from_contact(
    foot_xy: np.ndarray,
    contact: np.ndarray,
    dt: float,
    *,
    min_stance_steps: int = 5,
) -> tuple[list[float], list[float]]:
    """Step length = forward foot-site displacement between consecutive stance onsets."""
    pos = np.asarray(foot_xy, dtype=np.float64)
    axis = forward_axis(pos)
    phases = segment_stance_phases(contact, min_steps=min_stance_steps)
    onsets = [s for s, _ in phases]
    lengths: list[float] = []
    durations: list[float] = []
    for i, (s, e) in enumerate(phases):
        durations.append((e - s) * dt)
        if i == 0:
            continue
        prev = onsets[i - 1]
        disp = pos[s] - pos[prev]
        lengths.append(float(np.dot(disp, axis)))
    return lengths, durations


def plot_stride_bars(
    values: list[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if values:
        x = np.arange(1, len(values) + 1)
        ax.bar(x, values, color=color, alpha=0.85, width=0.7)
        ax.axhline(float(np.mean(values)), color="gray", ls="--", lw=1.0, label=f"mean={np.mean(values):.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("stride index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_left_right_stride_panels(
    left_vals: list[float],
    right_vals: list[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for ax, vals, side, color in (
        (axes[0], left_vals, "left", "#2563eb"),
        (axes[1], right_vals, "right", "#dc2626"),
    ):
        if vals:
            x = np.arange(1, len(vals) + 1)
            ax.bar(x, vals, color=color, alpha=0.85, width=0.7)
            ax.axhline(float(np.mean(vals)), color="gray", ls="--", lw=1.0, label=f"mean={np.mean(vals):.3f}")
            ax.set_xticks(x)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {side}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].set_xlabel("stride index")
    fig.suptitle(f"{title} (left / right panels)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rom_panels(rom_left: dict[str, float], rom_right: dict[str, float], out_path: Path) -> None:
    joints = list(rom_left.keys())
    x = np.arange(len(joints))
    w = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, rom, side, color in (
        (axes[0], rom_left, "left", "#2563eb"),
        (axes[1], rom_right, "right", "#dc2626"),
    ):
        ax.bar(x, [rom[j] for j in joints], color=color, alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([j.capitalize() for j in joints])
        ax.set_ylabel("ROM (deg)")
        ax.set_title(f"Joint ROM — {side}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].set_xlabel("joint")
    fig.suptitle("Joint ROM symmetry (left / right panels)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grf_symmetry_panels(
    peak_l: float,
    peak_r: float,
    imp_l: float,
    imp_r: float,
    out_path: Path,
) -> None:
    labels = ["peak vGRF (N)", "GRF impulse (N·s)"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    for ax, vals, side, color in (
        (axes[0], [peak_l, imp_l], "left", "#2563eb"),
        (axes[1], [peak_r, imp_r], "right", "#dc2626"),
    ):
        ax.barh(labels, vals, color=color, alpha=0.85, height=0.5)
        ax.set_title(f"GRF symmetry — {side}")
        ax.grid(True, axis="x", alpha=0.3)
    si_peak = symmetry_index(peak_l, peak_r)
    si_imp = symmetry_index(imp_l, imp_r)
    fig.suptitle(f"GRF symmetry (SI peak={si_peak:.1f}%, impulse={si_imp:.1f}%)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_symmetry_index_overview(metrics: dict[str, float], out_path: Path) -> None:
    labels = list(metrics.keys())
    vals = [metrics[k] for k in labels]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(labels))))
    y = np.arange(len(labels))
    colors = ["#16a34a" if v < 15 else "#ca8a04" if v < 30 else "#dc2626" for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.axvline(15, color="green", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(30, color="orange", ls=":", lw=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("symmetry index (%) — lower is better")
    ax.set_title("Symmetry index overview (§5)")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_fullbody_config(args.config_name)
    env = make_env(cfg, motion_paths=[args.motion_path], use_mujoco=True)
    try:
        qpos_idx = resolve_leg_qpos(env.model)
        foot_idx = resolve_foot_site_indices(list(env.sites_for_mimic))
    finally:
        env.stop()

    data = np.load(Path(args.rollout_npz), allow_pickle=True)
    dt = float(np.asarray(data["dt"]).reshape(-1)[0]) if "dt" in data else 0.01
    contact_l = np.asarray(data["contact_left"], dtype=bool).reshape(-1)
    contact_r = np.asarray(data["contact_right"], dtype=bool).reshape(-1)
    vgrf_l = np.maximum(np.asarray(data["left_GRF"], dtype=np.float64).reshape(-1), 0.0)
    vgrf_r = np.maximum(np.asarray(data["right_GRF"], dtype=np.float64).reshape(-1), 0.0)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    site_pos = np.asarray(data["site_pos"], dtype=np.float64)
    foot_l_xy = site_pos[:, foot_idx["left"], :2]
    foot_r_xy = site_pos[:, foot_idx["right"], :2]

    step_l, stance_l = step_lengths_from_contact(foot_l_xy, contact_l, dt)
    step_r, stance_r = step_lengths_from_contact(foot_r_xy, contact_r, dt)

    plot_left_right_stride_panels(
        step_l,
        step_r,
        title="Step length (toe site forward, per stride)",
        ylabel="m",
        out_path=out_dir / "step_length_per_stride_left_right.png",
    )
    plot_left_right_stride_panels(
        stance_l,
        stance_r,
        title="Stance duration (per stride)",
        ylabel="s",
        out_path=out_dir / "stance_time_per_stride_left_right.png",
    )

    peak_l, peak_r = float(np.max(vgrf_l)), float(np.max(vgrf_r))
    imp_l = float(np.sum(vgrf_l[contact_l]) * dt)
    imp_r = float(np.sum(vgrf_r[contact_r]) * dt)
    plot_grf_symmetry_panels(peak_l, peak_r, imp_l, imp_r, out_dir / "grf_symmetry_left_right.png")

    rom_l, rom_r = {}, {}
    for joint, sides in qpos_idx.items():
        rom_l[joint] = float(np.rad2deg(np.max(qpos[:, sides["left"]]) - np.min(qpos[:, sides["left"]])))
        rom_r[joint] = float(np.rad2deg(np.max(qpos[:, sides["right"]]) - np.min(qpos[:, sides["right"]])))
    plot_rom_panels(rom_l, rom_r, out_dir / "joint_rom_left_right.png")

    si = {
        "step_length_mean": symmetry_index(float(np.mean(step_l)) if step_l else 0.0, float(np.mean(step_r)) if step_r else 0.0),
        "stance_time_mean": symmetry_index(float(np.mean(stance_l)) if stance_l else 0.0, float(np.mean(stance_r)) if stance_r else 0.0),
        "GRF_peak": symmetry_index(peak_l, peak_r),
        "GRF_impulse": symmetry_index(imp_l, imp_r),
    }
    for joint in rom_l:
        si[f"ROM_{joint}"] = symmetry_index(rom_l[joint], rom_r[joint])

    plot_symmetry_index_overview(si, out_dir / "symmetry_index_overview.png")

    summary = {
        "step_length_left_m": step_l,
        "step_length_right_m": step_r,
        "step_length_mean_left_m": float(np.mean(step_l)) if step_l else None,
        "step_length_mean_right_m": float(np.mean(step_r)) if step_r else None,
        "stance_duration_left_s": stance_l,
        "stance_duration_right_s": stance_r,
        "stance_time_total_left_s": float(np.sum(contact_l) * dt),
        "stance_time_total_right_s": float(np.sum(contact_r) * dt),
        "GRF_peak_left_N": peak_l,
        "GRF_peak_right_N": peak_r,
        "GRF_impulse_left_Ns": imp_l,
        "GRF_impulse_right_Ns": imp_r,
        "joint_ROM_deg": {"left": rom_l, "right": rom_r},
        "symmetry_index_pct": si,
    }
    with (out_dir / "symmetry_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Symmetry plots saved to {out_dir}")
    for k, v in si.items():
        print(f"  SI {k}: {v:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
