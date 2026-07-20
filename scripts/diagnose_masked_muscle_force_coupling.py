#!/usr/bin/env python
"""Compare per-muscle forces: full open-loop replay vs masked-actuator replay (same ctrl)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.distill.config import repo_root
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.muscle_replay import (
    actuator_ids_for_muscle_names,
    load_muscle_trajectory,
    step_env_with_recorded_ctrl,
)
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint, setup_headless


@dataclass
class ForceTrace:
    label: str
    actuator_force: np.ndarray  # [T, nu]
    actuator_act: np.ndarray
    actuator_ctrl: np.ndarray
    root_z: np.ndarray
    root_xy: np.ndarray
    pelvis_yaw: np.ndarray
    knee_l_q: np.ndarray
    knee_r_q: np.ndarray
    knee_l_tau: np.ndarray
    knee_r_tau: np.ndarray
    episode_return: float
    done_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", default="KIT/3/walk_6m_straight_line04_poses")
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument(
        "--checkpoint_path",
        default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2",
    )
    p.add_argument(
        "--muscle-trajectory",
        default=None,
        help="Default: outputs/replay/muscle_mimic/<controller>/<motion>/muscle_trajectory.npz",
    )
    p.add_argument(
        "--mask-preset",
        default="foot1",
        choices=sorted(MUSCLE_MASK_PRESETS),
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Default: outputs/_nonformal_runs/muscle_force_coupling/<controller>/<motion>/<preset>/",
    )
    p.add_argument("--max-steps", type=int, default=0, help="0 = full trajectory")
    p.add_argument(
        "--export-csv-from-npz",
        default=None,
        help="Only export per_muscle_force_diff.csv from an existing muscle_force_traces.npz",
    )
    return p.parse_args()


def build_env(config, motion_path: str):
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBody"
    env_params["headless"] = True
    env_params.pop("prosthesis", None)
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


def joint_qpos_index(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(joint_name)
    return int(model.jnt_qposadr[jid])


def joint_qvel_index(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(joint_name)
    return int(model.jnt_dofadr[jid])


def rollout_force_trace(
    env,
    traj,
    *,
    label: str,
    disabled_actuators: np.ndarray,
    n_steps: int,
) -> ForceTrace:
    nu = int(env.model.nu)
    knee_l_qi = joint_qpos_index(env.model, "knee_angle_l")
    knee_r_qi = joint_qpos_index(env.model, "knee_angle_r")
    knee_l_vi = joint_qvel_index(env.model, "knee_angle_l")
    knee_r_vi = joint_qvel_index(env.model, "knee_angle_r")

    env.reset()
    env.data.qpos[:] = np.asarray(traj.qpos[0], dtype=np.float64)
    env.data.qvel[:] = np.asarray(traj.qvel[0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    forces = np.zeros((n_steps, nu), dtype=np.float64)
    acts = np.zeros((n_steps, nu), dtype=np.float64)
    ctrls = np.zeros((n_steps, nu), dtype=np.float64)
    root_z = np.zeros(n_steps, dtype=np.float64)
    root_xy = np.zeros((n_steps, 2), dtype=np.float64)
    pelvis_yaw = np.zeros(n_steps, dtype=np.float64)
    knee_l_q = np.zeros(n_steps, dtype=np.float64)
    knee_r_q = np.zeros(n_steps, dtype=np.float64)
    knee_l_tau = np.zeros(n_steps, dtype=np.float64)
    knee_r_tau = np.zeros(n_steps, dtype=np.float64)

    episode_return = 0.0
    done_count = 0
    disabled = np.asarray(disabled_actuators, dtype=np.int32)

    for step in range(n_steps):
        ctrl = np.asarray(traj.actuator_ctrl[step], dtype=np.float64)
        _obs, reward, done = step_env_with_recorded_ctrl(env, ctrl, disabled)
        episode_return += float(reward)
        done_count += int(bool(done))

        forces[step] = np.asarray(env.data.actuator_force, dtype=np.float64)
        acts[step] = np.asarray(env.data.act, dtype=np.float64)[:nu]
        ctrls[step] = np.asarray(env.data.ctrl, dtype=np.float64)
        root_z[step] = float(env.data.qpos[2])
        root_xy[step] = np.asarray(env.data.qpos[:2], dtype=np.float64)
        pelvis_yaw[step] = float(env.data.qpos[3])
        knee_l_q[step] = float(env.data.qpos[knee_l_qi])
        knee_r_q[step] = float(env.data.qpos[knee_r_qi])
        knee_l_tau[step] = float(env.data.qfrc_actuator[knee_l_vi])
        knee_r_tau[step] = float(env.data.qfrc_actuator[knee_r_vi])

    return ForceTrace(
        label=label,
        actuator_force=forces,
        actuator_act=acts,
        actuator_ctrl=ctrls,
        root_z=root_z,
        root_xy=root_xy,
        pelvis_yaw=pelvis_yaw,
        knee_l_q=knee_l_q,
        knee_r_q=knee_r_q,
        knee_l_tau=knee_l_tau,
        knee_r_tau=knee_r_tau,
        episode_return=float(episode_return),
        done_count=int(done_count),
    )


def first_index_where(arr: np.ndarray, cond) -> int | None:
    idx = np.flatnonzero(cond(arr))
    return int(idx[0]) if idx.size else None


def analyze_traces(
    baseline: ForceTrace,
    masked: ForceTrace,
    names: list[str],
    disabled_names: tuple[str, ...],
    disabled_ids: np.ndarray,
    dt: float,
) -> dict:
    diff = masked.actuator_force - baseline.actuator_force
    abs_diff = np.abs(diff)
    ctrl_diff = masked.actuator_ctrl - baseline.actuator_ctrl
    act_diff = masked.actuator_act - baseline.actuator_act

    disabled_set = set(int(x) for x in disabled_ids)
    disabled_name_set = set(disabled_names)
    left_ids = {i for i, n in enumerate(names) if n.endswith("_l")}
    right_ids = {i for i, n in enumerate(names) if n.endswith("_r")}

    per_muscle = []
    for i, name in enumerate(names):
        if i in disabled_set:
            region = "disabled"
        elif name.endswith("_l"):
            region = "left_other"
        elif name.endswith("_r"):
            region = "right"
        else:
            region = "torso"
        per_muscle.append(
            {
                "actuator_id": int(i),
                "name": name,
                "region": region,
                "baseline_peak_force_n": float(np.max(np.abs(baseline.actuator_force[:, i]))),
                "masked_peak_force_n": float(np.max(np.abs(masked.actuator_force[:, i]))),
                "max_abs_force_diff_n": float(np.max(abs_diff[:, i])),
                "mean_abs_force_diff_n": float(np.mean(abs_diff[:, i])),
                "rms_force_diff_n": float(np.sqrt(np.mean(diff[:, i] ** 2))),
                "max_abs_ctrl_diff": float(np.max(np.abs(ctrl_diff[:, i]))),
                "max_abs_act_diff": float(np.max(np.abs(act_diff[:, i]))),
                "first_large_force_diff_step": first_index_where(abs_diff[:, i], lambda x: x > 50.0),
            }
        )

    per_muscle.sort(key=lambda row: row["max_abs_force_diff_n"], reverse=True)

    fall_baseline = first_index_where(baseline.root_z, lambda z: z < 0.5)
    fall_masked = first_index_where(masked.root_z, lambda z: z < 0.5)
    diverge_root = first_index_where(np.abs(masked.root_z - baseline.root_z), lambda d: d > 0.05)
    diverge_knee_l = first_index_where(np.abs(masked.knee_l_q - baseline.knee_l_q), lambda d: d > 0.05)

    def region_stats(region: str) -> dict:
        ids = [row["actuator_id"] for row in per_muscle if row["region"] == region]
        if not ids:
            return {}
        sub = abs_diff[:, ids]
        return {
            "count": len(ids),
            "max_abs_force_diff_n": float(np.max(sub)),
            "mean_abs_force_diff_n": float(np.mean(sub)),
            "rms_force_diff_n": float(np.sqrt(np.mean(sub**2))),
        }

    enabled_ids = [i for i in range(len(names)) if i not in disabled_set]
    enabled_abs = abs_diff[:, enabled_ids]

    return {
        "baseline_return": baseline.episode_return,
        "masked_return": masked.episode_return,
        "baseline_done_count": baseline.done_count,
        "masked_done_count": masked.done_count,
        "disabled_muscle_names": list(disabled_names),
        "disabled_actuator_ids": [int(x) for x in disabled_ids],
        "note_ctrl_same_for_enabled": bool(np.max(np.abs(ctrl_diff[:, enabled_ids])) < 1e-6) if enabled_ids else True,
        "first_root_fall_step": {"baseline": fall_baseline, "masked": fall_masked},
        "first_root_z_diverge_step": diverge_root,
        "first_left_knee_q_diverge_step": diverge_knee_l,
        "global_enabled_muscle_force_diff": {
            "max_abs_n": float(np.max(enabled_abs)) if enabled_ids else 0.0,
            "mean_abs_n": float(np.mean(enabled_abs)) if enabled_ids else 0.0,
            "rms_n": float(np.sqrt(np.mean(enabled_abs**2))) if enabled_ids else 0.0,
        },
        "region_force_diff": {
            "disabled": region_stats("disabled"),
            "left_other": region_stats("left_other"),
            "right": region_stats("right"),
            "torso": region_stats("torso"),
        },
        "joint_summary": {
            "knee_l_tau_max_abs_diff": float(np.max(np.abs(masked.knee_l_tau - baseline.knee_l_tau))),
            "knee_r_tau_max_abs_diff": float(np.max(np.abs(masked.knee_r_tau - baseline.knee_r_tau))),
            "pelvis_yaw_max_abs_diff_rad": float(np.max(np.abs(masked.pelvis_yaw - baseline.pelvis_yaw))),
            "root_xy_max_drift_m": float(np.max(np.linalg.norm(masked.root_xy - baseline.root_xy, axis=1))),
        },
        "all_muscles": per_muscle,
        "top_force_diff_muscles": per_muscle[:25],
        "disabled_muscle_force_check": [
            {
                "name": names[i],
                "baseline_peak_n": float(np.max(np.abs(baseline.actuator_force[:, i]))),
                "masked_peak_n": float(np.max(np.abs(masked.actuator_force[:, i]))),
            }
            for i in sorted(disabled_set)
        ],
        "interpretation": (
            "ctrl is identical for enabled muscles in both rollouts; any force difference on enabled muscles "
            "comes from changed body state (qpos/qvel) altering FLV muscle length/velocity, not from changed ctrl. "
            "Disabled muscles lose active force; remaining muscles still receive same ctrl but produce different "
            "forces once whole-body kinematics diverge."
        ),
        "dt": float(dt),
    }


def write_per_muscle_force_diff_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "rank_by_max_abs_force_diff",
        "actuator_id",
        "name",
        "region",
        "baseline_peak_force_n",
        "masked_peak_force_n",
        "peak_force_change_n",
        "max_abs_force_diff_n",
        "mean_abs_force_diff_n",
        "rms_force_diff_n",
        "max_abs_ctrl_diff",
        "max_abs_act_diff",
        "first_large_force_diff_step",
        "first_large_force_diff_time_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            step = row.get("first_large_force_diff_step")
            writer.writerow(
                {
                    "rank_by_max_abs_force_diff": rank,
                    "actuator_id": row["actuator_id"],
                    "name": row["name"],
                    "region": row["region"],
                    "baseline_peak_force_n": f"{row['baseline_peak_force_n']:.6f}",
                    "masked_peak_force_n": f"{row['masked_peak_force_n']:.6f}",
                    "peak_force_change_n": f"{row['masked_peak_force_n'] - row['baseline_peak_force_n']:.6f}",
                    "max_abs_force_diff_n": f"{row['max_abs_force_diff_n']:.6f}",
                    "mean_abs_force_diff_n": f"{row['mean_abs_force_diff_n']:.6f}",
                    "rms_force_diff_n": f"{row['rms_force_diff_n']:.6f}",
                    "max_abs_ctrl_diff": f"{row['max_abs_ctrl_diff']:.6f}",
                    "max_abs_act_diff": f"{row['max_abs_act_diff']:.6f}",
                    "first_large_force_diff_step": "" if step is None else int(step),
                    "first_large_force_diff_time_s": "" if step is None else f"{step * row.get('_dt', 0.0):.4f}",
                }
            )


def _trace_from_npz(data, prefix: str) -> ForceTrace:
    n = int(data[f"{prefix}_actuator_force"].shape[0])
    z = np.zeros(n, dtype=np.float64)
    z2 = np.zeros((n, 2), dtype=np.float64)

    def _arr(key: str) -> np.ndarray:
        full = f"{prefix}_{key}"
        return np.asarray(data[full], dtype=np.float64) if full in data.files else z.copy()

    return ForceTrace(
        label=prefix,
        actuator_force=np.asarray(data[f"{prefix}_actuator_force"], dtype=np.float64),
        actuator_act=np.asarray(data[f"{prefix}_act"], dtype=np.float64),
        actuator_ctrl=np.asarray(data[f"{prefix}_ctrl"], dtype=np.float64),
        root_z=_arr("root_z"),
        root_xy=z2,
        pelvis_yaw=z,
        knee_l_q=_arr("knee_l_q"),
        knee_r_q=_arr("knee_r_q"),
        knee_l_tau=_arr("knee_l_tau"),
        knee_r_tau=_arr("knee_r_tau"),
        episode_return=0.0,
        done_count=0,
    )


def export_csv_from_npz(npz_path: Path, csv_path: Path | None = None) -> Path:
    """Rebuild per-muscle CSV from a saved muscle_force_traces.npz."""
    data = np.load(npz_path, allow_pickle=False)
    names = [str(x) for x in data["actuator_names"].reshape(-1)]
    disabled_ids = np.asarray(data["disabled_actuator_ids"], dtype=np.int32)
    disabled_names = tuple(str(x) for x in data["disabled_muscle_names"].reshape(-1))
    dt = float(np.asarray(data["dt"]).reshape(-1)[0])
    baseline = _trace_from_npz(data, "baseline")
    masked = _trace_from_npz(data, "masked")
    summary = analyze_traces(baseline, masked, names, disabled_names, disabled_ids, dt)
    rows = summary["all_muscles"]
    for row in rows:
        row["_dt"] = dt
    out = csv_path if csv_path is not None else npz_path.parent / "per_muscle_force_diff.csv"
    write_per_muscle_force_diff_csv(out, rows)
    return out


def plot_diagnosis(
    out_dir: Path,
    baseline: ForceTrace,
    masked: ForceTrace,
    names: list[str],
    disabled_ids: np.ndarray,
    summary: dict,
    dt: float,
) -> None:
    t = np.arange(baseline.root_z.shape[0], dtype=np.float64) * dt
    disabled_set = set(int(x) for x in disabled_ids)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, baseline.root_z, label="baseline root_z")
    axes[0].plot(t, masked.root_z, label="masked root_z")
    axes[0].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("root z (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, baseline.knee_l_q, label="baseline knee_l")
    axes[1].plot(t, masked.knee_l_q, label="masked knee_l")
    axes[1].plot(t, baseline.knee_r_q, label="baseline knee_r", alpha=0.7)
    axes[1].plot(t, masked.knee_r_q, label="masked knee_r", alpha=0.7)
    axes[1].set_ylabel("knee q (rad)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    enabled = [i for i in range(len(names)) if i not in disabled_set]
    mean_abs = np.mean(np.abs(masked.actuator_force[:, enabled] - baseline.actuator_force[:, enabled]), axis=1)
    axes[2].plot(t, mean_abs, color="tab:red")
    axes[2].set_ylabel("mean |ΔF| enabled (N)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=150)
    plt.close(fig)

    top = summary["top_force_diff_muscles"][:12]
    fig, axes = plt.subplots(len(top), 1, figsize=(10, 2.2 * len(top)), sharex=True)
    if len(top) == 1:
        axes = [axes]
    for ax, row in zip(axes, top):
        i = int(row["actuator_id"])
        ax.plot(t, baseline.actuator_force[:, i], label="baseline")
        ax.plot(t, masked.actuator_force[:, i], label="masked")
        ax.set_ylabel(f"{row['name']}\n(N)", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "top_muscle_force_diff.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.export_csv_from_npz:
        out = export_csv_from_npz(Path(args.export_csv_from_npz))
        print(f"Saved: {out}")
        return 0

    os.environ.setdefault("MUJOCO_GL", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    safe = safe_motion_name(args.motion_path)
    traj_path = (
        Path(args.muscle_trajectory)
        if args.muscle_trajectory
        else repo_root()
        / "outputs"
        / "replay"
        / "muscle_mimic"
        / args.controller_type
        / safe
        / "muscle_trajectory.npz"
    )
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root()
        / "outputs"
        / "_nonformal_runs"
        / "muscle_force_coupling"
        / args.controller_type
        / safe
        / args.mask_preset
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = load_muscle_trajectory(traj_path)
    n_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(traj.n_frames)

    config, _, _ = load_checkpoint(args.checkpoint_path)
    apply_temporal_params(config)
    env = build_env(config, args.motion_path)
    try:
        names = actuator_names(env.model)
        disabled_names = MUSCLE_MASK_PRESETS[args.mask_preset]
        disabled_ids = actuator_ids_for_muscle_names(env.model, disabled_names)

        print(f"Trajectory: {traj_path} ({n_steps} steps)")
        print(f"Mask preset: {args.mask_preset} -> {list(disabled_names)}")
        print("Rolling baseline (full muscles)...")
        baseline = rollout_force_trace(env, traj, label="baseline", disabled_actuators=np.asarray([], dtype=np.int32), n_steps=n_steps)
        print(f"  return={baseline.episode_return:.2f} done={baseline.done_count}")

        env.reset()
        env.data.qpos[:] = np.asarray(traj.qpos[0], dtype=np.float64)
        env.data.qvel[:] = np.asarray(traj.qvel[0], dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)
        print("Rolling masked replay...")
        masked = rollout_force_trace(env, traj, label="masked", disabled_actuators=disabled_ids, n_steps=n_steps)
        print(f"  return={masked.episode_return:.2f} done={masked.done_count}")

        summary = analyze_traces(baseline, masked, names, disabled_names, disabled_ids, float(traj.dt))

        np.savez_compressed(
            out_dir / "muscle_force_traces.npz",
            actuator_names=np.asarray(names),
            disabled_actuator_ids=disabled_ids,
            disabled_muscle_names=np.asarray(list(disabled_names)),
            baseline_actuator_force=baseline.actuator_force,
            masked_actuator_force=masked.actuator_force,
            baseline_act=baseline.actuator_act,
            masked_act=masked.actuator_act,
            baseline_ctrl=baseline.actuator_ctrl,
            masked_ctrl=masked.actuator_ctrl,
            baseline_root_z=baseline.root_z,
            masked_root_z=masked.root_z,
            baseline_knee_l_q=baseline.knee_l_q,
            masked_knee_l_q=masked.knee_l_q,
            baseline_knee_r_q=baseline.knee_r_q,
            masked_knee_r_q=masked.knee_r_q,
            baseline_knee_l_tau=baseline.knee_l_tau,
            masked_knee_l_tau=masked.knee_l_tau,
            dt=np.asarray(traj.dt),
        )
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        rows = summary["all_muscles"]
        for row in rows:
            row["_dt"] = float(traj.dt)
        csv_path = out_dir / "per_muscle_force_diff.csv"
        write_per_muscle_force_diff_csv(csv_path, rows)

        plot_diagnosis(out_dir, baseline, masked, names, disabled_ids, summary, float(traj.dt))

        print(f"Saved: {out_dir / 'muscle_force_traces.npz'}")
        print(f"Saved: {out_dir / 'summary.json'}")
        print(f"Saved: {csv_path}")
        print(
            "Enabled-muscle force diff: "
            f"max={summary['global_enabled_muscle_force_diff']['max_abs_n']:.1f} N, "
            f"mean={summary['global_enabled_muscle_force_diff']['mean_abs_n']:.1f} N"
        )
        print(f"First root diverge step: {summary['first_root_z_diverge_step']}")
        print("Top changed muscles:")
        for row in summary["top_force_diff_muscles"][:8]:
            print(
                f"  {row['name']}: max|ΔF|={row['max_abs_force_diff_n']:.1f} N, "
                f"ctrlΔ={row['max_abs_ctrl_diff']:.3g}, region={row['region']}"
            )
    finally:
        env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
