#!/usr/bin/env python
"""Diagnostic validation for prosthesis-muscle coupling.

This script establishes upper-bound and failure-semantics runs without changing
the existing Stage1/coupled-action training routes.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from musclemimic.proknee import MuscleProKneeHybridEnv


GMR_CACHE_ROOT = Path("/home/user/Workspace/musclemimic/data/caches/AMASS/MyoFullBody/gmr")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate prosthesis-muscle coupling upper bounds")
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["oracle_qpos_torque", "oracle_torque_only", "oracle_qpos_only"],
        choices=["oracle_qpos_torque", "oracle_torque_only", "oracle_qpos_only"],
    )
    parser.add_argument("--output-dir", default="/home/user/Workspace/musclemimic/refine-logs/validation_runs")
    parser.add_argument("--joint-kp", default="120,45,35,12")
    parser.add_argument("--joint-kd", default="35,38,34,18")
    parser.add_argument("--pd-torque-limit-list", default="200,150,100,60")
    parser.add_argument("--pd-torque-slew-limit-list", default=None)
    parser.add_argument("--oracle-torque-ff-limit-list", default=None)
    parser.add_argument("--prosthesis-muscle-scale", type=float, default=0.0)
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument("--hard-root-height-min", type=float, default=0.65)
    parser.add_argument("--hard-root-up-min", type=float, default=0.5)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def parse_vec(text: str | None, n: int) -> tuple[float, ...] | None:
    if text is None or str(text).lower() == "none":
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(vals)}")
    return tuple(vals)


def normalize_motion_path(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path_text.removesuffix(".npz")
    try:
        return str(path.resolve().relative_to(GMR_CACHE_ROOT)).removesuffix(".npz")
    except ValueError:
        return path_text.removesuffix(".npz")


def root_up_z(qpos: np.ndarray) -> float:
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-8:
        quat = quat / norm
    _qw, qx, qy, _qz = quat
    return float(1.0 - 2.0 * (qx * qx + qy * qy))


def left_toe_z(env: MuscleProKneeHybridEnv) -> float:
    for name, sid in env.audit.sites:
        if "toe" in name.lower() or "toes" in name.lower():
            return float(env.env.data.site_xpos[int(sid), 2])
    return float("nan")


def make_env(args) -> MuscleProKneeHybridEnv:
    return MuscleProKneeHybridEnv(
        args.checkpoint,
        dataset_group=None,
        rel_dataset_path=[normalize_motion_path(args.motion_path)],
        deterministic_oracle=True,
        apply_teacher_action=True,
        target_mode="qpos",
        pd_override=True,
        pd_kp=parse_vec(args.joint_kp, 4),
        pd_kd=parse_vec(args.joint_kd, 4),
        pd_torque_limit=parse_vec(args.pd_torque_limit_list, 4),
        pd_torque_slew_limit=parse_vec(args.pd_torque_slew_limit_list, 4),
        oracle_torque_ff_scale=1.0,
        oracle_torque_ff_limit=parse_vec(args.oracle_torque_ff_limit_list, 4),
        torque_feedforward_target_mode="pd_residual",
        prosthesis_muscle_scale=args.prosthesis_muscle_scale,
    )


def highlight_left_leg(env: MuscleProKneeHybridEnv) -> None:
    body_ids = {int(env.env.model.jnt_bodyid[j.joint_id]) for j in env.audit.joints}
    for gid in range(env.env.model.ngeom):
        if int(env.env.model.geom_bodyid[gid]) in body_ids:
            alpha = float(env.env.model.geom_rgba[gid, 3])
            env.env.model.geom_rgba[gid] = np.asarray([0.95, 0.08, 0.08, alpha], dtype=env.env.model.geom_rgba.dtype)


def run_mode(args, mode: str) -> dict:
    env = make_env(args)
    if mode == "oracle_torque_only":
        env.pd_kp[:] = 0.0
        env.pd_kd[:] = 0.0
    data = env.reset()
    os.makedirs(args.output_dir, exist_ok=True)
    safe_motion = normalize_motion_path(args.motion_path).replace("/", "_")
    csv_path = os.path.join(args.output_dir, f"{safe_motion}_{mode}.csv")
    video_path = os.path.join(args.output_dir, f"{safe_motion}_{mode}.mp4")

    renderer = None
    writer = None
    if args.record_video:
        highlight_left_leg(env)
        renderer = mujoco.Renderer(env.env.model, width=args.width, height=args.height)
        writer = imageio.get_writer(video_path, fps=args.fps, quality=8)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 6.0
        cam.elevation = -20.0
        cam.azimuth = 90.0
    else:
        cam = None

    rows = []
    done_step = None
    hard_fail_step = None
    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "step",
                    "mode",
                    "done",
                    "hard_fail",
                    "root_height",
                    "root_up_z",
                    "ncon",
                    "toe_z",
                    "tracking_mae",
                    "torque_mae",
                    "knee_q",
                    "ankle_q",
                    "subtalar_q",
                    "mtp_q",
                    "knee_target",
                    "ankle_target",
                    "subtalar_target",
                    "mtp_target",
                    "applied_knee",
                    "applied_ankle",
                    "applied_subtalar",
                    "applied_mtp",
                    "oracle_knee",
                    "oracle_ankle",
                    "oracle_subtalar",
                    "oracle_mtp",
                ]
            )
            for step in range(1, args.steps + 1):
                oracle_action = env.oracle.act(env._oracle_obs)
                qpos_oracle_after, _qvel_oracle_after, oracle_torque = env._oracle_step_prosthesis_labels(
                    oracle_action
                )
                q_current = np.asarray(env.env.data.qpos[env.audit.qpos_indices], dtype=np.float32)
                if mode == "oracle_qpos_torque":
                    target = qpos_oracle_after[env.audit.qpos_indices].astype(np.float32)
                    ff = env._oracle_equivalent_feedforward_target(oracle_torque, target)
                elif mode == "oracle_torque_only":
                    target = q_current.copy()
                    ff = oracle_torque.astype(np.float32)
                elif mode == "oracle_qpos_only":
                    target = qpos_oracle_after[env.audit.qpos_indices].astype(np.float32)
                    ff = np.zeros(env.action_dim, dtype=np.float32)
                else:
                    raise ValueError(mode)

                raw_obs, _reward, _absorbing, done, info = env._step_prosthesis_replacement_pd(
                    oracle_action,
                    target,
                    ff,
                )
                env._oracle_obs = env.oracle.update_obs(raw_obs)
                q = np.asarray(env.env.data.qpos[env.audit.qpos_indices], dtype=np.float64)
                applied = np.asarray(info.get("prosthesis_applied_torque_mean", np.zeros(4)), dtype=np.float64)
                oracle = np.asarray(oracle_torque, dtype=np.float64)
                root_h = float(env.env.data.qpos[2])
                root_up = root_up_z(np.asarray(env.env.data.qpos, dtype=np.float64))
                hard_fail = root_h < args.hard_root_height_min or root_up < args.hard_root_up_min
                tracking_mae = float(np.mean(np.abs(q - target)))
                torque_mae = float(np.mean(np.abs(applied - oracle)))
                row = [
                    step,
                    mode,
                    int(bool(done)),
                    int(bool(hard_fail)),
                    f"{root_h:.6f}",
                    f"{root_up:.6f}",
                    int(env.env.data.ncon),
                    f"{left_toe_z(env):.6f}",
                    f"{tracking_mae:.6f}",
                    f"{torque_mae:.6f}",
                    *[f"{float(x):.6f}" for x in q],
                    *[f"{float(x):.6f}" for x in target],
                    *[f"{float(x):.6f}" for x in applied],
                    *[f"{float(x):.6f}" for x in oracle],
                ]
                w.writerow(row)
                rows.append(
                    {
                        "step": step,
                        "done": bool(done),
                        "hard_fail": bool(hard_fail),
                        "root_height": root_h,
                        "root_up_z": root_up,
                        "tracking_mae": tracking_mae,
                        "torque_mae": torque_mae,
                    }
                )
                if writer is not None and renderer is not None and cam is not None:
                    cam.lookat[:] = np.asarray(env.env.data.qpos[:3], dtype=np.float64)
                    renderer.update_scene(env.env.data, camera=cam)
                    writer.append_data(renderer.render())
                if done and done_step is None:
                    done_step = step
                if hard_fail and hard_fail_step is None:
                    hard_fail_step = step
                if (done and not args.ignore_done) or hard_fail:
                    break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.env.stop()

    root_min = min((r["root_height"] for r in rows), default=float("nan"))
    root_up_min = min((r["root_up_z"] for r in rows), default=float("nan"))
    tracking_mean = float(np.mean([r["tracking_mae"] for r in rows])) if rows else float("nan")
    torque_mean = float(np.mean([r["torque_mae"] for r in rows])) if rows else float("nan")
    return {
        "mode": mode,
        "csv": csv_path,
        "video": video_path if args.record_video else None,
        "steps": len(rows),
        "done_step": done_step,
        "hard_fail_step": hard_fail_step,
        "root_min": root_min,
        "root_up_min": root_up_min,
        "tracking_mae": tracking_mean,
        "torque_mae": torque_mean,
    }


def main() -> int:
    args = parse_args()
    summaries = [run_mode(args, mode) for mode in args.modes]
    summary_path = os.path.join(args.output_dir, "summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "csv",
                "video",
                "steps",
                "done_step",
                "hard_fail_step",
                "root_min",
                "root_up_min",
                "tracking_mae",
                "torque_mae",
            ],
        )
        w.writeheader()
        w.writerows(summaries)
    print(f"Saved summary: {summary_path}")
    for item in summaries:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

