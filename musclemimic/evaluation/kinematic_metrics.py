"""Kinematic / mimic metrics (MuscleMimic Table-2 style)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from musclemimic.evaluation.types import MetricValue, RolloutBuffer


def _quat_yaw_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in np.asarray(quat).reshape(-1)[:4]]
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_deg(delta: float) -> float:
    return abs(((delta + 180.0) % 360.0) - 180.0)


def _mean_rmse_from_steps(info_steps: list[dict], key: str, scale: float = 1.0) -> tuple[float | None, float | None]:
    vals = [float(step[key]) for step in info_steps if key in step]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64) * scale
    return float(np.mean(arr)), float(np.sqrt(np.mean(arr**2)))


def compute_kinematic_metrics(buffer: RolloutBuffer, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    traj_len = max(int(buffer.traj_length), 1)
    frame_coverage = float(buffer.steps) / float(traj_len)
    success = int(buffer.done_reason == "completed" and frame_coverage >= 0.99)

    joint_mean, joint_rmse = _mean_rmse_from_steps(buffer.info_steps, "err_joint_pos", scale=180.0 / math.pi)
    vel_mean, vel_rmse = _mean_rmse_from_steps(buffer.info_steps, "err_joint_vel", scale=180.0 / math.pi)
    root_mean, root_rmse = _mean_rmse_from_steps(buffer.info_steps, "err_root_xyz", scale=100.0)
    yaw_mean, yaw_rmse = _mean_rmse_from_steps(buffer.info_steps, "err_root_yaw", scale=180.0 / math.pi)
    rpos_mean, rpos_rmse = _mean_rmse_from_steps(buffer.info_steps, "err_rpos", scale=100.0)
    site_abs_mean, site_abs_rmse = _mean_rmse_from_steps(buffer.info_steps, "err_site_abs", scale=100.0)

    if root_mean is None and buffer.root_pos and buffer.reference_root_pos:
        diffs = [
            np.linalg.norm(np.asarray(a) - np.asarray(b))
            for a, b in zip(buffer.root_pos, buffer.reference_root_pos, strict=False)
        ]
        root_mean = float(np.mean(diffs)) * 100.0
        root_rmse = float(np.sqrt(np.mean(np.square(diffs)))) * 100.0

    if yaw_mean is None and buffer.root_quat and buffer.reference_root_pos:
        pass

    per_joint_mean = None
    if buffer.qpos and buffer.reference_qpos:
        q = np.stack([np.asarray(x) for x in buffer.qpos], axis=0)
        qr = np.stack([np.asarray(x) for x in buffer.reference_qpos], axis=0)
        n = min(q.shape[1], qr.shape[1])
        err = np.abs(q[:, :n] - qr[:, :n])
        per_joint_mean = (np.mean(err, axis=0) * (180.0 / math.pi)).tolist()

    foot_site_keys = {"left_ankle_mimic", "left_toes_mimic", "right_ankle_mimic", "right_toes_mimic", "pelvis_mimic"}
    site_names = meta.get("site_names") or []
    foot_indices = [i for i, n in enumerate(site_names) if n in foot_site_keys]

    return {
        "success": MetricValue(success, unit="0/1").to_json(),
        "frame_coverage": MetricValue(frame_coverage, unit="ratio").to_json(),
        "joint_angle_error_deg": {
            "mean": MetricValue(joint_mean, unit="deg", reason=None if joint_mean is not None else "missing err_joint_pos").to_json(),
            "rmse": MetricValue(joint_rmse, unit="deg", reason=None if joint_rmse is not None else "missing err_joint_pos").to_json(),
            "per_joint_mean": MetricValue(per_joint_mean, unit="deg").to_json(),
        },
        "joint_velocity_error_deg_s": {
            "mean": MetricValue(vel_mean, unit="deg/s").to_json(),
            "rmse": MetricValue(vel_rmse, unit="deg/s").to_json(),
        },
        "root_position_error_cm": {
            "mean": MetricValue(root_mean, unit="cm").to_json(),
            "rmse": MetricValue(root_rmse, unit="cm").to_json(),
        },
        "root_yaw_error_deg": {
            "mean": MetricValue(yaw_mean, unit="deg").to_json(),
            "rmse": MetricValue(yaw_rmse, unit="deg").to_json(),
        },
        "relative_site_position_error_cm": {
            "mean": MetricValue(rpos_mean, unit="cm").to_json(),
            "rmse": MetricValue(rpos_rmse, unit="cm").to_json(),
            "foot_subset_indices": foot_indices,
        },
        "absolute_site_position_error_cm": {
            "mean": MetricValue(site_abs_mean, unit="cm").to_json(),
            "rmse": MetricValue(site_abs_rmse, unit="cm").to_json(),
        },
        "episode_length": MetricValue(int(buffer.steps), unit="steps").to_json(),
        "episode_return": MetricValue(float(buffer.episode_return), unit="reward").to_json(),
        "done_reason": MetricValue(buffer.done_reason, unit="str").to_json(),
    }
