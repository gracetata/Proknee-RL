"""Kinematic joint replay from recorded rollout qpos (no muscle actuation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np

from musclemimic.evaluation.types import RolloutBuffer


@dataclass
class JointTrajectory:
    motion_path: str
    dt: float
    qpos: np.ndarray
    qvel: np.ndarray
    source_controller: str
    source_checkpoint: str
    replay_mode: str = "joint_kinematic"
    metadata: dict[str, Any] | None = None

    @property
    def n_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.n_frames * self.dt)


def joint_trajectory_from_buffer(
    buffer: RolloutBuffer,
    *,
    source_controller: str,
    source_checkpoint: str,
) -> JointTrajectory:
    qpos = np.stack([np.asarray(x, dtype=np.float64) for x in buffer.qpos], axis=0)
    qvel = np.stack([np.asarray(x, dtype=np.float64) for x in buffer.qvel], axis=0)
    return JointTrajectory(
        motion_path=str(buffer.motion_path),
        dt=float(buffer.dt),
        qpos=qpos,
        qvel=qvel,
        source_controller=source_controller,
        source_checkpoint=source_checkpoint,
        metadata={
            "steps": int(buffer.steps),
            "traj_length": int(buffer.traj_length),
            "done_reason": str(buffer.done_reason),
            "episode_return": float(buffer.episode_return),
        },
    )


def save_joint_trajectory(path: Path, traj: JointTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        motion_path=np.asarray(traj.motion_path),
        dt=np.asarray(traj.dt),
        qpos=traj.qpos,
        qvel=traj.qvel,
        source_controller=np.asarray(traj.source_controller),
        source_checkpoint=np.asarray(traj.source_checkpoint),
        replay_mode=np.asarray(traj.replay_mode),
        metadata=np.asarray(json.dumps(traj.metadata or {})),
    )


def load_joint_trajectory(path: Path) -> JointTrajectory:
    data = np.load(path, allow_pickle=False)
    meta_raw = data["metadata"]
    meta_str = meta_raw.item() if meta_raw.ndim == 0 else str(meta_raw.reshape(-1)[0])
    metadata = json.loads(meta_str) if meta_str else {}
    return JointTrajectory(
        motion_path=str(data["motion_path"].item() if data["motion_path"].ndim == 0 else data["motion_path"].reshape(-1)[0]),
        dt=float(np.asarray(data["dt"]).reshape(-1)[0]),
        qpos=np.asarray(data["qpos"], dtype=np.float64),
        qvel=np.asarray(data["qvel"], dtype=np.float64),
        source_controller=str(
            data["source_controller"].item()
            if data["source_controller"].ndim == 0
            else data["source_controller"].reshape(-1)[0]
        ),
        source_checkpoint=str(
            data["source_checkpoint"].item()
            if data["source_checkpoint"].ndim == 0
            else data["source_checkpoint"].reshape(-1)[0]
        ),
        replay_mode=str(
            data["replay_mode"].item() if data["replay_mode"].ndim == 0 else data["replay_mode"].reshape(-1)[0]
        ),
        metadata=metadata,
    )


def apply_kinematic_state(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray | None) -> None:
    """Set full-body joint state directly; zero muscle controls (mechanical mimic)."""
    data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    if qvel is not None:
        data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def max_pose_drift(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> float:
    apply_kinematic_state(model, data, qpos, None)
    return float(np.max(np.abs(data.qpos - qpos)))


def record_kinematic_replay(
    *,
    env,
    traj: JointTrajectory,
    record_path: Path,
    width: int = 640,
    height: int = 480,
    fps: int | None = None,
    cam_distance: float = 3.5,
    cam_elevation: float = 0.0,
    cam_azimuth: float = 90.0,
) -> Path:
    """Replay saved joint trajectory frame-by-frame and write mp4."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    model = env.model
    data = env.data
    fps = int(fps if fps is not None else round(1.0 / max(traj.dt, 1e-6)))

    renderer = mujoco.Renderer(model, width=width, height=height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance
    cam.elevation = cam_elevation
    cam.azimuth = cam_azimuth

    with imageio.get_writer(record_path, fps=fps, quality=8) as writer:
        for i in range(traj.n_frames):
            apply_kinematic_state(model, data, traj.qpos[i], traj.qvel[i])
            cam.lookat[:] = np.asarray(data.qpos[:3], dtype=np.float64)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()
    return record_path


def write_replay_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
