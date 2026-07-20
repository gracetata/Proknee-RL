"""Observation builders and phase labels for MuscleMimic-ProKnee."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .constants import LeftLegAudit


PHASE_NAMES = ("static", "start_stop", "steady", "high_dynamic")


@dataclass(frozen=True)
class ObservationSpec:
    obs_dim: int
    priv_dim: int
    proprio_dim: int
    history_len: int


class ProprioHistory:
    def __init__(self, history_len: int, dim: int):
        self.history_len = int(history_len)
        self.dim = int(dim)
        self._buf = deque(maxlen=self.history_len)
        self.reset()

    def reset(self):
        self._buf.clear()
        for _ in range(self.history_len):
            self._buf.append(np.zeros(self.dim, dtype=np.float32))

    def append(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.dim:
            raise ValueError(f"Expected proprio dim {self.dim}, got {obs.shape[0]}")
        self._buf.append(obs)
        return self.as_array()

    def as_array(self) -> np.ndarray:
        return np.stack(list(self._buf), axis=0).astype(np.float32)


def root_xy_speed(data) -> float:
    return float(np.linalg.norm(np.asarray(data.qvel[:2], dtype=np.float64)))


def phase_features(data, previous_speed: float | None = None) -> np.ndarray:
    speed = root_xy_speed(data)
    accel = 0.0 if previous_speed is None else speed - float(previous_speed)
    static = float(speed < 0.15)
    high_dynamic = float(speed > 1.5)
    start_stop = float(abs(accel) > 0.08)
    steady = float((not static) and (not start_stop))
    return np.asarray([static, start_stop, steady, high_dynamic, speed, accel], dtype=np.float32)


def build_global_observation(
    env,
    audit: LeftLegAudit,
    previous_speed: float | None = None,
    *,
    obs_mode: str = "easy",
) -> np.ndarray:
    """Compact global observation for the Stage-1/2 policy.

    This is intentionally lower-dimensional than the raw MuscleMimic observation
    while still keeping global context: root state, selected joint state, selected
    mimic sites, oracle ctrl summary, and phase/speed features.
    """

    data = env.data
    qpos = np.asarray(data.qpos, dtype=np.float32)
    qvel = np.asarray(data.qvel, dtype=np.float32)

    root = np.concatenate([qpos[:7], qvel[:6]]).astype(np.float32)

    left_qpos = qpos[audit.qpos_indices]
    left_qvel = qvel[audit.qvel_indices]

    # A compact full-body state: root + all hinge/slide qpos/qvel excluding freejoint quaternion details.
    compact_body = np.concatenate([qpos[7:].astype(np.float32), qvel[6:].astype(np.float32)])

    site_xyz = []
    root_xyz = np.asarray(data.qpos[:3], dtype=np.float32)
    for _name, sid in audit.sites:
        site_xyz.append(np.asarray(data.site_xpos[sid], dtype=np.float32) - root_xyz)
    site_xyz = np.concatenate(site_xyz) if site_xyz else np.zeros(0, dtype=np.float32)

    parts = [
        root,
        compact_body,
        left_qpos,
        left_qvel,
        site_xyz.astype(np.float32),
    ]
    if obs_mode != "pose_only":
        ctrl = np.asarray(data.ctrl, dtype=np.float32)
        ctrl_summary = np.asarray([ctrl.mean(), ctrl.std(), ctrl.min(), ctrl.max()], dtype=np.float32)
        parts.append(ctrl_summary)
    parts.append(phase_features(data, previous_speed))
    return np.concatenate(parts).astype(np.float32)


def build_priv_info(
    env,
    audit: LeftLegAudit,
    oracle_target: np.ndarray,
    previous_speed: float | None = None,
    *,
    obs_mode: str = "easy",
) -> np.ndarray:
    del oracle_target
    data = env.data
    qpos = np.asarray(data.qpos, dtype=np.float32)
    qvel = np.asarray(data.qvel, dtype=np.float32)
    contact_count = np.asarray([float(data.ncon)], dtype=np.float32)
    parts = [qpos, qvel]
    if obs_mode != "pose_only":
        ctrl = np.asarray(data.ctrl, dtype=np.float32)
        parts.append(ctrl)
    parts.extend([audit.qpos_indices.astype(np.float32), contact_count, phase_features(data, previous_speed)])
    return np.concatenate(parts).astype(np.float32)


def build_observation_spec(
    env,
    audit: LeftLegAudit,
    history_len: int = 30,
    *,
    obs_mode: str = "easy",
) -> ObservationSpec:
    dummy_target = np.zeros(len(audit.joints), dtype=np.float32)
    obs = build_global_observation(env, audit, obs_mode=obs_mode)
    priv = build_priv_info(env, audit, dummy_target, obs_mode=obs_mode)
    return ObservationSpec(
        obs_dim=int(obs.shape[0]),
        priv_dim=int(priv.shape[0]),
        proprio_dim=int(obs.shape[0]),
        history_len=int(history_len),
    )
