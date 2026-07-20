"""Reference-motion phase labelling for MuscleMimic-ProKnee."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PhaseStats:
    path: str
    n_frames: int
    static_frac: float
    dynamic_frac: float
    start_stop_frac: float
    high_dynamic_frac: float


def load_qpos(path: str | Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "qpos" not in data:
        raise KeyError(f"{path} does not contain qpos")
    return np.asarray(data["qpos"], dtype=np.float32)


def root_speed_from_qpos(qpos: np.ndarray, dt: float = 0.01) -> np.ndarray:
    if qpos.shape[0] < 2:
        return np.zeros(qpos.shape[0], dtype=np.float32)
    xy = qpos[:, :2]
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / float(dt)
    return np.concatenate([[speed[0]], speed]).astype(np.float32)


def phase_labels_from_speed(
    speed: np.ndarray,
    static_threshold: float = 0.15,
    high_dynamic_threshold: float = 1.5,
    accel_threshold: float = 0.08,
) -> np.ndarray:
    accel = np.concatenate([[0.0], np.diff(speed)]).astype(np.float32)
    labels = np.full(speed.shape, "steady", dtype=object)
    labels[speed < static_threshold] = "static"
    labels[np.abs(accel) > accel_threshold] = "start_stop"
    labels[speed > high_dynamic_threshold] = "high_dynamic"
    return labels


def summarize_reference(path: str | Path, dt: float = 0.01) -> PhaseStats:
    qpos = load_qpos(path)
    speed = root_speed_from_qpos(qpos, dt=dt)
    labels = phase_labels_from_speed(speed)
    n = max(1, len(labels))
    return PhaseStats(
        path=str(path),
        n_frames=int(qpos.shape[0]),
        static_frac=float(np.mean(labels == "static")),
        dynamic_frac=float(np.mean(labels != "static")),
        start_stop_frac=float(np.mean(labels == "start_stop")),
        high_dynamic_frac=float(np.mean(labels == "high_dynamic")),
    )


def balanced_phase_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency frame weights for phase-aware sampling."""

    labels = np.asarray(labels)
    weights = np.zeros(labels.shape, dtype=np.float32)
    for label in np.unique(labels):
        mask = labels == label
        weights[mask] = 1.0 / max(float(mask.sum()), 1.0)
    total = float(weights.sum())
    return weights / total if total > 0 else np.ones_like(weights) / max(len(weights), 1)
