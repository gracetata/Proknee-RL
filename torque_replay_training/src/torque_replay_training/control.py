"""Pure control composition helpers used by replay and tests."""

from __future__ import annotations

import numpy as np


def compose_prosthesis_torque(
    baseline: np.ndarray,
    normalized_action: np.ndarray,
    residual_limits: np.ndarray,
    torque_limits: np.ndarray,
    *,
    residual_scale: float = 1.0,
    exact_baseline: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical prosthesis torque and residual torque.

    ``exact_baseline`` is reserved for equivalence tests: zero residual must pass
    through without clipping or smoothing the recorded healthy torque.
    """

    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    action = np.asarray(normalized_action, dtype=np.float64).reshape(-1)
    residual_limits = np.asarray(residual_limits, dtype=np.float64).reshape(-1)
    torque_limits = np.asarray(torque_limits, dtype=np.float64).reshape(-1)
    if not (baseline.size == action.size == residual_limits.size == torque_limits.size):
        raise ValueError("baseline, action, residual_limits, and torque_limits must have equal length")
    residual = np.clip(action, -1.0, 1.0) * residual_limits * float(residual_scale)
    command = baseline + residual
    if not exact_baseline:
        command = np.clip(command, -torque_limits, torque_limits)
    return command, residual


def root_up_z(qpos: np.ndarray) -> float:
    """World-z component of the pelvis local z axis for a [w,x,y,z] quaternion."""

    quat = np.asarray(qpos, dtype=np.float64).reshape(-1)[3:7]
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return -1.0
    _qw, qx, qy, _qz = quat / norm
    return float(1.0 - 2.0 * (qx * qx + qy * qy))

