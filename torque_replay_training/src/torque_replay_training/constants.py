"""Shared, explicit control-boundary constants."""

from __future__ import annotations

PROSTHESIS_JOINT_NAMES = (
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)

DEFAULT_TORQUE_LIMITS = (120.0, 120.0, 35.0, 25.0)
DEFAULT_RESIDUAL_LIMITS = (20.0, 20.0, 8.0, 5.0)

