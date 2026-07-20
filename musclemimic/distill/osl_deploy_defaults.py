"""Calibrated OSL FSM deployment defaults.

Validated on KIT/425 joint replay (`joint_replay_osl_fsm_knee15_loadrange_footoff`):
- mask_preset=knee15
- loadcell clipped to [-body_weight_n, 0]
- foot subtalar/mtp hold PD disabled
"""

from __future__ import annotations

import numpy as np

from musclemimic.prosthesis.constants import DEFAULT_OSL_MASK_PRESET

DEFAULT_OSL_KNEE_TORQUE_LIMIT = 140.0
DEFAULT_OSL_ANKLE_TORQUE_LIMIT = 120.0
DEFAULT_OSL_FOOT_TORQUE_LIMIT = 0.0
DEFAULT_OSL_TORQUE_SLEW_LIMIT = 35.0
DEFAULT_OSL_LOADCELL_SCALE = 1.0
DEFAULT_OSL_CLIP_LOADCELL_TO_BODY_WEIGHT = True
DEFAULT_OSL_PROSTHESIS_MUSCLE_SCALE = 0.0

__all__ = [
    "DEFAULT_OSL_ANKLE_TORQUE_LIMIT",
    "DEFAULT_OSL_CLIP_LOADCELL_TO_BODY_WEIGHT",
    "DEFAULT_OSL_FOOT_TORQUE_LIMIT",
    "DEFAULT_OSL_KNEE_TORQUE_LIMIT",
    "DEFAULT_OSL_LOADCELL_SCALE",
    "DEFAULT_OSL_MASK_PRESET",
    "DEFAULT_OSL_PROSTHESIS_MUSCLE_SCALE",
    "DEFAULT_OSL_TORQUE_SLEW_LIMIT",
    "clip_loadcell_fz_n",
    "loadcell_range_for_body_weight",
]


def loadcell_range_for_body_weight(body_weight_n: float) -> tuple[float, float]:
    bw = float(body_weight_n)
    return (-bw, 0.0)


def clip_loadcell_fz_n(loadcell_fz_n: float, body_weight_n: float) -> float:
    lo, hi = loadcell_range_for_body_weight(body_weight_n)
    return float(np.clip(loadcell_fz_n, lo, hi))


def resolve_loadcell_clip_kwargs(
    loadcell_range: tuple[float, float] | None,
    *,
    clip_loadcell_to_body_weight: bool = DEFAULT_OSL_CLIP_LOADCELL_TO_BODY_WEIGHT,
) -> dict[str, float | bool | None]:
    if loadcell_range is not None:
        lo, hi = loadcell_range
        return {
            "clip_loadcell_to_body_weight": True,
            "loadcell_clip_min_n": float(lo),
            "loadcell_clip_max_n": float(hi),
        }
    return {"clip_loadcell_to_body_weight": bool(clip_loadcell_to_body_weight)}
