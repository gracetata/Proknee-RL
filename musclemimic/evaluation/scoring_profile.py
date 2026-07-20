"""Motion-type aware composite scoring: drop unfair dimensions/submetrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from musclemimic.evaluation.motion_taxonomy import classify_motion_type

ALL_DIMENSIONS: tuple[str, ...] = (
    "task_reliability",
    "pose_tracking",
    "global_trajectory",
    "contact_biomechanics",
    "bilateral_symmetry",
    "control_quality",
)

_KNOWN_TYPES = frozenset(
    {
        "walk",
        "run",
        "turn_left",
        "turn_right",
        "circle_walk",
        "jump",
        "transition",
        "other",
    }
)


@dataclass(frozen=True)
class ScoringProfile:
    motion_type: str
    active_dimensions: tuple[str, ...]
    excluded_dimensions: dict[str, str] = field(default_factory=dict)
    excluded_submetrics: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relaxed_submetrics: dict[str, dict[str, float]] = field(default_factory=dict)
    summary_zh: str = ""


def is_straight_locomotion(motion_type: str) -> bool:
    return motion_type in {"walk", "run"}


def resolve_scoring_profile(motion_path_or_type: str) -> ScoringProfile:
    motion_type = (
        motion_path_or_type
        if motion_path_or_type in _KNOWN_TYPES
        else classify_motion_type(motion_path_or_type)
    )

    if is_straight_locomotion(motion_type):
        return ScoringProfile(
            motion_type=motion_type,
            active_dimensions=ALL_DIMENSIONS,
            summary_zh="直行类动作（走/跑）：使用六维综合评分。",
        )

    if motion_type in {"turn_left", "turn_right", "circle_walk"}:
        return ScoringProfile(
            motion_type=motion_type,
            active_dimensions=(
                "task_reliability",
                "pose_tracking",
                "global_trajectory",
                "contact_biomechanics",
                "control_quality",
            ),
            excluded_dimensions={
                "bilateral_symmetry": "转弯/绕圈时左右承重与步态本就不对称，不评左右对称。",
            },
            excluded_submetrics={
                "global_trajectory": ("root_yaw_error_deg",),
                "contact_biomechanics": (
                    "peak_vgrf_norm_left",
                    "peak_vgrf_norm_right",
                    "mean_vgrf_stance_norm_left",
                    "mean_vgrf_stance_norm_right",
                ),
            },
            relaxed_submetrics={
                "global_trajectory": {
                    "root_position_error_cm": 80.0,
                },
            },
            summary_zh="转弯/绕圈：五维评分（去掉左右对称；去掉航向角与峰值带宽等直行假设项）。",
        )

    # jump / transition / other
    return ScoringProfile(
        motion_type=motion_type,
        active_dimensions=(
            "task_reliability",
            "pose_tracking",
            "global_trajectory",
            "control_quality",
        ),
        excluded_dimensions={
            "bilateral_symmetry": "非周期步态不评左右对称。",
            "contact_biomechanics": "跳跃/过渡动作的接触力学不适用步行峰值带宽标定。",
        },
        excluded_submetrics={
            "global_trajectory": ("root_yaw_error_deg",),
        },
        relaxed_submetrics={
            "global_trajectory": {
                "root_position_error_cm": 90.0,
            },
        },
        summary_zh="非步态周期动作：四维评分（任务、姿态、轨迹、控制）。",
    )


def uses_symmetric_gait_scoring(motion_path_or_type: str) -> bool:
    """True when bilateral_symmetry is an active composite-score dimension (6D hexagon)."""
    profile = resolve_scoring_profile(motion_path_or_type)
    return "bilateral_symmetry" in profile.active_dimensions


def eval_run_bucket(motion_path_or_type: str) -> str:
    """Subfolder under runs/<controller>/ for grouped eval artifacts."""
    if uses_symmetric_gait_scoring(motion_path_or_type):
        return "6d_symmetric_gait"
    profile = resolve_scoring_profile(motion_path_or_type)
    if len(profile.active_dimensions) == 5:
        return "5d_asymmetric_gait"
    return "4d_other"


def renormalize_weights(
    base_weights: dict[str, float], active_dimensions: tuple[str, ...]
) -> dict[str, float]:
    picked = {k: float(base_weights.get(k, 0.0)) for k in active_dimensions}
    total = sum(picked.values())
    if total <= 0:
        n = max(len(active_dimensions), 1)
        return {k: 1.0 / n for k in active_dimensions}
    return {k: v / total for k, v in picked.items()}


def apply_profile_to_config(
    config: dict[str, Any], profile: ScoringProfile
) -> dict[str, Any]:
    """Return a shallow copy of config with only active dimensions and adjusted thresholds."""
    cfg = dict(config)
    dims_in = config.get("dimensions", {})
    dims_out: dict[str, Any] = {}
    for dim_id in profile.active_dimensions:
        if dim_id not in dims_in:
            continue
        spec = dict(dims_in[dim_id])
        submetrics = []
        skip = set(profile.excluded_submetrics.get(dim_id, ()))
        relax = profile.relaxed_submetrics.get(dim_id, {})
        for sm in spec.get("submetrics", []):
            sid = str(sm["id"])
            if sid in skip:
                continue
            sm_copy = dict(sm)
            if sid in relax and sm_copy.get("type") == "lower_is_better":
                sm_copy["bad"] = float(relax[sid])
            submetrics.append(sm_copy)
        spec["submetrics"] = submetrics
        dims_out[dim_id] = spec
    cfg["dimensions"] = dims_out
    cfg["dimension_weights"] = renormalize_weights(
        config.get("dimension_weights", {}), profile.active_dimensions
    )
    return cfg
