"""Tests for composite locomotion scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musclemimic.evaluation.radar_plot import (
    _closed_ring,
    _vertices_from_scores,
)
from musclemimic.evaluation.scoring_profile import (
    apply_profile_to_config,
    resolve_scoring_profile,
)
from musclemimic.evaluation.composite_score import (
    build_composite_report,
    compute_dimension_scores,
    compute_total_score,
    load_composite_config,
    score_higher_is_better,
    score_in_band,
    score_lower_is_better,
    score_symmetry,
)


def test_score_lower_is_better_boundaries():
    assert score_lower_is_better(5.0, good=6.0, bad=15.0) == 100.0
    assert score_lower_is_better(20.0, good=6.0, bad=15.0) == 0.0
    mid = score_lower_is_better(10.5, good=6.0, bad=15.0)
    assert 0.0 < mid < 100.0


def test_score_higher_is_better_boundaries():
    assert score_higher_is_better(1.0, good=1.0, bad=0.0) == 100.0
    assert score_higher_is_better(0.0, good=1.0, bad=0.0) == 0.0


def test_score_in_band_center():
    s = score_in_band(1.0, low=0.75, high=1.15, ideal=1.0)
    assert s >= 80.0


def test_score_symmetry():
    assert score_symmetry(0.0) == 100.0
    assert score_symmetry(40.0) == 0.0
    assert score_symmetry(20.0) == pytest.approx(50.0)


def test_turn_profile_excludes_symmetry_and_renormalizes():
    profile = resolve_scoring_profile("turn_left")
    assert "bilateral_symmetry" not in profile.active_dimensions
    assert len(profile.active_dimensions) == 5
    cfg = load_composite_config()
    adapted = apply_profile_to_config(cfg, profile)
    w = adapted["dimension_weights"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert "bilateral_symmetry" not in w
    assert "root_yaw_error_deg" not in [
        sm["id"] for sm in adapted["dimensions"]["global_trajectory"]["submetrics"]
    ]


def test_jump_profile_is_four_dimensions():
    profile = resolve_scoring_profile("jump")
    assert len(profile.active_dimensions) == 4
    assert "contact_biomechanics" not in profile.active_dimensions


def test_dimension_weights_sum_to_one():
    cfg = load_composite_config()
    w = cfg["dimension_weights"]
    assert sum(float(v) for v in w.values()) == pytest.approx(1.0, abs=1e-6)


def test_radar_polygon_closes_to_first_vertex():
    import numpy as np

    from musclemimic.evaluation.radar_plot import _axis_angles

    scores = np.array([100.0, 81.0, 79.0, 73.0, 43.0, 66.0])
    angles = _axis_angles(len(scores))
    verts = _vertices_from_scores(scores, angles)
    closed = _closed_ring(verts)
    assert np.allclose(closed[-1], closed[0])
    assert np.allclose(closed[-1], verts[0])
    assert closed[-2][1] == pytest.approx(verts[-1][1], abs=0.01)
    # Closing edge must reach the top vertex (100), not an intermediate y on the axis.
    assert closed[-1][1] == pytest.approx(1.0, abs=0.02)
    assert closed[-2][0] == pytest.approx(verts[-1][0], abs=0.02)


def test_fail_cap_when_not_successful():
    cfg = load_composite_config()
    metrics = {
        "official_imitation": {
            "success": {"value": 0},
            "frame_coverage": {"value": 0.5},
            "joint_angle_error_deg": {"mean": {"value": 5.0}},
            "joint_velocity_error_deg_s": {"mean": {"value": 20.0}},
            "relative_site_position_error_cm": {"mean": {"value": 2.0}},
            "root_position_error_cm": {"mean": {"value": 20.0}},
            "root_yaw_error_deg": {"mean": {"value": 3.0}},
        },
        "force": {
            "peak_vGRF_left_filtered": {"value": 800.0},
            "peak_vGRF_left_filtered_norm": {"value": 1.0},
            "peak_vGRF_right_filtered_norm": {"value": 1.0},
            "mean_vGRF_left_filtered": {"value": 250.0},
            "mean_vGRF_right_filtered": {"value": 250.0},
            "contact_switch_count": {"value": 30},
        },
        "symmetry": {
            "GRF_symmetry_index_peak_filtered": {"value": 10.0},
            "GRF_symmetry_index_impulse_filtered": {"value": 10.0},
            "contact_duration_symmetry_index": {"value": 5.0},
        },
    }
    dims = compute_dimension_scores(cfg, metrics)
    total = compute_total_score(dims, cfg, metrics)
    assert total <= cfg["fail_cap_total"]


@pytest.mark.skipif(
    not Path(
        "outputs/eval/KIT_3_walk_6m_straight_line04_poses/metrics.json"
    ).is_file(),
    reason="KIT line04 metrics not present",
)
def test_build_report_on_kit_line04():
    root = Path(__file__).resolve().parents[1]
    metrics = root / "outputs/eval/KIT_3_walk_6m_straight_line04_poses/metrics.json"
    sym = (
        root
        / "outputs/eval/KIT_3_walk_6m_straight_line04_poses/analysis/section5_symmetry/symmetry_metrics_summary.json"
    )
    torq = (
        root
        / "outputs/eval/KIT_3_walk_6m_straight_line04_poses/analysis/section4_torque/torque_metrics_summary.json"
    )
    report = build_composite_report(
        metrics,
        symmetry_summary_path=sym if sym.is_file() else None,
        torque_summary_path=torq if torq.is_file() else None,
    )
    assert 0.0 <= report["total_score"] <= 100.0
    assert len(report["dimensions"]) == 6
