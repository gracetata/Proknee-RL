"""Tests for FSM prosthesis torque summaries."""

from __future__ import annotations

import numpy as np

from musclemimic.evaluation.composite_score import (
    _torque_summary_derived,
    load_composite_config,
    score_lower_is_better,
)
from musclemimic.evaluation.light_summaries import (
    TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID,
    _joint_torque_summary,
)


def test_joint_torque_summary_on_varying_series():
    dt = 0.01
    tau = np.sin(np.linspace(0, 8 * np.pi, 200)) * 50.0
    qvel = np.zeros_like(tau)
    out = _joint_torque_summary(tau, qvel, dt)
    assert out["peak_torque_Nm"] > 40.0
    assert out["torque_smoothness_mean_Nm_per_step"] > 0.1
    assert out["torque_jerk_mean_Nm"] > 0.0


def test_fsm_peak_torque_uses_prosthesis_left_knee_in_aggregate():
    torque_summary = {
        "torque_source_mode": TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID,
        "joints": {
            "knee": {
                "left": {
                    "torque_smoothness_mean_Nm_per_step": 2.2,
                    "torque_jerk_mean_Nm": 0.4,
                    "peak_torque_Nm": 123.0,
                },
                "right": {
                    "torque_smoothness_mean_Nm_per_step": 0.15,
                    "torque_jerk_mean_Nm": 0.02,
                    "peak_torque_Nm": 10.0,
                },
            },
            "hip": {
                "left": {
                    "torque_smoothness_mean_Nm_per_step": 0.2,
                    "torque_jerk_mean_Nm": 0.03,
                    "peak_torque_Nm": 12.0,
                },
                "right": {
                    "torque_smoothness_mean_Nm_per_step": 0.16,
                    "torque_jerk_mean_Nm": 0.02,
                    "peak_torque_Nm": 12.0,
                },
            },
            "ankle": {
                "left": {
                    "torque_smoothness_mean_Nm_per_step": 0.5,
                    "torque_jerk_mean_Nm": 0.08,
                    "peak_torque_Nm": 49.0,
                },
                "right": {
                    "torque_smoothness_mean_Nm_per_step": 0.38,
                    "torque_jerk_mean_Nm": 0.05,
                    "peak_torque_Nm": 28.0,
                },
            },
        },
    }
    peak = _torque_summary_derived("leg_peak_torque_max", torque_summary)
    assert peak == 123.0
    peak_score = score_lower_is_better(peak, good=90.0, bad=130.0)
    assert 0.0 < peak_score < 100.0
