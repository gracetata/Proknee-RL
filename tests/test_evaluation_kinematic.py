"""Unit tests for kinematic metrics from synthetic rollout buffers."""

from __future__ import annotations

import math

import numpy as np

from musclemimic.evaluation.kinematic_metrics import compute_kinematic_metrics
from musclemimic.evaluation.types import RolloutBuffer


def test_kinematic_metrics_from_info_steps():
    buf = RolloutBuffer(motion_path="KIT/1/walking_test", dt=0.01, traj_length=100)
    for i in range(50):
        buf.append_step(
            t=i * 0.01,
            qpos=np.zeros(10),
            qvel=np.zeros(10),
            root_pos=np.array([0.0, 0.0, 1.0]),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ref_root_pos=np.array([0.0, 0.0, 1.0]),
            ref_qpos=np.zeros(10),
            site_pos=None,
            ref_site_pos=None,
            left_grf=0.0,
            right_grf=0.0,
            left_grf_world=np.zeros(3),
            right_grf_world=np.zeros(3),
            contact_left=False,
            contact_right=False,
            joint_torque=np.zeros(4),
            prosthesis_tau=np.zeros(4),
            reward=1.0,
            done=False,
            info={
                "err_joint_pos": 0.01,
                "err_joint_vel": 0.02,
                "err_root_xyz": 0.03,
                "err_rpos": 0.04,
                "err_site_abs": 0.05,
            },
        )
    buf.done_reason = "completed"
    metrics = compute_kinematic_metrics(buf, {})
    assert metrics["frame_coverage"]["value"] == 0.5
    assert metrics["success"]["value"] == 0
    deg = 180.0 / math.pi
    assert abs(metrics["joint_angle_error_deg"]["mean"]["value"] - 0.01 * deg) < 1e-5
