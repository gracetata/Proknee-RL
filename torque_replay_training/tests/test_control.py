from __future__ import annotations

import numpy as np

from torque_replay_training.control import compose_prosthesis_torque, root_up_z


def test_zero_residual_is_exact_baseline() -> None:
    baseline = np.asarray([130.0, -140.0, 10.0, -5.0])
    command, residual = compose_prosthesis_torque(
        baseline,
        np.zeros(4),
        np.asarray([20.0, 20.0, 8.0, 5.0]),
        np.asarray([120.0, 120.0, 35.0, 25.0]),
        exact_baseline=True,
    )
    np.testing.assert_array_equal(command, baseline)
    np.testing.assert_array_equal(residual, np.zeros(4))


def test_residual_and_command_are_bounded() -> None:
    command, residual = compose_prosthesis_torque(
        np.asarray([115.0, -115.0, 0.0, 0.0]),
        np.asarray([2.0, -2.0, 0.5, -0.5]),
        np.asarray([20.0, 20.0, 8.0, 5.0]),
        np.asarray([120.0, 120.0, 35.0, 25.0]),
    )
    np.testing.assert_allclose(residual, [20.0, -20.0, 4.0, -2.5])
    np.testing.assert_allclose(command, [120.0, -120.0, 4.0, -2.5])


def test_root_up_z() -> None:
    qpos = np.zeros(7)
    qpos[3] = 1.0
    assert root_up_z(qpos) == 1.0
