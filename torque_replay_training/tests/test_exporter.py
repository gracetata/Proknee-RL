from __future__ import annotations

import pytest

from torque_replay_training.exporter import _control_steps_for_trajectory


def test_reference_frames_convert_to_transitions() -> None:
    assert _control_steps_for_trajectory(773) == 772
    assert _control_steps_for_trajectory(2) == 1


def test_reference_trajectory_requires_two_frames() -> None:
    with pytest.raises(ValueError, match="at least two state frames"):
        _control_steps_for_trajectory(1)
