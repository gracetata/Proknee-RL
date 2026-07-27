from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_motion_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_motion_catalog", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_motion_categories_are_unambiguous() -> None:
    assert MODULE.classify_motion("KIT/314/walking_slow03_poses") == "walking_slow"
    assert MODULE.classify_motion("KIT/7/WalkInClockwiseCircle01_poses") == "clockwise_circle"
    assert (
        MODULE.classify_motion("KIT/7/WalkInCounterClockwiseCircle01_poses")
        == "counterclockwise_circle"
    )
    assert MODULE.classify_motion("KIT/11/WalkingStraightBackwards05_poses") == "straight_backward"
    assert MODULE.classify_motion("KIT/167/turn_left01_poses") == "turn_left"
    assert MODULE.classify_motion("KIT/675/walk_with_handrail_left01_poses") == "handrail_walking"
