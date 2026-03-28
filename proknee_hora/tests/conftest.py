"""Pytest configuration for proknee_hora tests.

Isaac Gym **must** be imported before torch.  This conftest ensures that
happens before any test module loads.
"""

import pytest

try:
    import isaacgym  # noqa: F401
    ISAACGYM_AVAILABLE = True
except (ImportError, OSError):
    ISAACGYM_AVAILABLE = False


def pytest_collection_modifyitems(config, items):
    """Skip tests that require Isaac Gym when it is not available."""
    if ISAACGYM_AVAILABLE:
        return
    skip_ig = pytest.mark.skip(reason="Isaac Gym not available or import-order conflict")
    for item in items:
        # Any test in files that import ProKneeBase/Teacher/Student
        if "env" in item.nodeid or "training" in item.nodeid:
            item.add_marker(skip_ig)
