"""Make the independent source tree and upstream checkout importable."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
# Local development currently uses an outer workspace containing a nested
# upstream checkout. A normal Git clone places this project directly beside
# the ``musclemimic`` Python package. Support both layouts.
NESTED_CHECKOUT = WORKSPACE_ROOT / "musclemimic"
UPSTREAM_ROOT = NESTED_CHECKOUT if (NESTED_CHECKOUT / "pyproject.toml").is_file() else WORKSPACE_ROOT
for path in (PROJECT_ROOT / "src", UPSTREAM_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
