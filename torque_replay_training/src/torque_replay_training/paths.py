"""Workspace path discovery without hard-coding the account name."""

from __future__ import annotations

from pathlib import Path


NEW_PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = NEW_PROJECT_ROOT.parent
NESTED_CHECKOUT = WORKSPACE_ROOT / "musclemimic"
UPSTREAM_ROOT = NESTED_CHECKOUT if (NESTED_CHECKOUT / "pyproject.toml").is_file() else WORKSPACE_ROOT
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "data" / "checkpoints" / "mm-10m-2"
DEFAULT_DATA_ROOT = NEW_PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = NEW_PROJECT_ROOT / "outputs"


def resolve_from_project(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else NEW_PROJECT_ROOT / value
