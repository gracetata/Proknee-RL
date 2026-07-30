#!/usr/bin/env python3
"""Read-only A100 status for the MuJoCo Warp v2 run."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(lines[-1]) if lines else None


def _json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"state": "writing", "error": str(error)}


def _tmux(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
        check=False,
    ).returncode == 0


def _tensorboard() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:6011/", timeout=2) as response:
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def main() -> None:
    output_root = ROOT / "outputs/a100_flat_walk_warp_v2"
    output = output_root / "seed_0"
    checkpoints = sorted((output / "stage1_nn").glob("*.pth"))
    guard = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/a100_gpu_guard.py"),
            "--gpus",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        gpu = json.loads(guard.stdout)
    except json.JSONDecodeError:
        gpu = {"raw": (guard.stdout + guard.stderr).strip()}
    print(
        json.dumps(
            {
                "checked_at": datetime.now(UTC).isoformat(),
                "git_head": subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "tmux_running": _tmux("proknee-a100-flat-walk-warp-v2"),
                "tensorboard_http_6011": _tensorboard(),
                "running_marker": (
                    ROOT / "runtime/a100_flat_walk_warp_v2.running"
                ).is_file(),
                "complete_marker": (
                    ROOT / "runtime/a100_flat_walk_warp_v2.complete"
                ).is_file(),
                "failed_marker": (
                    ROOT / "runtime/a100_flat_walk_warp_v2.failed"
                ).is_file(),
                "latest_metrics": _jsonl(output / "metrics.jsonl"),
                "latest_checkpoint": (
                    str(checkpoints[-1]) if checkpoints else None
                ),
                "full_validation": _json(
                    output_root / "validation_full_trajectories.json"
                ),
                "random_validation": _json(
                    output_root / "validation_random_1000.json"
                ),
                "gpu_guard_exit_code": guard.returncode,
                "gpu": gpu,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
