#!/usr/bin/env python3
"""Read-only status for the official MJLAB/RSL-RL A100 run."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _tmux(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
        check=False,
    ).returncode == 0


def _tensorboard_http() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:6011/", timeout=2) as response:
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def _latest_run() -> Path | None:
    experiment = ROOT / "outputs/mjlab_a100/proknee_mjlab_flat_walk"
    runs = sorted(path for path in experiment.glob("*") if path.is_dir())
    return runs[-1] if runs else None


def _latest_scalars(run: Path | None) -> dict[str, dict[str, float | int]]:
    if run is None:
        return {}
    events = sorted(run.glob("events.out.tfevents.*"))
    if not events:
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(events[-1]))
        accumulator.Reload()
        result = {}
        for tag in accumulator.Tags().get("scalars", []):
            values = accumulator.Scalars(tag)
            if values:
                result[tag] = {
                    "iteration": int(values[-1].step),
                    "value": float(values[-1].value),
                }
        return result
    except Exception as error:  # status must remain usable during partial writes
        return {"error": {"iteration": -1, "value": str(error)}}  # type: ignore[dict-item]


def main() -> None:
    run = _latest_run()
    checkpoints = sorted(run.glob("model_*.pt")) if run else []
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
                "training_tmux": _tmux("proknee-a100-mjlab"),
                "tensorboard_tmux": _tmux("proknee-tensorboard-6011"),
                "tensorboard_http_6011": _tensorboard_http(),
                "latest_run": str(run) if run else None,
                "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
                "latest_scalars": _latest_scalars(run),
                "replay_gate": (
                    json.loads(
                        (ROOT / "outputs/mjlab_a100_replay_gate.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    if (ROOT / "outputs/mjlab_a100_replay_gate.json").is_file()
                    else None
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
