#!/usr/bin/env python3
"""Print one JSON snapshot of the guarded A100 training and TensorBoard state."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def _tmux_session(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def _latest_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> None:
    status = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "watchdog_running": _tmux_session("proknee-a100-watchdog"),
        "pipeline_running": _tmux_session("proknee-a100-train"),
        "tensorboard_http_6011": _http_ok("http://127.0.0.1:6011/"),
        "complete": (ROOT / "runtime/a100_train_v1.complete").is_file(),
        "failed": (ROOT / "runtime/a100_train_v1.failed").is_file(),
        "pipeline_failed": (ROOT / "runtime/a100_pipeline.failed").is_file(),
        "seeds": {},
    }
    for seed in (0, 1, 2):
        directory = ROOT / f"outputs/a100_train_v1/seed_{seed}"
        checkpoints = sorted(directory.glob("policy_*.msgpack"))
        status["seeds"][str(seed)] = {
            "latest_metrics": _latest_json(directory / "metrics.jsonl"),
            "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
            "train_log_exists": (directory / "train.log").is_file(),
        }
    guard = subprocess.run(
        [
            str(ROOT.parent / ".venv/bin/python"),
            str(ROOT / "scripts/a100_gpu_guard.py"),
            "--gpus",
            "5",
            "6",
            "7",
        ],
        capture_output=True,
        text=True,
    )
    status["gpu_guard_exit_code"] = guard.returncode
    try:
        status["gpu_guard"] = json.loads(guard.stdout or guard.stderr)
    except json.JSONDecodeError:
        status["gpu_guard"] = {"error": (guard.stdout + guard.stderr).strip(), "exit_code": guard.returncode}
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
