#!/usr/bin/env python3
"""Print a status snapshot for the A100 flat-walking training run."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _tmux(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    ).returncode == 0


def _latest_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(lines[-1]) if lines else None


def _json_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {"state": "writing"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        return {"state": "writing", "json_error": str(error)}


def _http_ok() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:6011/", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> None:
    output = ROOT / "outputs/a100_flat_walk_v1/seed_0"
    checkpoints = sorted((output / "stage1_nn").glob("*.pth"))
    split_path = ROOT / "configs/flat_walk_split.json"
    split = _json_file(split_path)
    guard = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/a100_gpu_guard.py"),
            "--gpus",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    try:
        gpu = json.loads(guard.stdout or guard.stderr)
    except json.JSONDecodeError:
        gpu = {"raw": (guard.stdout + guard.stderr).strip()}
    print(
        json.dumps(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "git_head": subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "tmux_running": _tmux("proknee-a100-flat-walk"),
                "tensorboard_http_6011": _http_ok(),
                "running_marker": (ROOT / "runtime/a100_flat_walk_v1.running").is_file(),
                "complete_marker": (ROOT / "runtime/a100_flat_walk_v1.complete").is_file(),
                "failed_marker": (ROOT / "runtime/a100_flat_walk_v1.failed").is_file(),
                "split_summary": split.get("summary") if split else None,
                "latest_metrics": _latest_jsonl(output / "metrics.jsonl"),
                "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
                "baseline_validation": _json_file(output.parent / "baseline_validation.json"),
                "policy_validation": _json_file(output.parent / "policy_validation.json"),
                "gpu_guard_exit_code": guard.returncode,
                "gpu": gpu,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
