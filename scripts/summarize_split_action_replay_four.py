#!/usr/bin/env python
"""Summarize first_done_step across split-action replay_four runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_root",
        default="outputs/eval_split_action_prosthesis/replay_four",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def short_motion(motion_path: str) -> str:
    parts = motion_path.split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else motion_path


def load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = {}
    for item in payload.get("motions", []):
        rows[short_motion(str(item["motion_path"]))] = {
            "first_done_step": int(item.get("first_done_step", -1)),
            "done_count": int(item.get("done_count", -1)),
            "n_steps": int(item.get("n_steps", -1)),
        }
    return {"run_name": run_dir.name, "motions": rows}


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root)
    runs = []
    for run_dir in sorted(eval_root.iterdir()):
        if not run_dir.is_dir():
            continue
        loaded = load_run(run_dir)
        if loaded is not None:
            runs.append(loaded)
    if not runs:
        print(f"No summary.json found under {eval_root}")
        return 1

    motion_keys = sorted({k for run in runs for k in run["motions"]})
    header = ["run_name", *motion_keys, "mean_first_done"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for run in runs:
        vals = []
        fds = []
        for key in motion_keys:
            item = run["motions"].get(key)
            if item is None:
                vals.append("-")
            else:
                fd = item["first_done_step"]
                vals.append(str(fd))
                fds.append(fd)
        mean_fd = f"{sum(fds)/len(fds):.1f}" if fds else "-"
        lines.append("| " + " | ".join([run["run_name"], *vals, mean_fd]) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
