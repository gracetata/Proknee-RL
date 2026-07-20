#!/usr/bin/env python
"""Summarize OSL FSM replay metrics across per_motion meta files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPLAY_FOUR = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-root", required=True, help="Run directory with per_motion/*/masked_policy_osl_fsm_meta.json")
    p.add_argument("--output", default=None)
    p.add_argument("--baseline-root", default=None, help="Optional second run root for comparison")
    return p.parse_args()


def short_motion(motion_path: str) -> str:
    parts = motion_path.split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else motion_path


def load_run(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for motion in REPLAY_FOUR:
        safe = motion.replace("/", "_")
        meta_path = root / "per_motion" / safe / "masked_policy_osl_fsm_meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log_path = root / "per_motion" / safe / "masked_policy_osl_fsm.csv"
        first_fall = None
        if log_path.is_file():
            import csv

            with log_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for idx, row in enumerate(reader, start=1):
                    if float(row["root_height"]) < 0.8:
                        first_fall = idx
                        break
        rows[short_motion(motion)] = {
            "done_count": int(meta.get("done_count", -1)),
            "recorded_steps": int(meta.get("recorded_steps", meta.get("n_steps", -1))),
            "episode_return": float(meta.get("episode_return", 0.0)),
            "disabled_muscle_scale": float(meta.get("disabled_muscle_scale", -1)),
            "first_fall_step_lt_0.8m": first_fall,
        }
    return rows


def render_table(name: str, rows: dict[str, dict]) -> list[str]:
    keys = [short_motion(m) for m in REPLAY_FOUR]
    header = ["metric", *keys, "mean"]
    lines = [f"### {name}", "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    fall_vals = []
    fall_cells = []
    for key in keys:
        item = rows.get(key)
        if item is None or item.get("first_fall_step_lt_0.8m") is None:
            fall_cells.append("-")
        else:
            fall_cells.append(str(item["first_fall_step_lt_0.8m"]))
            fall_vals.append(int(item["first_fall_step_lt_0.8m"]))
    mean_fall = f"{sum(fall_vals)/len(fall_vals):.1f}" if fall_vals else "-"
    done_cells = [str(rows[k]["done_count"]) if k in rows else "-" for k in keys]
    lines.append("| first_fall_step | " + " | ".join([*fall_cells, mean_fall]) + " |")
    lines.append("| done_count | " + " | ".join([*done_cells, "-"]) + " |")
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    root = Path(args.eval_root)
    text_lines = render_table(root.name, load_run(root))
    if args.baseline_root:
        base = Path(args.baseline_root)
        text_lines.extend(render_table(base.name, load_run(base)))
    text = "\n".join(text_lines)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
