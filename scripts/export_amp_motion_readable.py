#!/usr/bin/env python3
"""
将 AMP 参考动作 amp_humanoid_walk.npy / amp_humanoid_run.npy 导出为可读文本与 CSV。

原始 .npy 为 numpy pickle 字典（SkeletonMotion），不适合直接当文本打开。
本脚本仅依赖 numpy，无需 Isaac Gym。

用法:
  python scripts/export_amp_motion_readable.py
  python scripts/export_amp_motion_readable.py /path/to/amp_humanoid_walk.npy -o /tmp/out

会在输出目录生成:
  <stem>_summary.txt  — 元数据与统计
  <stem>_frames.csv    — 每帧时间、根位置、根线速度（关节 0）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def load_motion_dict(path: str) -> dict:
    d = __import__("numpy").load(path, allow_pickle=True).item()
    if d.get("__name__") != "SkeletonMotion":
        print(f"警告: __name__={d.get('__name__')!r}，仍尝试按 SkeletonMotion 结构解析。", file=sys.stderr)
    return d


def export_one(npy_path: Path, out_dir: Path) -> None:
    import numpy as np

    d = load_motion_dict(str(npy_path))
    fps = int(np.asarray(d["fps"]).flat[0])
    rot = d["rotation"]["arr"]
    n_frames = int(rot.shape[0])
    n_joints = int(rot.shape[1])
    root_t = d["root_translation"]["arr"]
    gv = d["global_velocity"]["arr"]
    ga = d["global_angular_velocity"]["arr"]

    stem = npy_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{stem}_summary.txt"
    csv_path = out_dir / f"{stem}_frames.csv"

    # 根节点线速度、角速度（关节索引 0）
    root_v = gv[:, 0, :]
    root_w = ga[:, 0, :]
    dt = 1.0 / fps
    duration = dt * (n_frames - 1) if n_frames > 1 else 0.0

    lines = [
        f"file: {npy_path.name}",
        f"__name__: {d.get('__name__', '?')}",
        f"fps: {fps}",
        f"frames: {n_frames}",
        f"joints: {n_joints}",
        f"duration_s (approx): {duration:.4f}",
        f"rotation arr: {tuple(rot.shape)} {rot.dtype}",
        f"root_translation arr: {tuple(root_t.shape)} {root_t.dtype}",
        f"global_velocity arr: {tuple(gv.shape)} {gv.dtype}",
        "",
        "root forward velocity (x) stats [m/s]:",
        f"  min={float(root_v[:, 0].min()):.4f} max={float(root_v[:, 0].max()):.4f} mean={float(root_v[:, 0].mean()):.4f}",
        "",
        "root position (world) stats [m]:",
        f"  x: min={float(root_t[:, 0].min()):.4f} max={float(root_t[:, 0].max()):.4f}",
        f"  y: min={float(root_t[:, 1].min()):.4f} max={float(root_t[:, 1].max()):.4f}",
        f"  z: min={float(root_t[:, 2].min()):.4f} max={float(root_t[:, 2].max()):.4f}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "time_s",
                "root_x",
                "root_y",
                "root_z",
                "root_vx",
                "root_vy",
                "root_vz",
                "root_wx",
                "root_wy",
                "root_wz",
            ]
        )
        for i in range(n_frames):
            t = i * dt
            w.writerow(
                [
                    i,
                    f"{t:.6f}",
                    *[f"{float(x):.8f}" for x in root_t[i]],
                    *[f"{float(x):.8f}" for x in root_v[i]],
                    *[f"{float(x):.8f}" for x in root_w[i]],
                ]
            )
    print(f"Wrote {csv_path} ({n_frames} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AMP .npy motion to summary txt + CSV.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Path(s) to amp_humanoid_*.npy (default: walk+run under IsaacGymEnvs/assets/amp/motions/)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: same directory as each .npy, subfolder readable/)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    default_dir = repo / "IsaacGymEnvs" / "assets" / "amp" / "motions"
    if not args.inputs:
        inputs = [
            default_dir / "amp_humanoid_walk.npy",
            default_dir / "amp_humanoid_run.npy",
        ]
    else:
        inputs = [Path(p) for p in args.inputs]

    for p in inputs:
        if not p.is_file():
            print(f"跳过（不存在）: {p}", file=sys.stderr)
            continue
        if args.out_dir:
            out = Path(args.out_dir)
        else:
            out = p.parent / "readable"
        export_one(p, out)

    print("完成。请打开 *_summary.txt 与 *_frames.csv（可用 Excel / LibreOffice）。")


if __name__ == "__main__":
    main()
