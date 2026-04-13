#!/usr/bin/env python3
"""
去除 AMP 参考动作中的连续重复帧，并导出可直接训练使用的新动作文件。

默认处理:
  - IsaacGymEnvs/assets/amp/motions/amp_humanoid_walk.npy
  - IsaacGymEnvs/assets/amp/motions/amp_humanoid_run.npy

输出:
  - amp_humanoid_walk_dedup.npy
  - amp_humanoid_run_dedup.npy
  - multi_walk_run_dedup.yaml
  - dedup_report.txt

关键策略:
  1) 仅删除“与前一帧几乎完全相同”的连续重复帧（可调 eps）。
  2) 为保持原动作总时长不变，按保留比例缩放 fps:
       new_fps = old_fps * (new_frames / old_frames)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.export_stage0_unified_knee_ankle_trajectories import local_rotation_to_dof


def _load_motion(path: Path) -> Dict:
    data = np.load(path, allow_pickle=True).item()
    if data.get("__name__") != "SkeletonMotion":
        raise ValueError(f"Unsupported motion format in {path}")
    return data


def _frame_diff_max(data: Dict, i: int, j: int) -> float:
    rot = data["rotation"]["arr"]
    root = data["root_translation"]["arr"]
    gv = data["global_velocity"]["arr"]
    ga = data["global_angular_velocity"]["arr"]
    d1 = np.max(np.abs(rot[i] - rot[j]))
    d2 = np.max(np.abs(root[i] - root[j]))
    d3 = np.max(np.abs(gv[i] - gv[j]))
    d4 = np.max(np.abs(ga[i] - ga[j]))
    return float(max(d1, d2, d3, d4))


def _dedup_indices_raw(data: Dict, eps: float) -> np.ndarray:
    n = int(data["rotation"]["arr"].shape[0])
    keep = [0]
    prev = 0
    for i in range(1, n):
        if _frame_diff_max(data, prev, i) > eps:
            keep.append(i)
            prev = i
    return np.asarray(keep, dtype=np.int64)


def _dedup_indices_dof(data: Dict, eps: float) -> np.ndarray:
    rot = torch.tensor(data["rotation"]["arr"], dtype=torch.float32)
    dof = local_rotation_to_dof(rot, device=torch.device("cpu")).numpy()
    n = int(dof.shape[0])
    keep = [0]
    prev = 0
    for i in range(1, n):
        if float(np.max(np.abs(dof[i] - dof[prev]))) > eps:
            keep.append(i)
            prev = i
    return np.asarray(keep, dtype=np.int64)


def _apply_indices(data: Dict, keep_idx: np.ndarray) -> Dict:
    out = dict(data)
    for k in ("rotation", "root_translation", "global_velocity", "global_angular_velocity"):
        sec = dict(out[k])
        sec["arr"] = out[k]["arr"][keep_idx].copy()
        out[k] = sec
    return out


def _rescale_fps(data: Dict, old_n: int, new_n: int) -> Tuple[float, float]:
    old_fps = float(np.asarray(data["fps"]).flat[0])
    if old_n > 1 and new_n > 1:
        # Keep clip duration: (N-1)/fps
        new_fps = old_fps * (float(new_n - 1) / float(old_n - 1))
    else:
        new_fps = old_fps
    data["fps"] = np.array(new_fps, dtype=np.float32)
    return old_fps, new_fps


def _dump_yaml(yaml_path: Path, walk_name: str, run_name: str) -> None:
    text = (
        "# Deduplicated multi-motion bundle (walk + run)\n"
        "motions:\n"
        f"  - file: {walk_name}\n"
        "    weight: 1.0\n"
        f"  - file: {run_name}\n"
        "    weight: 1.0\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Deduplicate consecutive repeated frames for AMP motions.")
    p.add_argument("--motions-dir", type=Path, default=None, help="Directory containing amp_humanoid_*.npy")
    p.add_argument("--eps", type=float, default=1e-8, help="Frame equality threshold")
    p.add_argument(
        "--space",
        type=str,
        default="dof",
        choices=("dof", "raw"),
        help="Compare space: dof (recommended) or raw arrays",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    motions_dir = (args.motions_dir or (repo / "IsaacGymEnvs" / "assets" / "amp" / "motions")).resolve()
    walk_in = motions_dir / "amp_humanoid_walk.npy"
    run_in = motions_dir / "amp_humanoid_run.npy"
    if not walk_in.is_file() or not run_in.is_file():
        raise FileNotFoundError(f"Missing source motions under {motions_dir}")

    outputs = []
    report_lines = []
    for src in (walk_in, run_in):
        data = _load_motion(src)
        n_old = int(data["rotation"]["arr"].shape[0])
        keep_idx = _dedup_indices_dof(data, eps=args.eps) if args.space == "dof" else _dedup_indices_raw(data, eps=args.eps)
        n_new = int(keep_idx.shape[0])

        new_data = _apply_indices(data, keep_idx)
        old_fps, new_fps = _rescale_fps(new_data, n_old, n_new)
        old_dur = (n_old - 1) / old_fps if n_old > 1 else 0.0
        new_dur = (n_new - 1) / new_fps if n_new > 1 else 0.0

        out_name = f"{src.stem}_dedup.npy"
        out_path = motions_dir / out_name
        np.save(out_path, new_data)
        outputs.append(out_name)

        report_lines.append(
            f"{src.name} [{args.space}]: frames {n_old} -> {n_new}, "
            f"fps {old_fps:.6f} -> {new_fps:.6f}, "
            f"duration {old_dur:.6f}s -> {new_dur:.6f}s"
        )

    yaml_out = motions_dir / "multi_walk_run_dedup.yaml"
    _dump_yaml(yaml_out, outputs[0], outputs[1])

    report = motions_dir / "dedup_report.txt"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("[ok] generated:")
    print(f"  - {motions_dir / outputs[0]}")
    print(f"  - {motions_dir / outputs[1]}")
    print(f"  - {yaml_out}")
    print(f"  - {report}")
    for line in report_lines:
        print(" ", line)


if __name__ == "__main__":
    main()
