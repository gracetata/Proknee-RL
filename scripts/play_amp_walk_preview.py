#!/usr/bin/env python3
"""
仅预览 AMP 参考动作 amp_humanoid_walk.npy（SkeletonMotion），使用 poselib 的 matplotlib 交互窗口。

需要：图形界面、matplotlib、运行目录能 import poselib。
不启动 Isaac Gym。

用法（仓库根目录）:
  conda activate proknee_tc   # 或你的 Python 3 环境
  python scripts/play_amp_walk_preview.py

可选:
  python scripts/play_amp_walk_preview.py /path/to/amp_humanoid_walk.npy

交互键（窗口焦点在图上时）:
  x 播放/暂停   z/c 上一帧/下一帧   w 循环   n 退出   h 帮助
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview amp_humanoid_walk.npy with poselib matplotlib")
    parser.add_argument(
        "npy",
        nargs="?",
        default="",
        help="Path to SkeletonMotion .npy (default: IsaacGymEnvs/assets/amp/motions/amp_humanoid_walk.npy)",
    )
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_npy = os.path.join(repo, "IsaacGymEnvs", "assets", "amp", "motions", "amp_humanoid_walk.npy")
    npy_path = os.path.abspath(args.npy) if args.npy else default_npy

    if not os.path.isfile(npy_path):
        print(f"找不到文件: {npy_path}", file=sys.stderr)
        sys.exit(1)

    poselib_parent = os.path.join(repo, "IsaacGymEnvs", "isaacgymenvs", "tasks", "amp", "poselib")
    if not os.path.isdir(poselib_parent):
        print(f"找不到 poselib 目录: {poselib_parent}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, poselib_parent)

    from poselib.skeleton.skeleton3d import SkeletonMotion
    from poselib.visualization.common import plot_skeleton_motion_interactive

    print(f"加载: {npy_path}")
    motion = SkeletonMotion.from_file(npy_path)
    print(f"帧数: {len(motion)}, fps: {motion.fps}")
    print('打开交互窗口后，按 h 查看快捷键，按 n 退出。')
    plot_skeleton_motion_interactive(motion, task_name="amp_humanoid_walk")


if __name__ == "__main__":
    main()
