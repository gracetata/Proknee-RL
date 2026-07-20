"""Video recording helpers for locomotion evaluation."""

from __future__ import annotations

import shutil
from pathlib import Path

from musclemimic.evaluation.logger import safe_motion_name


def find_recorded_video(env, motion_path: str) -> Path | None:
    """Locate video written by loco_mujoco VideoRecorder on the env."""
    recorder = getattr(env, "_video_recorder", None) or getattr(env, "video_recorder", None)
    if recorder is None:
        params = getattr(env, "recorder_params", None) or {}
        if not params:
            return None
        base = Path(params.get("path", "."))
        tag = params.get("tag", safe_motion_name(motion_path))
        name = params.get("video_name", "rollout")
        for ext in (".mp4", ".avi", ".webm"):
            candidate = base / tag / f"{name}{ext}"
            if candidate.is_file():
                return candidate
        return None
    path = getattr(recorder, "path", None) or getattr(recorder, "output_path", None)
    if path and Path(path).is_file():
        return Path(path)
    return None


def finalize_rollout_video(
    *,
    env,
    motion_path: str,
    motion_dir: Path,
    success: bool,
    controller_type: str,
) -> Path | None:
    src = find_recorded_video(env, motion_path)
    if src is None or not src.is_file():
        return None
    tag = "success" if success else "fail"
    dest = motion_dir / f"rollout_{controller_type}_{tag}{src.suffix}"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest
