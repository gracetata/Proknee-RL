"""Web-compatible mp4 encoding (H.264 baseline + faststart)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def web_mp4_path(path: str | Path) -> Path:
    p = Path(path)
    if p.stem.endswith("_web"):
        return p
    return p.with_name(f"{p.stem}_web{p.suffix or '.mp4'}")


def encode_web_mp4(
    src_path: str | Path,
    dst_path: str | Path | None = None,
    *,
    remove_src: bool = True,
) -> Path:
    """Re-encode to browser/IDE-friendly mp4; drop the raw imageio file by default."""
    src = Path(src_path)
    dst = Path(dst_path) if dst_path is not None else web_mp4_path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-movflags",
        "+faststart",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if remove_src and src.resolve() != dst.resolve():
        src.unlink(missing_ok=True)
    return dst
