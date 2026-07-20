#!/usr/bin/env python
"""Render front-view MyoFullBody muscle tendons; knee15 left-leg muscles in blue."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.collections import LineCollection

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.prosthesis.constants import KNEE15_DISABLED_MUSCLE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="figures/myofullbody_knee15_front_muscles.png",
        help="Output PNG path",
    )
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=14.0)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def tendon_ids_for_muscles(model: mujoco.MjModel, muscle_names: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in muscle_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"Missing actuator {name!r}")
        out[name] = int(model.actuator_trnid[aid, 0])
    return out


def tendon_polyline(model: mujoco.MjModel, data: mujoco.MjData, tendon_id: int) -> np.ndarray:
    points: list[np.ndarray] = []
    adr = int(model.tendon_adr[tendon_id])
    num = int(model.tendon_num[tendon_id])
    for i in range(num):
        wtype = int(model.wrap_type[adr + i])
        objid = int(model.wrap_objid[adr + i])
        if wtype == int(mujoco.mjtWrap.mjWRAP_SITE):
            points.append(np.asarray(data.site_xpos[objid], dtype=np.float64))
        elif wtype in {
            int(mujoco.mjtWrap.mjWRAP_SPHERE),
            int(mujoco.mjtWrap.mjWRAP_CYLINDER),
        }:
            points.append(np.asarray(data.geom_xpos[objid], dtype=np.float64))
        elif wtype == int(mujoco.mjtWrap.mjWRAP_PULLEY):
            points.append(np.asarray(data.site_xpos[objid], dtype=np.float64))
    if len(points) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(points, axis=0)


def front_projection(points: np.ndarray) -> np.ndarray:
    """Project 3D points to front view: x horizontal, z vertical."""
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([points[:, 0], points[:, 2]], axis=1)


def body_scatter(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    pts = []
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            pts.append(np.asarray(data.geom_xpos[gid], dtype=np.float64))
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return front_projection(np.stack(pts, axis=0))


def render_png(output: Path, width: float, height: float, dpi: int) -> None:
    env = MyoFullBody(disable_fingers=True, headless=True)
    try:
        model, data = env.model, env.data
        env.reset()
        mujoco.mj_forward(model, data)

        highlight_ids = set(tendon_ids_for_muscles(model, KNEE15_DISABLED_MUSCLE_NAMES).values())
        gray_segments: list[np.ndarray] = []
        blue_segments: list[np.ndarray] = []

        for tid in range(model.ntendon):
            poly = tendon_polyline(model, data, tid)
            if poly.shape[0] < 2:
                continue
            seg2d = front_projection(poly)
            segs = np.stack([seg2d[:-1], seg2d[1:]], axis=1)
            if tid in highlight_ids:
                blue_segments.extend(segs)
            else:
                gray_segments.extend(segs)

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fb")

        body_pts = body_scatter(model, data)
        if body_pts.size:
            ax.scatter(body_pts[:, 0], body_pts[:, 1], s=4, c="#d9dde3", alpha=0.35, linewidths=0)

        if gray_segments:
            ax.add_collection(
                LineCollection(gray_segments, colors="#b8bcc4", linewidths=0.8, alpha=0.55, zorder=2)
            )
        if blue_segments:
            ax.add_collection(
                LineCollection(blue_segments, colors="#1f6fd6", linewidths=2.8, alpha=0.95, zorder=3)
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (front view)")
        ax.set_ylabel("Z (height)")
        ax.set_title("MyoFullBody front view: knee15 left-leg muscles (blue)")
        ax.grid(True, alpha=0.15, linewidth=0.5)

        labels = ", ".join(KNEE15_DISABLED_MUSCLE_NAMES)
        fig.text(
            0.5,
            0.02,
            f"knee15 disabled muscles ({len(KNEE15_DISABLED_MUSCLE_NAMES)}): {labels}",
            ha="center",
            va="bottom",
            fontsize=7,
            wrap=True,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output, dpi=dpi, facecolor="white")
        plt.close(fig)
    finally:
        env.stop()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    render_png(output, args.width, args.height, args.dpi)
    print(f"Saved: {output}")
    print(f"Highlighted {len(KNEE15_DISABLED_MUSCLE_NAMES)} knee15 left-leg muscles in blue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
