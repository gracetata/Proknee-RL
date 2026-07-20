"""Hexagonal spider chart for composite locomotion dimension scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Stable axis order (clockwise from top); must match conf_eval_composite_score.yaml.
DIMENSION_ORDER: tuple[str, ...] = (
    "task_reliability",
    "pose_tracking",
    "global_trajectory",
    "contact_biomechanics",
    "bilateral_symmetry",
    "control_quality",
)

# Text alignment per vertex (top → clockwise).
_VERTEX_LABEL_ALIGN: list[tuple[str, str]] = [
    ("center", "bottom"),  # top
    ("left", "bottom"),    # top-right
    ("left", "top"),       # bottom-right
    ("center", "top"),     # bottom
    ("right", "top"),      # bottom-left
    ("right", "bottom"),   # top-left
]


def _setup_cjk_font() -> str | None:
    """Prefer a CJK-capable font so Chinese axis labels render."""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK HK",
        "AR PL UMing CN",
        "WenQuanYi Micro Hei",
        "SimHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            family = prop.get_name()
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return family
    return None


def _axis_angles(n: int, *, start: float = np.pi / 2) -> np.ndarray:
    """Vertex angles: first axis at top, then clockwise."""
    return start - np.arange(n, dtype=float) * (2.0 * np.pi / n)


def _vertices_from_scores(
    scores: np.ndarray, angles: np.ndarray, *, max_r: float = 1.0
) -> np.ndarray:
    """Map each score to (x, y) on its own spoke (one radius per angle)."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    angles = np.asarray(angles, dtype=float).reshape(-1)
    if scores.shape != angles.shape:
        raise ValueError(
            f"scores/angles length mismatch: {scores.shape} vs {angles.shape}"
        )
    radii = (scores / 100.0) * max_r
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _closed_ring(vertices: np.ndarray) -> np.ndarray:
    """Append first vertex so the polyline explicitly closes."""
    return np.vstack([vertices, vertices[0:1]])


def _format_axis_label(dim_block: dict[str, Any], dim_id: str) -> str:
    zh = dim_block.get("label_zh") or ""
    en = dim_block.get("label_en") or dim_id
    if zh and en:
        return f"{zh}\n{en}"
    return zh or en


def _ordered_dimensions(dimensions: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    ordered: list[tuple[str, dict[str, Any]]] = []
    for dim_id in DIMENSION_ORDER:
        if dim_id in dimensions:
            ordered.append((dim_id, dimensions[dim_id]))
    for dim_id, block in dimensions.items():
        if dim_id not in DIMENSION_ORDER:
            ordered.append((dim_id, block))
    return ordered


def plot_composite_radar(
    report: dict[str, Any],
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    dimensions = report.get("dimensions", {})
    if not dimensions:
        raise ValueError("No dimensions in composite report")

    _setup_cjk_font()

    ordered = _ordered_dimensions(dimensions)
    dim_ids = [d for d, _ in ordered]
    n = len(dim_ids)
    angles = _axis_angles(n)
    scores = np.array([float(block["score"]) for _, block in ordered], dtype=float)
    labels = [_format_axis_label(block, dim_id) for dim_id, block in ordered]

    max_r = 1.0
    label_r = max_r + 0.22

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.axis("off")

    # Hexagonal grid rings.
    grid_levels = [20, 40, 60, 80, 100]
    for level in grid_levels:
        r = level / 100.0 * max_r
        ring = _vertices_from_scores(np.full(n, level), angles, max_r=max_r)
        closed_ring = _closed_ring(ring)
        ax.plot(
            closed_ring[:, 0],
            closed_ring[:, 1],
            color="#cbd5e1",
            linewidth=0.9,
            zorder=1,
        )
        # Tick label slightly right of top spoke (avoid overlapping data vertex).
        ax.text(
            0.04,
            r - 0.02,
            str(level),
            ha="left",
            va="center",
            fontsize=8,
            color="#64748b",
            zorder=2,
        )

    # Radial spokes.
    outer = _vertices_from_scores(np.full(n, 100.0), angles, max_r=max_r)
    for vx, vy in outer:
        ax.plot([0.0, vx], [0.0, vy], color="#cbd5e1", linewidth=0.9, zorder=1)

    # Data polygon: single closed polyline (no separate Polygon patch).
    data_xy = _vertices_from_scores(scores, angles, max_r=max_r)
    closed_data = _closed_ring(data_xy)
    ax.fill(
        closed_data[:, 0],
        closed_data[:, 1],
        facecolor="#2563eb",
        edgecolor="none",
        alpha=0.30,
        zorder=3,
    )
    # Outline: closed polyline; markers only on the six data vertices (not on duplicate close point).
    ax.plot(
        closed_data[:, 0],
        closed_data[:, 1],
        color="#1d4ed8",
        linewidth=2.2,
        zorder=4,
    )
    ax.plot(
        data_xy[:, 0],
        data_xy[:, 1],
        linestyle="none",
        marker="o",
        markersize=6,
        color="#1d4ed8",
        zorder=5,
    )

    # Axis labels and per-vertex score annotations.
    for i, (angle, label, score) in enumerate(zip(angles, labels, scores)):
        dr = (score / 100.0) * max_r
        lx, ly = label_r * np.cos(angle), label_r * np.sin(angle)
        ha, va = _VERTEX_LABEL_ALIGN[i % len(_VERTEX_LABEL_ALIGN)]
        ax.text(
            lx,
            ly,
            label,
            ha=ha,
            va=va,
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
            linespacing=1.15,
            zorder=6,
        )
        sx, sy = (dr + 0.06) * np.cos(angle), (dr + 0.06) * np.sin(angle)
        ax.text(
            sx,
            sy,
            f"{score:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#1e3a8a",
            zorder=6,
        )

    motion = report.get("motion_path", "")
    ctrl = report.get("controller_type", "")
    total = report.get("total_score", 0.0)
    motion_type = report.get("motion_type", "")
    n_dim = len(dim_ids)
    if title is None:
        title = (
            f"综合评分 Composite score ({n_dim}D)\n{motion}\n"
            f"{motion_type} | {ctrl} | total={total:.1f}/100"
        )
    ax.set_title(title, fontsize=12, pad=12, y=1.02)

    margin = 0.35
    ax.set_xlim(-max_r - margin, max_r + margin)
    ax.set_ylim(-max_r - margin, max_r + margin)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
