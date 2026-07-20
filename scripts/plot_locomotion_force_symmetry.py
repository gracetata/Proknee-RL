#!/usr/bin/env python
"""Plot force/contact metrics (simulator.md §3) — left vs right panel comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from musclemimic.distill.config import repo_root

SECTION_SUBDIR = "section3_force"


def resolve_out_dir(base: Path) -> Path:
    base = Path(base)
    if not base.is_absolute():
        base = repo_root() / base
    if base.name == SECTION_SUBDIR:
        return base
    return base / SECTION_SUBDIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--body_weight_N", type=float, default=None, help="Optional; else from metrics")
    return p.parse_args()


def plot_left_right_panels(
    time: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    left_raw: np.ndarray | None = None,
    right_raw: np.ndarray | None = None,
    fill_contact_l: np.ndarray | None = None,
    fill_contact_r: np.ndarray | None = None,
    hline_bw: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    def _panel(ax, y, raw, side_label, color, contact, fill_alpha=0.12):
        if contact is not None:
            ax.fill_between(time, 0, np.max(y) * 1.05 if y.size else 1, where=contact, color=color, alpha=fill_alpha)
        ax.plot(time, y, color=color, lw=1.3, label="filtered")
        if raw is not None and raw.size == y.size:
            ax.plot(time, raw, color=color, alpha=0.35, ls="--", lw=0.9, label="raw")
        if hline_bw is not None and hline_bw > 0:
            ax.axhline(hline_bw / 2.0, color="gray", ls=":", lw=0.8, label="½ body weight")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {side_label}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    _panel(axes[0], left, left_raw, "left", "#2563eb", fill_contact_l)
    _panel(axes[1], right, right_raw, "right", "#dc2626", fill_contact_r)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"{title} (left / right panels)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def loading_rate_series(vgrf: np.ndarray, contact: np.ndarray, dt: float) -> np.ndarray:
    v = np.maximum(np.asarray(vgrf, dtype=np.float64).reshape(-1), 0.0)
    c = np.asarray(contact, dtype=bool).reshape(-1)
    rate = np.zeros_like(v)
    if v.size < 2:
        return rate
    dv = np.diff(v)
    rate[1:] = np.maximum(dv, 0.0) / max(dt, 1e-9)
    rate[~c] = 0.0
    return rate


def cumulative_stance_impulse(vgrf: np.ndarray, contact: np.ndarray, dt: float) -> np.ndarray:
    v = np.maximum(np.asarray(vgrf, dtype=np.float64).reshape(-1), 0.0)
    c = np.asarray(contact, dtype=bool).reshape(-1)
    inc = v * c.astype(np.float64) * dt
    return np.cumsum(inc)


def grf_smoothness_series(vgrf: np.ndarray, dt: float) -> np.ndarray:
    v = np.maximum(np.asarray(vgrf, dtype=np.float64).reshape(-1), 0.0)
    if v.size < 2:
        return np.zeros_like(v)
    return np.abs(np.diff(v, prepend=v[0])) / max(dt, 1e-9)


def symmetry_index(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    denom = 0.5 * (np.abs(l) + np.abs(r))
    si = np.zeros_like(l)
    mask = denom > 1e-6
    si[mask] = np.abs(l[mask] - r[mask]) / denom[mask] * 100.0
    return si


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(Path(args.rollout_npz), allow_pickle=True)
    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    dt = float(np.asarray(data["dt"]).reshape(-1)[0]) if "dt" in data else float(np.median(np.diff(time)) if time.size > 1 else 0.01)

    vgrf_l = np.asarray(data["left_GRF"], dtype=np.float64).reshape(-1)
    vgrf_r = np.asarray(data["right_GRF"], dtype=np.float64).reshape(-1)
    raw_l = np.asarray(data["left_GRF_raw"], dtype=np.float64).reshape(-1) if "left_GRF_raw" in data else None
    raw_r = np.asarray(data["right_GRF_raw"], dtype=np.float64).reshape(-1) if "right_GRF_raw" in data else None
    contact_l = np.asarray(data["contact_left"], dtype=bool).reshape(-1)
    contact_r = np.asarray(data["contact_right"], dtype=bool).reshape(-1)

    if "left_GRF_world" in data:
        lw = np.asarray(data["left_GRF_world"], dtype=np.float64)
        rw = np.asarray(data["right_GRF_world"], dtype=np.float64)
        if lw.ndim == 2:
            horiz_l = np.linalg.norm(lw[:, :2], axis=1)
            horiz_r = np.linalg.norm(rw[:, :2], axis=1)
        else:
            horiz_l = horiz_r = np.zeros_like(vgrf_l)
    else:
        horiz_l = horiz_r = np.zeros_like(vgrf_l)

    bw = args.body_weight_N
    if bw is None:
        metrics_path = Path(args.rollout_npz).parent / "leg_metrics.json"
        if metrics_path.is_file():
            meta = json.loads(metrics_path.read_text(encoding="utf-8"))
            pl = meta.get("force", {}).get("peak_vGRF_left_raw", {}).get("value", 0)
            pr = meta.get("force", {}).get("peak_vGRF_right_raw", {}).get("value", 0)
            bw = float(pl + pr) if pl and pr else None
        if bw is None:
            bw = float(np.max(vgrf_l) + np.max(vgrf_r)) * 1.2

    # 3.1 Peak / time-series vertical GRF
    plot_left_right_panels(
        time,
        vgrf_l,
        vgrf_r,
        title="Vertical GRF (world z)",
        ylabel="N",
        out_path=out_dir / "vgrf_vertical_left_right.png",
        left_raw=raw_l,
        right_raw=raw_r,
        fill_contact_l=contact_l,
        fill_contact_r=contact_r,
        hline_bw=bw,
    )

    # 3.2 GRF impulse (cumulative during stance)
    plot_left_right_panels(
        time,
        cumulative_stance_impulse(vgrf_l, contact_l, dt),
        cumulative_stance_impulse(vgrf_r, contact_r, dt),
        title="GRF impulse (cumulative, stance only)",
        ylabel="N·s",
        out_path=out_dir / "grf_impulse_cumulative_left_right.png",
        fill_contact_l=contact_l,
        fill_contact_r=contact_r,
    )

    # 3.3 Loading rate
    plot_left_right_panels(
        time,
        loading_rate_series(vgrf_l, contact_l, dt),
        loading_rate_series(vgrf_r, contact_r, dt),
        title="Loading rate (positive dGRF/dt)",
        ylabel="N/s",
        out_path=out_dir / "loading_rate_left_right.png",
        fill_contact_l=contact_l,
        fill_contact_r=contact_r,
    )

    # 3.4 GRF smoothness (|dGRF/dt|)
    plot_left_right_panels(
        time,
        grf_smoothness_series(vgrf_l, dt),
        grf_smoothness_series(vgrf_r, dt),
        title="GRF smoothness (|dGRF/dt|)",
        ylabel="N/s",
        out_path=out_dir / "grf_smoothness_left_right.png",
        fill_contact_l=contact_l,
        fill_contact_r=contact_r,
    )

    # 3.6 Contact duration (boolean)
    plot_left_right_panels(
        time,
        contact_l.astype(np.float64),
        contact_r.astype(np.float64),
        title="Foot contact (stance)",
        ylabel="contact (0/1)",
        out_path=out_dir / "contact_left_right.png",
    )

    # 3.8 Foot slip proxy — horizontal GRF magnitude while in contact
    horiz_l_masked = horiz_l * contact_l.astype(np.float64)
    horiz_r_masked = horiz_r * contact_r.astype(np.float64)
    plot_left_right_panels(
        time,
        horiz_l_masked,
        horiz_r_masked,
        title="Foot slip proxy (horizontal |GRF| in contact)",
        ylabel="N",
        out_path=out_dir / "foot_slip_proxy_left_right.png",
        fill_contact_l=contact_l,
        fill_contact_r=contact_r,
    )

    summary = {
        "peak_vGRF_left_N": float(np.max(vgrf_l)),
        "peak_vGRF_right_N": float(np.max(vgrf_r)),
        "GRF_impulse_left_Ns": float(cumulative_stance_impulse(vgrf_l, contact_l, dt)[-1]),
        "GRF_impulse_right_Ns": float(cumulative_stance_impulse(vgrf_r, contact_r, dt)[-1]),
        "contact_duration_left_s": float(np.sum(contact_l) * dt),
        "contact_duration_right_s": float(np.sum(contact_r) * dt),
        "contact_switch_count": int(np.sum(np.diff(contact_l.astype(int)) != 0) + np.sum(np.diff(contact_r.astype(int)) != 0)),
        "GRF_symmetry_index_peak_pct": float(symmetry_index(np.max(vgrf_l), np.max(vgrf_r))),
        "GRF_symmetry_index_impulse_pct": float(
            symmetry_index(
                cumulative_stance_impulse(vgrf_l, contact_l, dt)[-1],
                cumulative_stance_impulse(vgrf_r, contact_r, dt)[-1],
            )
        ),
    }
    with (out_dir / "force_symmetry_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Force symmetry plots saved to {out_dir}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
