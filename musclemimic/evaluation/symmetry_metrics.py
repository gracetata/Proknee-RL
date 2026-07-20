"""Symmetry-focused metric helpers."""

from __future__ import annotations

from musclemimic.evaluation.force_metrics import compute_grf_metrics


def _symmetry_index(left: float, right: float) -> float | None:
    denom = 0.5 * (abs(left) + abs(right))
    if denom < 1e-9:
        return None
    return float(abs(left - right) / denom * 100.0)


def compute_symmetry_metrics(buffer, meta: dict) -> dict:
    grf = compute_grf_metrics(buffer, meta)
    out: dict = {}
    for key in ("GRF_symmetry_index_peak_filtered", "GRF_symmetry_index_impulse_filtered"):
        if key in grf:
            out[key] = grf[key]
    cd_l = grf.get("contact_duration_left", {}).get("value")
    cd_r = grf.get("contact_duration_right", {}).get("value")
    if cd_l is not None and cd_r is not None:
        out["contact_duration_symmetry_index"] = {
            "value": _symmetry_index(float(cd_l), float(cd_r)),
            "unit": "%",
        }
    return out
