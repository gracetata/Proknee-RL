"""Composite locomotion score: normalize sub-metrics to 0-100 per dimension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from musclemimic.distill.config import repo_root
from musclemimic.evaluation.motion_taxonomy import classify_motion_type
from musclemimic.evaluation.scoring_profile import (
    apply_profile_to_config,
    resolve_scoring_profile,
)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def score_lower_is_better(x: float, good: float, bad: float) -> float:
    if x <= good:
        return 100.0
    if x >= bad:
        return 0.0
    return _clamp(100.0 * (bad - x) / (bad - good))


def score_higher_is_better(x: float, good: float, bad: float) -> float:
    if x >= good:
        return 100.0
    if x <= bad:
        return 0.0
    return _clamp(100.0 * (x - bad) / (good - bad))


def score_in_band(
    x: float,
    low: float,
    high: float,
    ideal: float | None = None,
    soft_margin: float = 0.15,
) -> float:
    """100 at ideal (or band center); linear decay outside [low, high] with soft margin."""
    if ideal is None:
        ideal = 0.5 * (low + high)
    if low <= x <= high:
        half = max((high - low) * 0.5, 1e-9)
        return _clamp(100.0 - 20.0 * abs(x - ideal) / half)
    span = max(high - low, 1e-9)
    if x < low:
        dist = (low - x) / span
    else:
        dist = (x - high) / span
    return _clamp(100.0 * (1.0 - dist / max(soft_margin, 1e-9)))


def score_symmetry(si_pct: float, si_bad: float = 40.0) -> float:
    return _clamp(100.0 * max(0.0, 1.0 - float(si_pct) / si_bad))


def _get_nested(obj: dict, dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, dict) and "value" in cur:
        return cur.get("value")
    return cur


def _body_weight_n(metrics: dict) -> float | None:
    peak = _get_nested(metrics, "force.peak_vGRF_left_filtered.value")
    norm = _get_nested(metrics, "force.peak_vGRF_left_filtered_norm.value")
    if peak is None or norm is None or float(norm) < 1e-6:
        return None
    return float(peak) / float(norm)


def _derive_value(name: str, metrics: dict, symmetry_summary: dict | None) -> float | None:
    bw = _body_weight_n(metrics)
    if name == "mean_vgrf_stance_norm_left":
        mean_v = _get_nested(metrics, "force.mean_vGRF_left_filtered.value")
        return float(mean_v) / bw if mean_v is not None and bw else None
    if name == "mean_vgrf_stance_norm_right":
        mean_v = _get_nested(metrics, "force.mean_vGRF_right_filtered.value")
        return float(mean_v) / bw if mean_v is not None and bw else None
    return None


def _torque_summary_derived(name: str, torque_summary: dict | None) -> float | None:
    if not torque_summary:
        return None
    joints = torque_summary.get("joints", {})
    smooth_vals: list[float] = []
    jerk_vals: list[float] = []
    peak_vals: list[float] = []
    for side_key in ("left", "right"):
        for jname in ("knee", "hip", "ankle"):
            block = joints.get(jname, {}).get(side_key, {})
            if not block:
                continue
            if "torque_smoothness_mean_Nm_per_step" in block:
                smooth_vals.append(float(block["torque_smoothness_mean_Nm_per_step"]))
            if "torque_jerk_mean_Nm" in block:
                jerk_vals.append(float(block["torque_jerk_mean_Nm"]))
            if "peak_torque_Nm" in block:
                peak_vals.append(float(block["peak_torque_Nm"]))
    if name == "leg_torque_smoothness_mean":
        return float(sum(smooth_vals) / len(smooth_vals)) if smooth_vals else None
    if name == "leg_torque_jerk_mean":
        return float(sum(jerk_vals) / len(jerk_vals)) if jerk_vals else None
    if name == "leg_peak_torque_max":
        return float(max(peak_vals)) if peak_vals else None
    return None


def _resolve_raw_value(
    spec: dict,
    metrics: dict,
    symmetry_summary: dict | None,
    torque_summary: dict | None,
) -> float | None:
    source = spec.get("source", "metrics")
    if source == "metrics":
        val = _get_nested(metrics, str(spec["path"]))
    elif source == "symmetry_summary":
        val = _get_nested(symmetry_summary or {}, str(spec["path"]))
    elif source == "derived":
        val = _derive_value(str(spec.get("derived")), metrics, symmetry_summary)
    elif source == "torque_summary":
        val = _torque_summary_derived(str(spec.get("derived")), torque_summary)
    else:
        val = None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def score_submetric(
    spec: dict,
    metrics: dict,
    symmetry_summary: dict | None,
    torque_summary: dict | None,
) -> tuple[float | None, float | None]:
    raw = _resolve_raw_value(spec, metrics, symmetry_summary, torque_summary)
    if raw is None:
        return None, None

    stype = spec.get("type", "lower_is_better")
    if stype == "lower_is_better":
        s = score_lower_is_better(raw, float(spec["good"]), float(spec["bad"]))
    elif stype == "higher_is_better":
        s = score_higher_is_better(raw, float(spec["good"]), float(spec["bad"]))
    elif stype == "in_band":
        s = score_in_band(
            raw,
            float(spec["band_low"]),
            float(spec["band_high"]),
            ideal=float(spec["band_ideal"]) if "band_ideal" in spec else None,
        )
    elif stype == "symmetry_si":
        s = score_symmetry(raw, float(spec.get("si_bad", 40.0)))
    else:
        return raw, None
    return raw, s


def load_composite_config(config_name: str = "conf_eval_composite_score") -> dict[str, Any]:
    cfg_dir = repo_root() / "fullbody"
    path = cfg_dir / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = OmegaConf.load(path)
    block = OmegaConf.to_container(cfg.get("composite_score", cfg), resolve=True)
    return dict(block or {})


def compute_dimension_scores(
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    symmetry_summary: dict | None = None,
    torque_summary: dict | None = None,
) -> dict[str, Any]:
    dims_cfg = config.get("dimensions", {})
    out: dict[str, Any] = {}
    for dim_id, dim_spec in dims_cfg.items():
        subscores: dict[str, Any] = {}
        weighted: list[tuple[float, float]] = []
        for sm in dim_spec.get("submetrics", []):
            sid = str(sm["id"])
            raw, score = score_submetric(sm, metrics, symmetry_summary, torque_summary)
            entry: dict[str, Any] = {"raw": raw, "score": score}
            if raw is None:
                entry["missing"] = True
            subscores[sid] = entry
            w = float(sm.get("weight", 1.0))
            if score is not None:
                weighted.append((score, w))

        if weighted:
            wsum = sum(w for _, w in weighted)
            dim_score = sum(s * w for s, w in weighted) / max(wsum, 1e-9)
        else:
            dim_score = 0.0

        out[dim_id] = {
            "label_zh": dim_spec.get("label_zh", dim_id),
            "label_en": dim_spec.get("label_en", dim_id),
            "score": _clamp(dim_score),
            "subscores": subscores,
        }
    return out


def compute_total_score(
    dimension_scores: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    active_weights: dict[str, float] | None = None,
) -> float:
    weights = active_weights or config.get("dimension_weights", {})
    total_w = 0.0
    acc = 0.0
    for dim_id, w in weights.items():
        block = dimension_scores.get(dim_id)
        if block is None:
            continue
        wf = float(w)
        total_w += wf
        acc += wf * float(block["score"])
    total = acc / max(total_w, 1e-9)

    success = _get_nested(metrics, "official_imitation.success.value")
    if success is not None and float(success) < 0.5:
        total = min(total, float(config.get("fail_cap_total", 50.0)))
    return _clamp(total)


def build_composite_report(
    metrics_path: Path | str,
    *,
    config_name: str = "conf_eval_composite_score",
    symmetry_summary_path: Path | str | None = None,
    torque_summary_path: Path | str | None = None,
    symmetry_summary: dict[str, Any] | None = None,
    torque_summary: dict[str, Any] | None = None,
    motion_path: str | None = None,
) -> dict[str, Any]:
    metrics_path = Path(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if symmetry_summary is None and symmetry_summary_path:
        symmetry_summary = json.loads(Path(symmetry_summary_path).read_text(encoding="utf-8"))
    if torque_summary is None and torque_summary_path:
        torque_summary = json.loads(Path(torque_summary_path).read_text(encoding="utf-8"))

    mpath = motion_path or metrics.get("motion_path")
    motion_type = metrics.get("motion_type") or (
        classify_motion_type(str(mpath)) if mpath else "other"
    )
    profile = resolve_scoring_profile(motion_type)
    base_config = load_composite_config(config_name)
    config = apply_profile_to_config(base_config, profile)

    dimensions = compute_dimension_scores(
        config,
        metrics,
        symmetry_summary=symmetry_summary,
        torque_summary=torque_summary,
    )
    total = compute_total_score(
        dimensions,
        config,
        metrics,
        active_weights=config.get("dimension_weights"),
    )

    return {
        "note": profile.summary_zh,
        "motion_path": mpath,
        "motion_type": motion_type,
        "scoring_profile": {
            "active_dimensions": list(profile.active_dimensions),
            "excluded_dimensions": profile.excluded_dimensions,
            "excluded_submetrics": {
                k: list(v) for k, v in profile.excluded_submetrics.items()
            },
            "n_dimensions": len(profile.active_dimensions),
        },
        "controller_type": metrics.get("controller_type"),
        "env_type": metrics.get("env_type"),
        "checkpoint_path": metrics.get("checkpoint_path"),
        "dimension_weights": config.get("dimension_weights"),
        "dimensions": dimensions,
        "total_score": round(total, 2),
    }


def save_composite_report(report: dict[str, Any], output_dir: Path | str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "composite_score.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return json_path, output_dir


def run_composite_scoring_on_motion_dir(
    motion_dir: Path | str,
    *,
    config_name: str = "conf_eval_composite_score",
    analysis_subdir: str = "analysis",
    write_radar: bool = True,
) -> dict[str, Any] | None:
    """Score from motion_dir/metrics.json and optional analysis summaries."""
    motion_dir = Path(motion_dir)
    metrics_path = motion_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    analysis = motion_dir / analysis_subdir
    sym = analysis / "section5_symmetry" / "symmetry_metrics_summary.json"
    torq = analysis / "section4_torque" / "torque_metrics_summary.json"
    report = build_composite_report(
        metrics_path,
        config_name=config_name,
        symmetry_summary_path=sym if sym.is_file() else None,
        torque_summary_path=torq if torq.is_file() else None,
    )
    out_dir = analysis if analysis.is_dir() else motion_dir
    save_composite_report(report, out_dir)
    if write_radar:
        from musclemimic.evaluation.radar_plot import plot_composite_radar

        plot_composite_radar(report, out_dir / "composite_score_radar.png")
    return report
