"""Force, contact, and torque metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from musclemimic.evaluation.contact_extractor import analyze_stance_grf_profile
from musclemimic.evaluation.types import MetricValue, RolloutBuffer


def _stance_mask(contact: list[bool]) -> np.ndarray:
    return np.asarray(contact, dtype=bool)


def _symmetry_index(left: float, right: float) -> float | None:
    denom = 0.5 * (abs(left) + abs(right))
    if denom < 1e-9:
        return None
    return float(abs(left - right) / denom * 100.0)


def _series_or_empty(buffer: RolloutBuffer, filtered_key: str, raw_key: str) -> tuple[np.ndarray, np.ndarray]:
    filt = np.asarray(getattr(buffer, filtered_key), dtype=np.float64).reshape(-1)
    raw_attr = getattr(buffer, raw_key, None)
    if raw_attr:
        raw = np.asarray(raw_attr, dtype=np.float64).reshape(-1)
    else:
        raw = filt.copy()
    return raw, filt


def _grf_block(
    raw: np.ndarray,
    filt: np.ndarray,
    contact_mask: np.ndarray,
    dt: float,
    body_weight: float | None,
    side: str,
) -> dict[str, Any]:
    bw = body_weight if body_weight and body_weight > 1e-6 else None
    lc = contact_mask

    def masked_mean(x, mask):
        if not np.any(mask):
            return None
        return float(np.mean(x[mask]))

    def impulse(x, mask):
        if not np.any(mask):
            return None
        return float(np.sum(x[mask]) * dt)

    peak_raw = float(np.max(raw)) if raw.size else 0.0
    peak_filt = float(np.max(filt)) if filt.size else 0.0
    imp_raw = impulse(raw, lc)
    imp_filt = impulse(filt, lc)

    stance = analyze_stance_grf_profile(filt, lc)

    return {
        f"peak_vGRF_{side}_raw": MetricValue(peak_raw, unit="N").to_json(),
        f"peak_vGRF_{side}_filtered": MetricValue(peak_filt, unit="N").to_json(),
        f"peak_vGRF_{side}_raw_norm": MetricValue(peak_raw / bw if bw else None, unit="N/body_weight").to_json(),
        f"peak_vGRF_{side}_filtered_norm": MetricValue(peak_filt / bw if bw else None, unit="N/body_weight").to_json(),
        f"mean_vGRF_{side}_raw": MetricValue(masked_mean(raw, lc), unit="N").to_json(),
        f"mean_vGRF_{side}_filtered": MetricValue(masked_mean(filt, lc), unit="N").to_json(),
        f"GRF_impulse_{side}_raw": MetricValue(imp_raw, unit="N*s").to_json(),
        f"GRF_impulse_{side}_filtered": MetricValue(imp_filt, unit="N*s").to_json(),
        f"stance_peak_count_{side}": MetricValue(stance.get("peak_count"), unit="count", reason=stance.get("reason")).to_json(),
        f"stance_first_peak_pct_{side}": MetricValue(stance.get("first_peak_pct"), unit="% stance").to_json(),
        f"stance_second_peak_pct_{side}": MetricValue(stance.get("second_peak_pct"), unit="% stance").to_json(),
        f"stance_midstance_valley_pct_{side}": MetricValue(stance.get("midstance_valley_pct"), unit="% stance").to_json(),
        f"stance_n_phases_{side}": MetricValue(stance.get("n_stances"), unit="count").to_json(),
    }


def compute_torque_metrics(buffer: RolloutBuffer, meta: dict[str, Any]) -> dict[str, Any]:
    dt = float(buffer.dt)
    tau_key = "prosthesis_tau" if buffer.prosthesis_tau and np.asarray(buffer.prosthesis_tau[0]).size else "joint_torque"
    series = buffer.prosthesis_tau if tau_key == "prosthesis_tau" else buffer.joint_torque
    if not series:
        return {"torque": MetricValue(None, reason="no torque series recorded").to_json()}

    tau = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in series], axis=0)
    qd = None
    if buffer.qvel:
        qd_full = np.stack([np.asarray(q, dtype=np.float64).reshape(-1) for q in buffer.qvel], axis=0)
        n_tau = tau.shape[1]
        qd = qd_full[:, :n_tau] if qd_full.shape[1] >= n_tau else None

    limits = meta.get("torque_limits")
    limit_arr = np.asarray(limits, dtype=np.float64).reshape(-1) if limits is not None else None

    rms = np.sqrt(np.mean(tau**2, axis=0))
    peak = np.max(np.abs(tau), axis=0)
    # Per simulator.md §4.3–4.4: per-step Δτ and Δ²τ in Nm (not scaled by 1/dt).
    smooth = np.mean(np.abs(np.diff(tau, axis=0)), axis=0) if tau.shape[0] > 1 else np.zeros(tau.shape[1])
    jerk = (
        np.mean(np.abs(tau[2:] - 2 * tau[1:-1] + tau[:-2]), axis=0)
        if tau.shape[0] > 2
        else np.zeros(tau.shape[1])
    )

    violation_rate = None
    if limit_arr is not None and limit_arr.size == tau.shape[1]:
        violation_rate = float(np.mean(np.abs(tau) > limit_arr[np.newaxis, :]))

    power = tau * qd if qd is not None else None
    pos_work = neg_work = net_work = None
    power_peak = power_mean_abs = None
    total_pos_pros = total_abs_pros = None
    if power is not None:
        pos = np.maximum(power, 0.0) * dt
        neg = np.minimum(power, 0.0) * dt
        pos_work = np.sum(pos, axis=0)
        neg_work = np.sum(neg, axis=0)
        net_work = np.sum(power * dt, axis=0)
        power_peak = np.max(np.abs(power), axis=0)
        power_mean_abs = np.mean(np.abs(power), axis=0)
        if tau_key == "prosthesis_tau":
            total_pos_pros = float(np.sum(pos))
            total_abs_pros = float(np.sum(np.abs(power) * dt))

    return {
        "torque_source": MetricValue(
            "qfrc_actuator" if tau_key == "joint_torque" else "prosthesis_tau",
            unit="str",
        ).to_json(),
        "torque_RMS_per_joint": MetricValue(rms.tolist(), unit="Nm").to_json(),
        "peak_torque_per_joint": MetricValue(peak.tolist(), unit="Nm").to_json(),
        "torque_smoothness_per_joint": MetricValue(smooth.tolist(), unit="Nm/step").to_json(),
        "torque_jerk_per_joint": MetricValue(jerk.tolist(), unit="Nm").to_json(),
        "torque_limit_violation_rate": MetricValue(violation_rate, unit="ratio", reason=None if violation_rate is not None else "no torque_limits").to_json(),
        "mechanical_power_peak_per_joint": MetricValue(
            power_peak.tolist() if power_peak is not None else None,
            unit="W",
            reason=None if power is not None else "missing qvel",
        ).to_json(),
        "mechanical_power_mean_abs_per_joint": MetricValue(
            power_mean_abs.tolist() if power_mean_abs is not None else None,
            unit="W",
            reason=None if power is not None else "missing qvel",
        ).to_json(),
        "positive_work_per_joint": MetricValue(pos_work.tolist() if pos_work is not None else None, unit="J").to_json(),
        "negative_work_per_joint": MetricValue(neg_work.tolist() if neg_work is not None else None, unit="J").to_json(),
        "net_work_per_joint": MetricValue(net_work.tolist() if net_work is not None else None, unit="J").to_json(),
        "total_positive_work_prosthesis": MetricValue(total_pos_pros, unit="J").to_json(),
        "total_abs_work_prosthesis": MetricValue(total_abs_pros, unit="J").to_json(),
    }


def compute_grf_metrics(buffer: RolloutBuffer, meta: dict[str, Any]) -> dict[str, Any]:
    if not buffer.left_grf:
        return {"grf": MetricValue(None, reason="no GRF recorded").to_json()}

    dt = float(buffer.dt)
    raw_l, filt_l = _series_or_empty(buffer, "left_grf", "left_grf_raw")
    raw_r, filt_r = _series_or_empty(buffer, "right_grf", "right_grf_raw")
    lc = _stance_mask(buffer.contact_left)
    rc = _stance_mask(buffer.contact_right)
    body_weight = float(meta.get("body_weight_N", 0.0))
    bw = body_weight if body_weight > 1e-6 else None

    out: dict[str, Any] = {}
    out.update(_grf_block(raw_l, filt_l, lc, dt, bw, "left"))
    out.update(_grf_block(raw_r, filt_r, rc, dt, bw, "right"))

    peak_l = float(np.max(filt_l))
    peak_r = float(np.max(filt_r))
    imp_l = float(np.sum(filt_l[lc]) * dt) if np.any(lc) else None
    imp_r = float(np.sum(filt_r[rc]) * dt) if np.any(rc) else None

    switch_count = int(np.sum(np.diff(lc.astype(int)) != 0) + np.sum(np.diff(rc.astype(int)) != 0))

    out.update(
        {
            # Backward-compatible aliases (filtered)
            "peak_vGRF_left": MetricValue(peak_l, unit="N").to_json(),
            "peak_vGRF_right": MetricValue(peak_r, unit="N").to_json(),
            "GRF_symmetry_index_peak_filtered": MetricValue(_symmetry_index(peak_l, peak_r), unit="%").to_json(),
            "GRF_symmetry_index_impulse_filtered": MetricValue(
                _symmetry_index(imp_l or 0.0, imp_r or 0.0) if imp_l is not None and imp_r is not None else None,
                unit="%",
            ).to_json(),
            "contact_duration_left": MetricValue(float(np.sum(lc) * dt), unit="s").to_json(),
            "contact_duration_right": MetricValue(float(np.sum(rc) * dt), unit="s").to_json(),
            "contact_switch_count": MetricValue(switch_count, unit="count").to_json(),
            "foot_slip_left": MetricValue(None, unit="m/s", reason="v2 TODO").to_json(),
            "foot_slip_right": MetricValue(None, unit="m/s", reason="v2 TODO").to_json(),
        }
    )
    return out


def compute_reference_force_metrics(buffer: RolloutBuffer, ref_path: str | None) -> dict[str, Any]:
    if not ref_path:
        return {}
    try:
        data = np.load(ref_path, allow_pickle=True)
    except Exception as exc:
        return {"reference_force": MetricValue(None, reason=str(exc)).to_json()}

    out: dict[str, Any] = {}
    if "left_vertical_GRF" in data and buffer.left_grf:
        ref_l = np.asarray(data["left_vertical_GRF"]).reshape(-1)
        cur_l = np.asarray(buffer.left_grf).reshape(-1)
        n = min(ref_l.size, cur_l.size)
        if n > 0:
            err = ref_l[:n] - cur_l[:n]
            out["vGRF_RMSE_vs_reference_left"] = MetricValue(float(np.sqrt(np.mean(err**2))), unit="N").to_json()
            if np.std(ref_l[:n]) > 1e-6 and np.std(cur_l[:n]) > 1e-6:
                out["vGRF_correlation_vs_reference_left"] = MetricValue(float(np.corrcoef(ref_l[:n], cur_l[:n])[0, 1]), unit="corr").to_json()
    if "target_prosthesis_tau" in data and buffer.prosthesis_tau:
        ref_t = np.asarray(data["target_prosthesis_tau"])[: min(len(buffer.prosthesis_tau), len(data["target_prosthesis_tau"]))]
        cur_t = np.stack([np.asarray(x) for x in buffer.prosthesis_tau[: ref_t.shape[0]]], axis=0)
        err = ref_t - cur_t
        out["torque_RMSE_vs_reference"] = MetricValue(float(np.sqrt(np.mean(err**2))), unit="Nm").to_json()
    return out
