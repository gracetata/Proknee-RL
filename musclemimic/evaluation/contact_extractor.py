"""Foot contact force extraction from MuJoCo contacts (world-frame GRF)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# Default sole geoms — must be overridden explicitly in config for other models.
DEFAULT_LEFT_FOOT_GEOMS = ("l_bofoot",)
DEFAULT_RIGHT_FOOT_GEOMS = ("r_bofoot",)

FLOOR_NAME_TOKENS = ("floor", "ground", "plane")


@dataclass
class FootContactConfig:
    left_foot_geoms: tuple[str, ...]
    right_foot_geoms: tuple[str, ...]
    contact_threshold: float = 1.0
    grf_filter: str = "moving_average"  # none | moving_average | lowpass
    grf_filter_window: int = 5
    grf_lowpass_cutoff_hz: float = 12.0
    sanity_weight_tolerance: float = 0.20
    print_geom_contributions: bool = False
    warn_on_missing_contacts: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "FootContactConfig":
        left = cfg.get("left_foot_geoms")
        right = cfg.get("right_foot_geoms")
        if not left:
            left = DEFAULT_LEFT_FOOT_GEOMS
            logger.info("eval.force.left_foot_geoms not set; using default sole geoms %s", left)
        if not right:
            right = DEFAULT_RIGHT_FOOT_GEOMS
            logger.info("eval.force.right_foot_geoms not set; using default sole geoms %s", right)
        return cls(
            left_foot_geoms=tuple(str(x) for x in left),
            right_foot_geoms=tuple(str(x) for x in right),
            contact_threshold=float(cfg.get("contact_threshold", 1.0)),
            grf_filter=str(cfg.get("grf_filter", "moving_average")),
            grf_filter_window=int(cfg.get("grf_filter_window", 5)),
            grf_lowpass_cutoff_hz=float(cfg.get("grf_lowpass_cutoff_hz", 12.0)),
            sanity_weight_tolerance=float(cfg.get("sanity_weight_tolerance", 0.20)),
            print_geom_contributions=bool(cfg.get("print_geom_contributions", False)),
            warn_on_missing_contacts=bool(cfg.get("warn_on_missing_contacts", True)),
        )


def list_foot_related_geoms(model: mujoco.MjModel) -> list[str]:
    names = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        low = name.lower()
        if any(tok in low for tok in ("foot", "toe", "bofoot", "heel")):
            names.append(name)
    return sorted(names)


def resolve_geom_ids(model: mujoco.MjModel, names: tuple[str, ...], side: str) -> dict[str, int]:
    if not names:
        raise ValueError(
            f"eval.force.{side}_foot_geoms is empty. "
            f"Explicitly list sole collision geoms in config (do not auto-collect all foot geoms). "
            f"Foot-related geoms in model: {list_foot_related_geoms(model)}"
        )
    mapping: dict[str, int] = {}
    missing = []
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            missing.append(name)
        else:
            mapping[name] = int(gid)
    if missing or not mapping:
        raise ValueError(
            f"Could not resolve {side} foot geoms {names}. Missing={missing}. "
            f"Foot-related geoms in model: {list_foot_related_geoms(model)}"
        )
    return mapping


def _is_floor_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
    if any(tok in name for tok in FLOOR_NAME_TOKENS):
        return True
    body_id = int(model.geom_bodyid[geom_id])
    body_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").lower()
    return body_id == 0 or "world" in body_name or "floor" in body_name


def contact_force_to_world(force_contact: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Map 3D force in contact frame to world frame.

    MuJoCo stores `frame` row-major; each row is a contact basis vector in world coords.
    """
    R = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    fc = np.asarray(force_contact, dtype=np.float64).reshape(3)
    return fc[0] * R[0] + fc[1] * R[1] + fc[2] * R[2]


def force_on_foot_geom(
    model: mujoco.MjModel,
    *,
    foot_gid: int,
    geom1: int,
    geom2: int,
    force_contact: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Force applied TO the foot geom, in world frame (positive z = upward support)."""
    f_world_geom1 = contact_force_to_world(force_contact, frame)
    if foot_gid == geom1:
        f_on_foot = f_world_geom1
    elif foot_gid == geom2:
        f_on_foot = -f_world_geom1
    else:
        return np.zeros(3, dtype=np.float64)

    other = geom2 if foot_gid == geom1 else geom1
    if _is_floor_geom(model, other):
        # Ground reaction on foot should push foot upward (+z).
        if f_on_foot[2] < 0.0:
            f_on_foot = -f_on_foot
    return f_on_foot


def list_foot_contact_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_geom_ids: set[int],
) -> list[dict[str, Any]]:
    pairs = []
    for cid in range(int(data.ncon)):
        c = data.contact[cid]
        g1, g2 = int(c.geom1), int(c.geom2)
        foot_gid = None
        if g1 in foot_geom_ids:
            foot_gid = g1
        elif g2 in foot_geom_ids:
            foot_gid = g2
        if foot_gid is None:
            continue
        other = g2 if foot_gid == g1 else g1
        f6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, cid, f6)
        f_on_foot = force_on_foot_geom(
            model,
            foot_gid=foot_gid,
            geom1=g1,
            geom2=g2,
            force_contact=f6[:3],
            frame=np.asarray(c.frame),
        )
        pairs.append(
            {
                "contact_id": cid,
                "foot_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, foot_gid),
                "other_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                "foot_is_geom1": foot_gid == g1,
                "fz_world": float(f_on_foot[2]),
                "f_world": f_on_foot.tolist(),
            }
        )
    return pairs


class GRFFilter:
    """Causal per-foot vertical GRF filter (applied during rollout)."""

    def __init__(self, config: FootContactConfig, dt: float):
        self.config = config
        self.dt = float(dt)
        self._left_buf: list[float] = []
        self._right_buf: list[float] = []

    def reset(self) -> None:
        self._left_buf.clear()
        self._right_buf.clear()

    def _moving_average(self, buf: list[float], value: float) -> float:
        w = max(1, int(self.config.grf_filter_window))
        buf.append(float(value))
        if len(buf) > w:
            del buf[0 : len(buf) - w]
        return float(np.mean(buf))

    def _lowpass(self, buf: list[float], value: float) -> float:
        # First-order low-pass (exponential smoothing); cutoff via alpha.
        cutoff = max(0.1, float(self.config.grf_lowpass_cutoff_hz))
        rc = 1.0 / (2.0 * np.pi * cutoff)
        alpha = self.dt / (rc + self.dt)
        prev = buf[-1] if buf else float(value)
        out = alpha * float(value) + (1.0 - alpha) * prev
        buf.append(out)
        if len(buf) > max(50, int(self.config.grf_filter_window) * 4):
            del buf[0]
        return float(out)

    def apply(self, side: str, raw_v: float) -> float:
        mode = str(self.config.grf_filter).lower()
        if mode in {"", "none", "off"}:
            return float(raw_v)
        buf = self._left_buf if side == "left" else self._right_buf
        if mode == "lowpass":
            return self._lowpass(buf, raw_v)
        return self._moving_average(buf, raw_v)


class FootContactExtractor:
    def __init__(self, model: mujoco.MjModel, config: FootContactConfig, *, dt: float = 0.01):
        self.model = model
        self.config = config
        self.dt = float(dt)
        self.body_weight_N = float(np.sum(model.body_mass) * 9.81)
        self.left_geom_map = resolve_geom_ids(model, config.left_foot_geoms, "left")
        self.right_geom_map = resolve_geom_ids(model, config.right_foot_geoms, "right")
        self.left_geom_ids = set(self.left_geom_map.values())
        self.right_geom_ids = set(self.right_geom_map.values())
        overlap = self.left_geom_ids & self.right_geom_ids
        if overlap:
            raise ValueError(f"Left/right foot geom sets overlap: {overlap}")
        self._force6 = np.zeros(6, dtype=np.float64)
        self._filter = GRFFilter(config, dt)
        self._sanity_done = False
        self._warned_left = self._warned_right = False

    def reset_filters(self) -> None:
        self._filter.reset()
        self._sanity_done = False

    def _aggregate_side(
        self,
        data: mujoco.MjData,
        geom_map: dict[str, int],
        side: str,
    ) -> tuple[np.ndarray, float, dict[str, float], list[dict[str, Any]]]:
        f_world = np.zeros(3, dtype=np.float64)
        per_geom: dict[str, float] = {name: 0.0 for name in geom_map}
        pairs: list[dict[str, Any]] = []
        foot_ids = set(geom_map.values())

        for cid in range(int(data.ncon)):
            contact = data.contact[cid]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            foot_gid = None
            if g1 in foot_ids:
                foot_gid = g1
            elif g2 in foot_ids:
                foot_gid = g2
            else:
                continue

            mujoco.mj_contactForce(self.model, data, cid, self._force6)
            f_on_foot = force_on_foot_geom(
                self.model,
                foot_gid=foot_gid,
                geom1=g1,
                geom2=g2,
                force_contact=self._force6[:3],
                frame=np.asarray(contact.frame),
            )
            f_world += f_on_foot
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_gid) or str(foot_gid)
            per_geom[gname] = per_geom.get(gname, 0.0) + float(f_on_foot[2])

            if self.config.print_geom_contributions:
                other = g2 if foot_gid == g1 else g1
                pairs.append(
                    {
                        "side": side,
                        "foot_geom": gname,
                        "other_geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                        "fz": float(f_on_foot[2]),
                    }
                )

        raw_v = max(float(f_world[2]), 0.0)
        return f_world, raw_v, per_geom, pairs

    def _maybe_warn_no_contacts(self, data: mujoco.MjData, side: str, raw_v: float, geom_map: dict[str, int]) -> None:
        if raw_v > self.config.contact_threshold:
            return
        warned = self._warned_left if side == "left" else self._warned_right
        if warned or not self.config.warn_on_missing_contacts:
            return
        all_foot = self.left_geom_ids | self.right_geom_ids
        active = list_foot_contact_pairs(self.model, data, all_foot)
        msg = (
            f"[GRF] No vertical force on configured {side} foot geoms {tuple(geom_map.keys())}. "
            f"Active foot-related contact pairs in scene: {active[:12]}"
        )
        warnings.warn(msg)
        logger.warning(msg)
        if side == "left":
            self._warned_left = True
        else:
            self._warned_right = True

    def sanity_check_support(self, raw_left: float, raw_right: float) -> None:
        if self._sanity_done:
            return
        total = float(raw_left) + float(raw_right)
        if total < 0.5 * self.body_weight_N:
            return
        err = abs(total - self.body_weight_N) / max(self.body_weight_N, 1e-6)
        if err > float(self.config.sanity_weight_tolerance):
            warnings.warn(
                f"[GRF sanity] left_vGRF + right_vGRF = {total:.1f} N vs body_weight "
                f"{self.body_weight_N:.1f} N (rel err {100*err:.1f}% > "
                f"{100*self.config.sanity_weight_tolerance:.0f}% tolerance). "
                "Check foot geom mapping / sign convention."
            )
        self._sanity_done = True

    def extract(self, data: mujoco.MjData) -> dict[str, Any]:
        left_world, raw_l, per_geom_l, pairs_l = self._aggregate_side(data, self.left_geom_map, "left")
        right_world, raw_r, per_geom_r, pairs_r = self._aggregate_side(data, self.right_geom_map, "right")

        self._maybe_warn_no_contacts(data, "left", raw_l, self.left_geom_map)
        self._maybe_warn_no_contacts(data, "right", raw_r, self.right_geom_map)
        self.sanity_check_support(raw_l, raw_r)

        filt_l = max(self._filter.apply("left", raw_l), 0.0)
        filt_r = max(self._filter.apply("right", raw_r), 0.0)
        thr = float(self.config.contact_threshold)

        out: dict[str, Any] = {
            "left_foot_GRF_world": left_world.astype(np.float32),
            "right_foot_GRF_world": right_world.astype(np.float32),
            "left_vertical_GRF_raw": np.asarray(raw_l, dtype=np.float32),
            "right_vertical_GRF_raw": np.asarray(raw_r, dtype=np.float32),
            "left_vertical_GRF": np.asarray(filt_l, dtype=np.float32),
            "right_vertical_GRF": np.asarray(filt_r, dtype=np.float32),
            "left_horizontal_GRF_norm": np.asarray(float(np.linalg.norm(left_world[:2])), dtype=np.float32),
            "right_horizontal_GRF_norm": np.asarray(float(np.linalg.norm(right_world[:2])), dtype=np.float32),
            "left_contact_bool": np.asarray(filt_l > thr, dtype=np.bool_),
            "right_contact_bool": np.asarray(filt_r > thr, dtype=np.bool_),
            "left_per_geom_vGRF": per_geom_l,
            "right_per_geom_vGRF": per_geom_r,
        }
        if self.config.print_geom_contributions and (pairs_l or pairs_r):
            logger.info("GRF contact pairs L=%s R=%s", pairs_l[:5], pairs_r[:5])
        return out


def segment_stance_phases(contact_mask: np.ndarray, min_steps: int = 5) -> list[tuple[int, int]]:
    mask = np.asarray(contact_mask, dtype=bool).reshape(-1)
    phases: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_steps:
                phases.append((start, i))
            start = None
    if start is not None and mask.size - start >= min_steps:
        phases.append((start, mask.size))
    return phases


def resample_stance_curve(values: np.ndarray, n_points: int = 101) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return np.full(n_points, float(x[0]) if x.size else 0.0)
    src = np.linspace(0.0, 100.0, num=x.size)
    dst = np.linspace(0.0, 100.0, num=n_points)
    return np.interp(dst, src, x)


def detect_peaks_on_curve(curve: np.ndarray, min_prominence: float = 0.05) -> list[int]:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    if y.size < 5:
        return []
    peaks = []
    ymax = float(np.max(y)) if y.size else 0.0
    prom = max(float(min_prominence) * ymax, 1e-3)
    for i in range(1, y.size - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] >= prom:
            peaks.append(i)
    return peaks


def analyze_stance_grf_profile(
    vgrf: np.ndarray,
    contact_mask: np.ndarray,
    *,
    min_stance_steps: int = 5,
    resample_points: int = 101,
) -> dict[str, Any]:
    """Average stance vGRF curves (0–100% stance) and detect peak structure."""
    v = np.maximum(np.asarray(vgrf, dtype=np.float64).reshape(-1), 0.0)
    phases = segment_stance_phases(np.asarray(contact_mask, dtype=bool), min_steps=min_stance_steps)
    if not phases:
        return {
            "n_stances": 0,
            "peak_count": None,
            "first_peak_pct": None,
            "second_peak_pct": None,
            "midstance_valley_pct": None,
            "reason": "no stance phases detected",
        }

    curves = [resample_stance_curve(v[s:e]) for s, e in phases]
    mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
    peaks = detect_peaks_on_curve(mean_curve)
    peak_pct = [100.0 * p / max(resample_points - 1, 1) for p in peaks]

    first_peak = peak_pct[0] if peaks else None
    second_peak = peak_pct[1] if len(peaks) > 1 else None
    midstance_valley = None
    if len(peaks) >= 2:
        i0, i1 = peaks[0], peaks[1]
        seg = mean_curve[i0 : i1 + 1]
        if seg.size >= 2:
            midstance_valley = 100.0 * (i0 + int(np.argmin(seg))) / max(resample_points - 1, 1)

    return {
        "n_stances": len(phases),
        "peak_count": len(peaks),
        "first_peak_pct": first_peak,
        "second_peak_pct": second_peak,
        "midstance_valley_pct": midstance_valley,
        "mean_stance_curve_pct": mean_curve.tolist(),
        "reason": None,
    }
