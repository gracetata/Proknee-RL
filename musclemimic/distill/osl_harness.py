"""Shared helpers for masked-obs policy + OSL FSM deployment and diagnostics."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np

from musclemimic.distill.osl_deploy_defaults import (
    DEFAULT_OSL_ANKLE_TORQUE_LIMIT,
    DEFAULT_OSL_CLIP_LOADCELL_TO_BODY_WEIGHT,
    DEFAULT_OSL_FOOT_TORQUE_LIMIT,
    DEFAULT_OSL_KNEE_TORQUE_LIMIT,
    DEFAULT_OSL_LOADCELL_SCALE,
    DEFAULT_OSL_MASK_PRESET,
    DEFAULT_OSL_TORQUE_SLEW_LIMIT,
)
from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS

REPO_ROOT = Path(__file__).resolve().parents[2]
OSL_DIR = REPO_ROOT.parent / "musclebenchmark" / "opensourceleg_fsm"
if str(OSL_DIR) not in sys.path:
    sys.path.insert(0, str(OSL_DIR))

from official_fsm import OpenSourceLegFSMController  # noqa: E402
from sim_adapter import (  # noqa: E402
    DEFAULT_MUJOCO_LOADCELL_SCALE,
    MuscleMimicOSLAdapter,
    OSLAdapterConfig,
    model_body_weight_n,
)

# Import after repo root is on path for scripts.
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from record_stage1_testset_visual import IndexedValues, step_with_stage1_pd  # noqa: E402


@dataclass
class OSLHarness:
    env: Any
    audit: Any
    disabled_actuators: np.ndarray
    adapter: MuscleMimicOSLAdapter
    controller: OpenSourceLegFSMController
    zero_kp: np.ndarray
    zero_kd: np.ndarray
    torque_limit: np.ndarray
    torque_slew: np.ndarray
    last_torque: np.ndarray
    body_weight_n: float

    def reset(self) -> None:
        self.adapter.reset_hold_targets(self.env.data)
        self.controller.reset()
        self.last_torque[:] = 0.0

    def step(
        self,
        policy_action: np.ndarray,
        *,
        disabled_muscle_scale: float,
        use_osl_torque: bool = True,
    ) -> tuple[np.ndarray, float, bool, dict]:
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        ff_torque = None
        diag: dict[str, Any] = {}
        if use_osl_torque:
            osl_inputs = self.adapter.read_inputs(self.env.model, self.env.data)
            command = self.controller.update(osl_inputs)
            _target_qpos, ff_torque, fsm_diag = self.adapter.command_to_targets(
                self.env.model, self.env.data, command
            )
            diag = {
                "state": fsm_diag.state,
                "loadcell_fz_n": float(osl_inputs.loadcell_fz_n),
                "knee_torque": float(fsm_diag.knee_torque),
                "ankle_torque": float(fsm_diag.ankle_torque),
                "subtalar_torque": float(fsm_diag.subtalar_torque),
                "mtp_torque": float(fsm_diag.mtp_torque),
            }
        current_q = np.asarray(self.env.data.qpos[self.audit.qpos_indices], dtype=np.float64)
        obs, reward, _absorbing, done, _info = step_with_stage1_pd(
            self.env,
            policy_action,
            IndexedValues(self.audit.qpos_indices, current_q),
            self.audit.qvel_indices,
            self.disabled_actuators,
            float(disabled_muscle_scale),
            self.zero_kp,
            self.zero_kd,
            self.torque_limit,
            self.torque_slew,
            self.last_torque,
            ff_torque,
        )
        diag["root_height"] = float(self.env.data.qpos[2])
        diag["disabled_ctrl_norm"] = (
            float(np.linalg.norm(self.env.data.ctrl[self.disabled_actuators]))
            if self.disabled_actuators.size
            else 0.0
        )
        if use_osl_torque and ff_torque is not None:
            diag["prosthesis_torque"] = np.asarray(ff_torque, dtype=np.float32)
        return np.asarray(obs), float(np.asarray(reward).item()), bool(done), diag


def actuator_ids_for_names(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids: list[int] = []
    missing: list[str] = []
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            missing.append(name)
        else:
            ids.append(int(aid))
    if missing:
        raise KeyError(f"Missing disabled muscle actuators: {missing}")
    return np.asarray(sorted(set(ids)), dtype=np.int32)


def build_osl_harness(
    env,
    *,
    knee_torque_limit: float = DEFAULT_OSL_KNEE_TORQUE_LIMIT,
    ankle_torque_limit: float = DEFAULT_OSL_ANKLE_TORQUE_LIMIT,
    foot_torque_limit: float = DEFAULT_OSL_FOOT_TORQUE_LIMIT,
    torque_slew_limit: float = DEFAULT_OSL_TORQUE_SLEW_LIMIT,
    loadcell_scale: float = DEFAULT_OSL_LOADCELL_SCALE,
    body_weight_n: float | None = None,
    mask_preset: str = DEFAULT_OSL_MASK_PRESET,
    clip_loadcell_to_body_weight: bool = DEFAULT_OSL_CLIP_LOADCELL_TO_BODY_WEIGHT,
    loadcell_clip_min_n: float | None = None,
    loadcell_clip_max_n: float = 0.0,
) -> OSLHarness:
    audit = audit_myofullbody_left_leg(env.model)
    if mask_preset not in MUSCLE_MASK_PRESETS:
        raise ValueError(f"Unknown OSL mask preset: {mask_preset}")
    disabled_actuators = actuator_ids_for_names(env.model, MUSCLE_MASK_PRESETS[mask_preset])
    weight = float(body_weight_n) if body_weight_n is not None else model_body_weight_n(env.model)
    adapter = MuscleMimicOSLAdapter(
        env.model,
        audit,
        OSLAdapterConfig(
            knee_torque_limit=knee_torque_limit,
            ankle_torque_limit=ankle_torque_limit,
            foot_torque_limit=foot_torque_limit,
            loadcell_scale=loadcell_scale,
            clip_loadcell_to_body_weight=clip_loadcell_to_body_weight,
            loadcell_clip_min_n=loadcell_clip_min_n,
            loadcell_clip_max_n=loadcell_clip_max_n,
        ),
        body_weight_n=weight,
    )
    controller = OpenSourceLegFSMController(body_weight_n=weight)
    zero_kp = np.zeros(4, dtype=np.float64)
    zero_kd = np.zeros(4, dtype=np.float64)
    torque_limit = np.asarray(
        [knee_torque_limit, ankle_torque_limit, foot_torque_limit, foot_torque_limit],
        dtype=np.float64,
    )
    torque_slew = np.full(4, float(torque_slew_limit), dtype=np.float64)
    return OSLHarness(
        env=env,
        audit=audit,
        disabled_actuators=disabled_actuators,
        adapter=adapter,
        controller=controller,
        zero_kp=zero_kp,
        zero_kd=zero_kd,
        torque_limit=torque_limit,
        torque_slew=torque_slew,
        last_torque=np.zeros(4, dtype=np.float64),
        body_weight_n=weight,
    )


def scheduled_muscle_scale(
    step: int,
    *,
    end_scale: float,
    ramp_steps: int = 0,
    start_scale: float = 1.0,
) -> float:
    """Linear ramp from start_scale to end_scale over ramp_steps; then hold end_scale."""
    if ramp_steps <= 0:
        return float(end_scale)
    if step >= ramp_steps:
        return float(end_scale)
    t = float(step) / float(ramp_steps)
    return float(start_scale + (end_scale - start_scale) * t)


def compare_mask_specs(checkpoint_spec, env_spec) -> dict:
    return {
        "checkpoint_masked_obs_dim": int(checkpoint_spec.masked_obs_dim),
        "env_masked_obs_dim": int(env_spec.masked_obs_dim),
        "dims_match": int(checkpoint_spec.masked_obs_dim) == int(env_spec.masked_obs_dim),
        "keep_indices_match": tuple(checkpoint_spec.keep_indices) == tuple(env_spec.keep_indices),
        "removed_obs_count_checkpoint": len(checkpoint_spec.removed_obs_names),
        "removed_obs_count_env": len(env_spec.removed_obs_names),
    }


def summarize_run_stats(run_stats) -> dict:
    """Flatten RunningMeanStd leaves for quick NaN/extreme checks."""

    def walk(path: tuple[str, ...], node) -> list[dict]:
        rows: list[dict] = []
        if isinstance(node, dict):
            for key, value in node.items():
                rows.extend(walk(path + (str(key),), value))
            return rows
        if hasattr(node, "shape"):
            arr = np.asarray(node)
            rows.append(
                {
                    "path": ".".join(path),
                    "shape": list(arr.shape),
                    "mean_abs": float(np.mean(np.abs(arr))) if arr.size else 0.0,
                    "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
                    "has_nan": bool(np.isnan(arr).any()) if arr.size else False,
                }
            )
        return rows

    rows = walk(tuple(), run_stats)
    return {
        "num_leaves": len(rows),
        "has_nan": any(r["has_nan"] for r in rows),
        "max_abs_overall": max((r["max_abs"] for r in rows), default=0.0),
        "leaves": rows[:30],
    }


def summarize_trajectory(rows: list[dict]) -> dict:
    if not rows:
        return {"steps": 0}
    root_heights = [float(r.get("root_height", 0.0)) for r in rows]
    first_fall_step = None
    for idx, height in enumerate(root_heights, start=1):
        if height < 0.8:
            first_fall_step = idx
            break
    done_steps = [int(bool(r.get("done", False))) for r in rows]
    first_done_step = next((i + 1 for i, d in enumerate(done_steps) if d), None)
    return {
        "steps": len(rows),
        "root_height_min": float(min(root_heights)),
        "root_height_end": float(root_heights[-1]),
        "first_fall_step_lt_0.8m": first_fall_step,
        "first_done_step": first_done_step,
        "done_count": int(sum(done_steps)),
        "disabled_ctrl_max": float(max(float(r.get("disabled_ctrl_norm", 0.0)) for r in rows)),
    }
