"""Lightweight metric summaries for composite scoring (no section PNG plots)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from musclemimic.distill.config import load_fullbody_config, make_env

LEG_JOINTS = {
    "knee": ("knee_angle_l", "knee_angle_r"),
    "hip": ("hip_flexion_l", "hip_flexion_r"),
    "ankle": ("ankle_angle_l", "ankle_angle_r"),
}

SECTION4_TORQUE = "section4_torque"
SECTION5_SYMMETRY = "section5_symmetry"
TORQUE_SUMMARY_NAME = "torque_metrics_summary.json"
SYMMETRY_SUMMARY_NAME = "symmetry_metrics_summary.json"
TORQUE_SOURCE_MUSCLE_QFRC = "muscle_qfrc"
TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID = "fsm_prosthesis_hybrid"


def _symmetry_index(left: float, right: float) -> float:
    denom = 0.5 * (abs(left) + abs(right))
    return float(abs(left - right) / denom * 100.0) if denom > 1e-6 else 0.0


def _lowpass_torque(tau: np.ndarray, dt: float, cutoff_hz: float = 6.0) -> np.ndarray:
    x = np.asarray(tau, dtype=np.float64).reshape(-1)
    if x.size < 8 or cutoff_hz <= 0:
        return x
    try:
        from scipy.signal import butter, filtfilt

        b, a = butter(
            2,
            min(cutoff_hz, 0.49 / max(dt, 1e-9)),
            fs=1.0 / max(dt, 1e-9),
            btype="low",
        )
        return filtfilt(b, a, x)
    except Exception:
        return x


def _torque_smoothness(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau.size < 2:
        return np.zeros_like(tau)
    return np.abs(np.diff(tau, prepend=tau[0]))


def _torque_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau.size < 3:
        return np.zeros_like(tau)
    j = np.zeros_like(tau)
    j[1:-1] = np.abs(tau[2:] - 2 * tau[1:-1] + tau[:-2])
    j[0] = j[1]
    j[-1] = j[-2]
    return j


def _resolve_leg_dof(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, (jl, jr) in LEG_JOINTS.items():
        out[key] = {}
        for side, jname in (("left", jl), ("right", jr)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"Joint not found: {jname}")
            out[key][side] = int(model.jnt_dofadr[jid])
    return out


def _resolve_leg_qpos(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, (jl, jr) in LEG_JOINTS.items():
        out[key] = {}
        for side, jname in (("left", jl), ("right", jr)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"Joint not found: {jname}")
            out[key][side] = int(model.jnt_qposadr[jid])
    return out


def _joint_torque_summary(tau: np.ndarray, qvel: np.ndarray, dt: float) -> dict[str, float]:
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    tau_f = _lowpass_torque(tau, dt)
    return {
        "torque_smoothness_mean_Nm_per_step": float(np.mean(_torque_smoothness(tau_f))),
        "torque_jerk_mean_Nm": float(np.mean(_torque_jerk(tau_f))),
        "peak_torque_Nm": float(np.max(np.abs(tau))),
    }


def build_torque_summary_from_rollout(
    rollout_npz: Path | str,
    motion_path: str,
    *,
    config_name: str = "conf_fullbody_gmr_resnet",
    torque_source_mode: str = TORQUE_SOURCE_MUSCLE_QFRC,
) -> dict[str, Any]:
    """Torque summary JSON compatible with composite_score (no plots)."""
    data = np.load(Path(rollout_npz), allow_pickle=True)
    if "joint_torque" not in data:
        return {"joints": {}, "torque_source_mode": torque_source_mode}
    time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
    dt = (
        float(np.asarray(data["dt"]).reshape(-1)[0])
        if "dt" in data
        else float(np.median(np.diff(time)) if time.size > 1 else 0.01)
    )
    tau_all = np.asarray(data["joint_torque"], dtype=np.float64)
    qvel_all = np.asarray(data["qvel"], dtype=np.float64)

    cfg = load_fullbody_config(config_name)
    env = make_env(cfg, motion_paths=[motion_path], use_mujoco=True)
    try:
        dof_idx = _resolve_leg_dof(env.model)
    finally:
        env.stop()

    if torque_source_mode == TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID:
        return _build_fsm_prosthesis_hybrid_torque_summary(
            tau_all,
            qvel_all,
            data,
            dof_idx,
            dt,
        )

    joints: dict[str, Any] = {}
    for joint, sides in dof_idx.items():
        block: dict[str, Any] = {}
        for side in ("left", "right"):
            block[side] = _joint_torque_summary(
                tau_all[:, sides[side]],
                qvel_all[:, sides[side]],
                dt,
            )
        joints[joint] = block
    return {
        "torque_source_mode": TORQUE_SOURCE_MUSCLE_QFRC,
        "note": "All six leg channels from MuJoCo qfrc_actuator (muscle actuator torque at hinge DOF).",
        "joints": joints,
    }


def _build_fsm_prosthesis_hybrid_torque_summary(
    tau_all: np.ndarray,
    qvel_all: np.ndarray,
    data: np.lib.npyio.NpzFile,
    dof_idx: dict[str, dict[str, int]],
    dt: float,
) -> dict[str, Any]:
    """Map FSM prosthesis torques onto left knee/ankle; keep other channels on qfrc."""
    if "prosthesis_tau" not in data:
        raise ValueError("fsm_prosthesis_hybrid requires prosthesis_tau in rollout_data.npz")
    prosthesis = np.asarray(data["prosthesis_tau"], dtype=np.float64)
    if prosthesis.ndim != 2 or prosthesis.shape[1] < 2:
        raise ValueError(f"Expected prosthesis_tau shape (T, 4+), got {prosthesis.shape}")

    channel_sources: dict[str, dict[str, str]] = {}
    joints: dict[str, Any] = {}
    for joint, sides in dof_idx.items():
        block: dict[str, Any] = {}
        src_row: dict[str, str] = {}
        for side in ("left", "right"):
            if side == "left" and joint == "knee":
                tau = prosthesis[:, 0]
                src = "prosthesis_tau[knee_angle_l]"
            elif side == "left" and joint == "ankle":
                tau = prosthesis[:, 1]
                src = "prosthesis_tau[ankle_angle_l]"
            else:
                dof = sides[side]
                tau = tau_all[:, dof]
                src = f"qfrc_actuator@{dof}"
            block[side] = _joint_torque_summary(tau, qvel_all[:, sides[side]], dt)
            src_row[side] = src
        joints[joint] = block
        channel_sources[joint] = src_row

    return {
        "torque_source_mode": TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID,
        "note": (
            "FSM hybrid control quality: left knee/ankle use applied prosthesis PD+FSM torques; "
            "left hip and right leg channels remain qfrc_actuator for the same 6-DOF aggregation as official policy."
        ),
        "channel_sources": channel_sources,
        "joints": joints,
    }


def build_symmetry_summary_from_rollout(
    rollout_npz: Path | str,
    motion_path: str,
    *,
    config_name: str = "conf_fullbody_gmr_resnet",
) -> dict[str, Any]:
    """Minimal symmetry_index_pct for ROM submetrics (GRF SI comes from metrics.json)."""
    data = np.load(Path(rollout_npz), allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)

    cfg = load_fullbody_config(config_name)
    env = make_env(cfg, motion_paths=[motion_path], use_mujoco=True)
    try:
        qpos_idx = _resolve_leg_qpos(env.model)
    finally:
        env.stop()

    rom_l: dict[str, float] = {}
    rom_r: dict[str, float] = {}
    for joint, sides in qpos_idx.items():
        rom_l[joint] = float(
            np.rad2deg(np.max(qpos[:, sides["left"]]) - np.min(qpos[:, sides["left"]]))
        )
        rom_r[joint] = float(
            np.rad2deg(np.max(qpos[:, sides["right"]]) - np.min(qpos[:, sides["right"]]))
        )

    si: dict[str, float] = {}
    for joint in rom_l:
        si[f"ROM_{joint}"] = _symmetry_index(rom_l[joint], rom_r[joint])
    return {"symmetry_index_pct": si, "joint_ROM_deg": {"left": rom_l, "right": rom_r}}


def torque_summary_path(analysis_dir: Path | str) -> Path:
    return Path(analysis_dir) / SECTION4_TORQUE / TORQUE_SUMMARY_NAME


def symmetry_summary_path(analysis_dir: Path | str) -> Path:
    return Path(analysis_dir) / SECTION5_SYMMETRY / SYMMETRY_SUMMARY_NAME


def persist_rollout_analysis_summaries(
    analysis_dir: Path | str,
    rollout_npz: Path | str,
    motion_path: str,
    *,
    config_name: str = "conf_fullbody_gmr_resnet",
    include_symmetry: bool = True,
    torque_source_mode: str = TORQUE_SOURCE_MUSCLE_QFRC,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build and write section4/5 JSON summaries expected by composite scoring."""
    analysis_dir = Path(analysis_dir)
    rollout_npz = Path(rollout_npz)

    torque_summary = build_torque_summary_from_rollout(
        rollout_npz,
        motion_path,
        config_name=config_name,
        torque_source_mode=torque_source_mode,
    )
    torq_dir = analysis_dir / SECTION4_TORQUE
    torq_dir.mkdir(parents=True, exist_ok=True)
    with torque_summary_path(analysis_dir).open("w", encoding="utf-8") as f:
        json.dump(torque_summary, f, indent=2, ensure_ascii=False)

    symmetry_summary = None
    if include_symmetry:
        symmetry_summary = build_symmetry_summary_from_rollout(
            rollout_npz, motion_path, config_name=config_name
        )
        sym_dir = analysis_dir / SECTION5_SYMMETRY
        sym_dir.mkdir(parents=True, exist_ok=True)
        with symmetry_summary_path(analysis_dir).open("w", encoding="utf-8") as f:
            json.dump(symmetry_summary, f, indent=2, ensure_ascii=False)

    return torque_summary, symmetry_summary


def load_or_build_rollout_analysis_summaries(
    analysis_dir: Path | str,
    rollout_npz: Path | str,
    motion_path: str,
    *,
    config_name: str = "conf_fullbody_gmr_resnet",
    include_symmetry: bool = True,
    rebuild: bool = False,
    torque_source_mode: str = TORQUE_SOURCE_MUSCLE_QFRC,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load persisted summaries or build them from rollout_data.npz."""
    analysis_dir = Path(analysis_dir)
    rollout_npz = Path(rollout_npz)
    torq_p = torque_summary_path(analysis_dir)
    sym_p = symmetry_summary_path(analysis_dir)

    if (
        not rebuild
        and torq_p.is_file()
        and (not include_symmetry or sym_p.is_file())
    ):
        torque_summary = json.loads(torq_p.read_text(encoding="utf-8"))
        symmetry_summary = (
            json.loads(sym_p.read_text(encoding="utf-8")) if include_symmetry and sym_p.is_file() else None
        )
        return torque_summary, symmetry_summary

    return persist_rollout_analysis_summaries(
        analysis_dir,
        rollout_npz,
        motion_path,
        config_name=config_name,
        include_symmetry=include_symmetry,
        torque_source_mode=torque_source_mode,
    )
