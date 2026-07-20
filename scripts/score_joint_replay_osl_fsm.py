#!/usr/bin/env python
"""Run OSL FSM hybrid replay (no video) and compute composite locomotion score."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
OSL_DIR = WORKSPACE_ROOT / "musclebenchmark" / "opensourceleg_fsm"
for path in (str(REPO_ROOT), str(OSL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from official_fsm import OpenSourceLegFSMController  # noqa: E402
from sim_adapter import MuscleMimicOSLAdapter, OSLAdapterConfig, model_body_weight_n  # noqa: E402

from musclemimic.evaluation.adapters import _get_reference
from musclemimic.evaluation.composite_score import (
    build_composite_report,
    save_composite_report,
)
from musclemimic.evaluation.light_summaries import (
    TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID,
    persist_rollout_analysis_summaries,
)
from musclemimic.evaluation.contact_extractor import FootContactConfig, FootContactExtractor
from musclemimic.evaluation.joint_replay import load_joint_trajectory
from musclemimic.evaluation.logger import save_metrics_json, save_rollout_npz
from musclemimic.evaluation.metrics import compute_all_metrics
from musclemimic.evaluation.motion_taxonomy import classify_motion_type
from musclemimic.evaluation.radar_plot import plot_composite_radar
from musclemimic.evaluation.rollout_loop import infer_done_reason
from musclemimic.evaluation.runner import default_eval_force_from_hydra
from musclemimic.evaluation.types import RolloutBuffer
from musclemimic.distill.osl_deploy_defaults import (
    DEFAULT_OSL_ANKLE_TORQUE_LIMIT,
    DEFAULT_OSL_CLIP_LOADCELL_TO_BODY_WEIGHT,
    DEFAULT_OSL_FOOT_TORQUE_LIMIT,
    DEFAULT_OSL_KNEE_TORQUE_LIMIT,
    DEFAULT_OSL_LOADCELL_SCALE,
    DEFAULT_OSL_MASK_PRESET,
    DEFAULT_OSL_PROSTHESIS_MUSCLE_SCALE,
    DEFAULT_OSL_TORQUE_SLEW_LIMIT,
)
from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.distill.osl_harness import actuator_ids_for_names
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS
from record_joint_replay_osl_fsm import (  # noqa: E402
    build_env,
    pin_non_left_joints,
)
from record_stage1_testset_visual import (  # noqa: E402
    IndexedValues,
    step_with_stage1_pd,
)

DEFAULT_REPLAY_BASE = REPO_ROOT / "outputs/replay/joint_mimic/official_mm10m2"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "data/checkpoints/mm-10m-2"
DEFAULT_OFFICIAL_EVAL_BASE = REPO_ROOT / "outputs/eval/runs/official_mm10m2/6d_symmetric_gait"

FOUR_REPLAY_MOTIONS = [
    "KIT_3_walk_6m_straight_line04_poses",
    "KIT_9_WalkingStraightForwards07_poses",
    "KIT_359_walking_run04_poses",
    "KIT_425_walking_slow07_poses",
]

DIM_ORDER = [
    "task_reliability",
    "pose_tracking",
    "global_trajectory",
    "contact_biomechanics",
    "bilateral_symmetry",
    "control_quality",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-trajectory", default=None, help="Path to joint_trajectory.npz")
    p.add_argument("--replay-dir", default=None, help="Replay motion directory (contains joint_trajectory.npz)")
    p.add_argument("--all-four", action="store_true", help="Score all four recorded FSM replay motions")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet", help="Env config for leg DOF mapping in analysis summaries.")
    p.add_argument("--skip-rollout", action="store_true", help="Reuse analysis/rollout_data.npz; only rebuild summaries + composite score.")
    p.add_argument("--prosthesis-muscle-scale", type=float, default=DEFAULT_OSL_PROSTHESIS_MUSCLE_SCALE)
    p.add_argument("--knee-torque-limit", type=float, default=DEFAULT_OSL_KNEE_TORQUE_LIMIT)
    p.add_argument("--ankle-torque-limit", type=float, default=DEFAULT_OSL_ANKLE_TORQUE_LIMIT)
    p.add_argument("--foot-torque-limit", type=float, default=DEFAULT_OSL_FOOT_TORQUE_LIMIT)
    p.add_argument("--loadcell-scale", type=float, default=DEFAULT_OSL_LOADCELL_SCALE)
    p.add_argument("--body-weight-n", type=float, default=None)
    p.add_argument("--torque-slew-limit", type=float, default=DEFAULT_OSL_TORQUE_SLEW_LIMIT)
    p.add_argument("--max-steps", type=int, default=0, help="0 = all frames in npz")
    p.add_argument("--write-comparison-table", action="store_true", help="Write fsm_replay_scores.md at workspace root")
    p.add_argument(
        "--comparison-table-path",
        default=str(WORKSPACE_ROOT / "fsm_replay_scores.md"),
    )
    return p.parse_args()


def _get_site_pos(env, model, data) -> np.ndarray | None:
    sites = getattr(env, "sites_for_mimic", []) or []
    if not sites:
        return None
    out = []
    for name in sites:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            out.append(np.asarray(data.site_xpos[sid], dtype=np.float32))
    return np.stack(out, axis=0) if out else None


def run_fsm_hybrid_rollout(
    traj_path: Path,
    *,
    checkpoint: str,
    max_steps: int = 0,
    prosthesis_muscle_scale: float = DEFAULT_OSL_PROSTHESIS_MUSCLE_SCALE,
    knee_torque_limit: float = DEFAULT_OSL_KNEE_TORQUE_LIMIT,
    ankle_torque_limit: float = DEFAULT_OSL_ANKLE_TORQUE_LIMIT,
    foot_torque_limit: float = DEFAULT_OSL_FOOT_TORQUE_LIMIT,
    loadcell_scale: float = DEFAULT_OSL_LOADCELL_SCALE,
    body_weight_n: float | None = None,
    torque_slew_limit: float = DEFAULT_OSL_TORQUE_SLEW_LIMIT,
    mask_preset: str = DEFAULT_OSL_MASK_PRESET,
) -> tuple[RolloutBuffer, dict]:
    traj = load_joint_trajectory(traj_path)
    n_steps = int(max_steps) if max_steps > 0 else traj.n_frames

    env, _agent_conf, _agent_state = build_env(checkpoint, traj.motion_path)
    bw = float(body_weight_n) if body_weight_n is not None else model_body_weight_n(env.model)
    audit = audit_myofullbody_left_leg(env.model)
    masked_actuators = actuator_ids_for_names(env.model, MUSCLE_MASK_PRESETS[mask_preset])

    left_qpos = np.asarray(audit.qpos_indices, dtype=np.int32)
    left_qvel = np.asarray(audit.qvel_indices, dtype=np.int32)
    all_qpos = np.arange(env.model.nq, dtype=np.int32)
    all_qvel = np.arange(env.model.nv, dtype=np.int32)
    non_left_qpos = np.setdiff1d(all_qpos, left_qpos)
    non_left_qvel = np.setdiff1d(all_qvel, left_qvel)

    adapter = MuscleMimicOSLAdapter(
        env.model,
        audit,
        OSLAdapterConfig(
            knee_torque_limit=knee_torque_limit,
            ankle_torque_limit=ankle_torque_limit,
            foot_torque_limit=foot_torque_limit,
            loadcell_scale=loadcell_scale,
            clip_loadcell_to_body_weight=DEFAULT_OSL_CLIP_LOADCELL_TO_BODY_WEIGHT,
        ),
        body_weight_n=bw,
    )
    controller = OpenSourceLegFSMController(body_weight_n=bw)

    eval_force = default_eval_force_from_hydra(None)
    contact = FootContactExtractor(
        env.model,
        FootContactConfig.from_dict(eval_force),
        dt=float(env.dt),
    )

    env.reset()
    pin_non_left_joints(env.model, env.data, traj.qpos[0], traj.qvel[0], non_left_qpos, non_left_qvel)
    env.data.qpos[left_qpos] = traj.qpos[0][left_qpos]
    env.data.qvel[left_qvel] = traj.qvel[0][left_qvel]
    mujoco.mj_forward(env.model, env.data)
    adapter.reset_hold_targets(env.data)
    controller.reset()
    contact.reset_filters()

    action_dim = int(env.info.action_space.shape[0])
    oracle_action = np.zeros(action_dim, dtype=np.float32)
    zero_kp = np.zeros(4, dtype=np.float64)
    zero_kd = np.zeros(4, dtype=np.float64)
    torque_limit = np.asarray(
        [knee_torque_limit, ankle_torque_limit, foot_torque_limit, foot_torque_limit],
        dtype=np.float64,
    )
    torque_slew = np.full(4, float(torque_slew_limit), dtype=np.float64)
    last_torque = np.zeros(4, dtype=np.float64)

    buffer = RolloutBuffer(
        motion_path=traj.motion_path,
        dt=float(env.dt),
        traj_length=n_steps,
    )

    try:
        for step in range(n_steps):
            frame = min(step, traj.n_frames - 1)
            pin_non_left_joints(
                env.model,
                env.data,
                traj.qpos[frame],
                traj.qvel[frame],
                non_left_qpos,
                non_left_qvel,
            )

            osl_inputs = adapter.read_inputs(env.model, env.data)
            command = controller.update(osl_inputs)
            _target_qpos, ff_torque, _diag = adapter.command_to_targets(env.model, env.data, command)
            current_q = np.asarray(env.data.qpos[audit.qpos_indices], dtype=np.float64)

            _obs, reward, absorbing, done, info = step_with_stage1_pd(
                env,
                oracle_action,
                IndexedValues(audit.qpos_indices, current_q),
                audit.qvel_indices,
                masked_actuators,
                prosthesis_muscle_scale,
                zero_kp,
                zero_kd,
                torque_limit,
                torque_slew,
                last_torque,
                ff_torque,
            )
            applied_prosthesis_tau = last_torque.copy()

            pin_non_left_joints(
                env.model,
                env.data,
                traj.qpos[frame],
                traj.qvel[frame],
                non_left_qpos,
                non_left_qvel,
            )

            ref = _get_reference(env)
            contact_forces = contact.extract(env.data)
            raw_l = float(
                contact_forces.get("left_vertical_GRF_raw", contact_forces.get("left_vertical_GRF", 0.0))
            )
            raw_r = float(
                contact_forces.get("right_vertical_GRF_raw", contact_forces.get("right_vertical_GRF", 0.0))
            )
            filt_l = float(contact_forces.get("left_vertical_GRF", raw_l))
            filt_r = float(contact_forces.get("right_vertical_GRF", raw_r))
            tau = np.asarray(env.data.qfrc_actuator, dtype=np.float64).copy()

            buffer.append_step(
                t=step * buffer.dt,
                qpos=np.asarray(env.data.qpos, dtype=np.float64).copy(),
                qvel=np.asarray(env.data.qvel, dtype=np.float64).copy(),
                root_pos=np.asarray(env.data.qpos[:3], dtype=np.float64).copy(),
                root_quat=np.asarray(env.data.qpos[3:7], dtype=np.float64).copy(),
                ref_root_pos=ref.get("root_pos"),
                ref_qpos=ref.get("qpos"),
                site_pos=_get_site_pos(env, env.model, env.data),
                ref_site_pos=ref.get("site_pos"),
                left_grf=filt_l,
                right_grf=filt_r,
                left_grf_raw=raw_l,
                right_grf_raw=raw_r,
                left_grf_world=contact_forces.get("left_foot_GRF_world", np.zeros(3)),
                right_grf_world=contact_forces.get("right_foot_GRF_world", np.zeros(3)),
                contact_left=bool(contact_forces.get("left_contact_bool", False)),
                contact_right=bool(contact_forces.get("right_contact_bool", False)),
                joint_torque=tau,
                prosthesis_tau=applied_prosthesis_tau,
                reward=float(reward),
                done=bool(done),
                info=dict(info),
            )

            if step % 500 == 0:
                print(f"  step {step + 1}/{n_steps} reward={float(reward):.3f} done={done}", flush=True)
            if done:
                print(f"  done at step {step + 1}", flush=True)
                break
    finally:
        env.stop()

    root_z = float(buffer.root_pos[-1][2]) if buffer.root_pos else None
    peak_vgrf = 0.0
    if buffer.left_grf:
        peak_vgrf = max(max(buffer.left_grf), max(buffer.right_grf))

    buffer.done_reason = infer_done_reason(
        done=bool(buffer.done_flags[-1]) if buffer.done_flags else False,
        absorbing=bool(buffer.info_steps[-1].get("terminated", False)) if buffer.info_steps else False,
        info=buffer.info_steps[-1] if buffer.info_steps else {},
        step=buffer.steps,
        max_steps=n_steps,
        traj_length=n_steps,
        root_z=root_z,
        peak_vgrf=peak_vgrf,
    )

    meta = {
        "body_weight_N": float(np.sum(env.model.body_mass) * 9.81),
        "site_names": list(getattr(env, "sites_for_mimic", []) or []),
        "torque_limits": None,
        "fsm_body_weight_n": bw,
        "loadcell_scale": loadcell_scale,
    }
    return buffer, meta


def score_replay_dir(replay_dir: Path, args: argparse.Namespace) -> dict:
    replay_dir = replay_dir.resolve()
    traj_path = replay_dir / "joint_trajectory.npz"
    if not traj_path.is_file():
        raise FileNotFoundError(f"Missing {traj_path}")

    analysis_dir = replay_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rollout_npz = analysis_dir / "rollout_data.npz"
    metrics_json = analysis_dir / "metrics.json"

    print(f"\n=== Scoring FSM hybrid: {replay_dir.name} ===")
    if args.skip_rollout:
        if not rollout_npz.is_file() or not metrics_json.is_file():
            raise FileNotFoundError(f"Missing {rollout_npz} or {metrics_json}; run without --skip-rollout first.")
        motion_path = json.loads(metrics_json.read_text(encoding="utf-8")).get("motion_path")
        if not motion_path:
            traj = load_joint_trajectory(traj_path)
            motion_path = traj.motion_path
    else:
        buffer, meta = run_fsm_hybrid_rollout(
            traj_path,
            checkpoint=args.checkpoint,
            max_steps=args.max_steps,
            prosthesis_muscle_scale=args.prosthesis_muscle_scale,
            knee_torque_limit=args.knee_torque_limit,
            ankle_torque_limit=args.ankle_torque_limit,
            foot_torque_limit=args.foot_torque_limit,
            loadcell_scale=args.loadcell_scale,
            body_weight_n=args.body_weight_n,
            torque_slew_limit=args.torque_slew_limit,
        )

        metrics = compute_all_metrics(buffer, meta=meta, force_reference_path=None)
        metrics["done_reason"] = buffer.done_reason
        metrics["motion_type"] = classify_motion_type(buffer.motion_path)
        metrics["controller_type"] = "joint_replay_osl_fsm"
        metrics["env_type"] = "joint_replay_osl_fsm_hybrid"
        metrics["checkpoint_path"] = args.checkpoint

        save_metrics_json(metrics_json, metrics)
        save_rollout_npz(rollout_npz, buffer)
        motion_path = buffer.motion_path

    torque_summary, symmetry_summary = persist_rollout_analysis_summaries(
        analysis_dir,
        rollout_npz,
        motion_path,
        config_name=args.config_name,
        include_symmetry=True,
        torque_source_mode=TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID,
    )

    report = build_composite_report(
        metrics_json,
        torque_summary=torque_summary,
        symmetry_summary=symmetry_summary,
        motion_path=motion_path,
    )
    save_composite_report(report, analysis_dir)
    plot_composite_radar(report, analysis_dir / "composite_score_radar.png")

    cq = report["dimensions"]["control_quality"]["score"]
    sym = report["dimensions"]["bilateral_symmetry"]["score"]
    print(f"  control_quality: {cq:.2f}")
    print(f"  bilateral_symmetry: {sym:.2f}")
    print(f"  TOTAL: {report['total_score']:.2f} / 100")
    print(f"  Wrote {analysis_dir / 'section4_torque' / 'torque_metrics_summary.json'}")
    print(f"  Wrote {analysis_dir / 'section5_symmetry' / 'symmetry_metrics_summary.json'}")
    print(f"  Wrote {analysis_dir / 'composite_score.json'}")
    print(f"  Wrote {analysis_dir / 'composite_score_radar.png'}")
    return report


def _load_composite(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _dim_scores(report: dict | None) -> dict[str, float]:
    if not report:
        return {d: float("nan") for d in DIM_ORDER}
    return {d: float(report["dimensions"][d]["score"]) for d in DIM_ORDER if d in report.get("dimensions", {})}


def write_comparison_table(path: Path) -> None:
    rows = []
    for safe in FOUR_REPLAY_MOTIONS:
        official = _load_composite(DEFAULT_OFFICIAL_EVAL_BASE / safe / "analysis" / "composite_score.json")
        fsm = _load_composite(DEFAULT_REPLAY_BASE / safe / "analysis" / "composite_score.json")
        motion_path = (
            (official or fsm or {}).get("motion_path")
            or "/".join(safe.split("_", 2)[:2] + [safe.split("_", 2)[2]]) if safe.count("_") >= 2 else safe
        )
        if motion_path == safe and safe.count("_") >= 2:
            a, b, c = safe.split("_", 2)
            motion_path = f"{a}/{b}/{c}"
        off_dims = _dim_scores(official)
        fsm_dims = _dim_scores(fsm)
        off_total = official["total_score"] if official else float("nan")
        fsm_total = fsm["total_score"] if fsm else float("nan")
        delta = fsm_total - off_total if official and fsm else float("nan")
        rows.append(
            {
                "safe": safe,
                "motion_path": motion_path,
                "official_total": off_total,
                "fsm_total": fsm_total,
                "delta": delta,
                "official_dims": off_dims,
                "fsm_dims": fsm_dims,
            }
        )

    dim_labels = {
        "task_reliability": "任务",
        "pose_tracking": "姿态",
        "global_trajectory": "轨迹",
        "contact_biomechanics": "接触",
        "bilateral_symmetry": "对称",
        "control_quality": "控制",
    }

    lines = [
        "# FSM Replay 综合评分对比",
        "",
        "对比 **官方 Policy rollout** 与 **Joint Replay + OSL FSM 混合动力学** 在四个已录制动作上的六维综合评分（0–100）。",
        "",
        "评分配置：`musclemimic/fullbody/conf_eval_composite_score.yaml`（对称步态 6D profile）。",
        "",
        "产物路径：",
        "",
        "- 官方 Policy：`musclemimic/outputs/eval/runs/official_mm10m2/6d_symmetric_gait/<SAFE>/analysis/`",
        "- FSM hybrid：`musclemimic/outputs/replay/joint_mimic/official_mm10m2/<SAFE>/analysis/`",
        "",
        "## 总分对比",
        "",
        "| 动作 | 官方 Policy | FSM hybrid | Δ (FSM − 官方) |",
        "|------|------------:|-----------:|-----------------:|",
    ]

    for row in rows:
        off_s = f"{row['official_total']:.2f}" if row["official_total"] == row["official_total"] else "—"
        fsm_s = f"{row['fsm_total']:.2f}" if row["fsm_total"] == row["fsm_total"] else "—"
        delta_s = f"{row['delta']:+.2f}" if row["delta"] == row["delta"] else "—"
        lines.append(f"| `{row['motion_path']}` | {off_s} | {fsm_s} | {delta_s} |")

    lines.extend(
        [
            "",
            "## 六维子分（官方 Policy / FSM hybrid）",
            "",
            "| 动作 | 任务 | 姿态 | 轨迹 | 接触 | 对称 | 控制 |",
            "|------|-----:|-----:|-----:|-----:|-----:|-----:|",
        ]
    )

    for row in rows:
        off = row["official_dims"]
        fsm = row["fsm_dims"]
        cells = []
        for dim in DIM_ORDER:
            o = off.get(dim, float("nan"))
            f = fsm.get(dim, float("nan"))
            if o == o and f == f:
                cells.append(f"{o:.1f} / {f:.1f}")
            elif o == o:
                cells.append(f"{o:.1f} / —")
            elif f == f:
                cells.append(f"— / {f:.1f}")
            else:
                cells.append("— / —")
        lines.append(f"| `{row['motion_path']}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "列含义：每格为「官方 / FSM」。",
            "",
            "## 雷达图",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['motion_path']}`："
            f"[官方](musclemimic/outputs/eval/runs/official_mm10m2/6d_symmetric_gait/{row['safe']}/analysis/composite_score_radar.png) · "
            f"[FSM](musclemimic/outputs/replay/joint_mimic/official_mm10m2/{row['safe']}/analysis/composite_score_radar.png)"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote comparison table: {path}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    targets: list[Path] = []
    if args.all_four:
        targets = [DEFAULT_REPLAY_BASE / name for name in FOUR_REPLAY_MOTIONS]
    elif args.replay_dir:
        targets = [Path(args.replay_dir)]
    elif args.joint_trajectory:
        targets = [Path(args.joint_trajectory).parent]
    else:
        print("Provide --joint-trajectory, --replay-dir, or --all-four", file=sys.stderr)
        return 2

    for replay_dir in targets:
        score_replay_dir(replay_dir, args)

    if args.write_comparison_table or args.all_four:
        write_comparison_table(Path(args.comparison_table_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
