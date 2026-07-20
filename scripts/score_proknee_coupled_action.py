#!/usr/bin/env python
"""Closed-loop ProKnee coupled-action rollout + composite locomotion score."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musclemimic.evaluation.adapters import _get_reference
from musclemimic.evaluation.composite_score import build_composite_report, save_composite_report
from musclemimic.evaluation.contact_extractor import FootContactConfig, FootContactExtractor
from musclemimic.evaluation.light_summaries import (
    TORQUE_SOURCE_FSM_PROSTHESIS_HYBRID,
    persist_rollout_analysis_summaries,
)
from musclemimic.evaluation.logger import safe_motion_name, save_metrics_json, save_rollout_npz
from musclemimic.evaluation.metrics import compute_all_metrics
from musclemimic.evaluation.motion_taxonomy import classify_motion_type
from musclemimic.evaluation.radar_plot import plot_composite_radar
from musclemimic.evaluation.rollout_loop import infer_done_reason
from musclemimic.evaluation.runner import default_eval_force_from_hydra
from musclemimic.evaluation.types import RolloutBuffer
from musclemimic.proknee import MuscleProKneeCoupledActionEnv
from musclemimic.proknee.action_models import CoupledActionPolicy

DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "data/checkpoints/mm-10m-2"

FOUR_REPLAY_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/425/walking_slow07_poses",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, help="Coupled-action checkpoint (.pt pickle)")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--motion-path", default=None)
    p.add_argument("--all-four", action="store_true")
    p.add_argument("--output-dir", default=None, help="Default: outputs/_nonformal_runs/proknee_coupled_eval/<safe_motion>")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--skip-rollout", action="store_true")
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    p.add_argument("--record-path", default=None, help="Optional MP4 path; delegates to record_proknee_coupled_action.py")
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


def build_mean_action(model, *, body_residual: bool):
    @jax.jit
    def mean_action(params, obs, priv):
        out = model.apply(params, obs, priv, body_residual=body_residual)
        if body_residual and out["body_mean"].shape[-1] > 0:
            return jnp.concatenate([out["prosthesis_mean"], out["body_mean"]], axis=-1)
        return out["prosthesis_mean"]

    return mean_action


def load_policy(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_coupled_rollout(
    *,
    policy_path: str,
    checkpoint: str,
    motion_path: str,
    max_steps: int,
) -> tuple[RolloutBuffer, dict]:
    ckpt = load_policy(policy_path)
    env = MuscleProKneeCoupledActionEnv(
        checkpoint,
        dataset_group=None,
        rel_dataset_path=[motion_path.removesuffix(".npz")],
        deterministic_oracle=True,
        action_mode=str(ckpt.get("action_mode", "torque")),
        action_torque_limit=tuple(ckpt.get("action_torque_limit", (110.0, 65.0, 45.0, 22.0))),
        action_torque_slew_limit=ckpt.get("action_torque_slew_limit", (10.0, 5.0, 3.0, 1.5)),
        prosthesis_muscle_scale=float(ckpt.get("prosthesis_muscle_scale", 0.0)),
        body_residual_scale=float(ckpt.get("body_residual_scale", 0.0)),
        mask_preset=str(ckpt.get("mask_preset", "strict19")),
        obs_mode=str(ckpt.get("obs_mode", "easy")),
    )
    body_residual = bool(ckpt.get("body_residual_adapter", False))
    model = CoupledActionPolicy(
        prosthesis_action_dim=int(ckpt.get("prosthesis_action_dim", env.action_dim)),
        body_residual_dim=int(ckpt.get("body_residual_dim", 0)),
        action_scale=tuple(ckpt.get("action_torque_limit", (110.0, 65.0, 45.0, 22.0))),
    )
    mean_fn = build_mean_action(model, body_residual=body_residual)
    body_dim = int(ckpt.get("body_residual_dim", 0))

    eval_force = default_eval_force_from_hydra(None)
    contact = FootContactExtractor(env.env.model, FootContactConfig.from_dict(eval_force), dt=float(env.env.dt))

    data = env.reset()
    contact.reset_filters()
    buffer = RolloutBuffer(motion_path=motion_path, dt=float(env.env.dt), traj_length=max_steps)

    try:
        for step in range(max_steps):
            obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
            priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
            action = np.asarray(mean_fn(ckpt["params"], obs, priv))[0]
            p_action = action[: env.action_dim]
            b_action = None if body_dim == 0 else action[env.action_dim : env.action_dim + body_dim]
            data = env.step_action(p_action, b_action)

            ref = _get_reference(env.env)
            contact_forces = contact.extract(env.env.data)
            raw_l = float(contact_forces.get("left_vertical_GRF_raw", contact_forces.get("left_vertical_GRF", 0.0)))
            raw_r = float(contact_forces.get("right_vertical_GRF_raw", contact_forces.get("right_vertical_GRF", 0.0)))
            filt_l = float(contact_forces.get("left_vertical_GRF", raw_l))
            filt_r = float(contact_forces.get("right_vertical_GRF", raw_r))
            prosthesis_tau = np.asarray(data.info.get("prosthesis_applied_torque_mean", np.zeros(4)), dtype=np.float64)
            tau = np.asarray(env.env.data.qfrc_actuator, dtype=np.float64).copy()

            buffer.append_step(
                t=step * buffer.dt,
                qpos=np.asarray(env.env.data.qpos, dtype=np.float64).copy(),
                qvel=np.asarray(env.env.data.qvel, dtype=np.float64).copy(),
                root_pos=np.asarray(env.env.data.qpos[:3], dtype=np.float64).copy(),
                root_quat=np.asarray(env.env.data.qpos[3:7], dtype=np.float64).copy(),
                ref_root_pos=ref.get("root_pos"),
                ref_qpos=ref.get("qpos"),
                site_pos=_get_site_pos(env.env, env.env.model, env.env.data),
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
                prosthesis_tau=prosthesis_tau,
                reward=float(data.reward),
                done=bool(data.done),
                info=dict(data.info),
            )
            if step % 200 == 0:
                print(
                    f"  step {step + 1}/{max_steps} reward={float(data.reward):.3f} done={data.done} "
                    f"tracking={float(data.info.get('prosthesis_tracking_mae', 0.0)):.4f}",
                    flush=True,
                )
            if data.done:
                print(f"  done at step {step + 1}", flush=True)
                break
    finally:
        env.env.stop()

    root_z = float(buffer.root_pos[-1][2]) if buffer.root_pos else None
    peak_vgrf = 0.0
    if buffer.left_grf:
        peak_vgrf = max(max(buffer.left_grf), max(buffer.right_grf))
    buffer.done_reason = infer_done_reason(
        done=bool(buffer.done_flags[-1]) if buffer.done_flags else False,
        absorbing=bool(buffer.info_steps[-1].get("terminated", False)) if buffer.info_steps else False,
        info=buffer.info_steps[-1] if buffer.info_steps else {},
        step=buffer.steps,
        max_steps=max_steps,
        traj_length=max_steps,
        root_z=root_z,
        peak_vgrf=peak_vgrf,
    )
    meta = {
        "body_weight_N": float(np.sum(env.env.model.body_mass) * 9.81),
        "site_names": list(getattr(env.env, "sites_for_mimic", []) or []),
        "torque_limits": list(ckpt.get("action_torque_limit", (110.0, 65.0, 45.0, 22.0))),
        "controller_type": "proknee_coupled_action",
        "mask_preset": str(ckpt.get("mask_preset", "strict19")),
        "obs_mode": str(ckpt.get("obs_mode", "easy")),
        "policy_path": policy_path,
    }
    return buffer, meta


def score_motion(args: argparse.Namespace, motion_path: str) -> dict:
    safe = safe_motion_name(motion_path)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        out_dir = REPO_ROOT / "outputs/_nonformal_runs/proknee_coupled_eval" / safe
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rollout_npz = analysis_dir / "rollout_data.npz"
    metrics_json = analysis_dir / "metrics.json"

    print(f"\n=== Scoring coupled action: {motion_path} ===")
    if args.skip_rollout:
        if not rollout_npz.is_file() or not metrics_json.is_file():
            raise FileNotFoundError(f"Missing {rollout_npz} or {metrics_json}; run without --skip-rollout first.")
        motion_path = json.loads(metrics_json.read_text(encoding="utf-8")).get("motion_path", motion_path)
    else:
        if args.record_path:
            import subprocess

            record_path = Path(args.record_path)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/record_proknee_coupled_action.py"),
                "--policy",
                args.policy,
                "--checkpoint",
                args.checkpoint,
                "--motion-path",
                motion_path,
                "--steps",
                str(args.max_steps),
                "--record-path",
                str(record_path),
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

        buffer, meta = run_coupled_rollout(
            policy_path=args.policy,
            checkpoint=args.checkpoint,
            motion_path=motion_path,
            max_steps=args.max_steps,
        )
        metrics = compute_all_metrics(buffer, meta=meta, force_reference_path=None)
        metrics["done_reason"] = buffer.done_reason
        metrics["motion_type"] = classify_motion_type(buffer.motion_path)
        metrics["controller_type"] = "proknee_coupled_action"
        metrics["env_type"] = "proknee_coupled_action_hybrid"
        metrics["checkpoint_path"] = args.checkpoint
        metrics["policy_path"] = args.policy
        save_metrics_json(metrics_json, metrics)
        save_rollout_npz(rollout_npz, buffer)

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
    print(f"  TOTAL: {report['total_score']:.2f} / 100")
    print(f"  Wrote {analysis_dir / 'composite_score.json'}")
    print(f"  Wrote {analysis_dir / 'composite_score_radar.png'}")
    return report


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    motions = FOUR_REPLAY_MOTIONS if args.all_four else [args.motion_path]
    if not motions or motions == [None]:
        print("Provide --motion-path or --all-four", file=sys.stderr)
        return 2

    reports = []
    for motion_path in motions:
        reports.append(score_motion(args, motion_path))

    if args.all_four:
        summary_path = REPO_ROOT / "outputs/_nonformal_runs/proknee_coupled_eval/replay_four_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": args.policy,
            "motions": [
                {"motion_path": m, "total_score": float(r["total_score"])} for m, r in zip(FOUR_REPLAY_MOTIONS, reports)
            ],
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
