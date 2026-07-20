#!/usr/bin/env python
"""Systematic diagnostics for the distilled prosthesis policy and labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import (
    apply_prosthesis_overrides,
    parse_optional_csv,
    parse_optional_vec4,
)
from musclemimic.distill.mapping import DistillMapping
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.rollout import contact_load_by_bodies, safe_motion_filename
from musclemimic.prosthesis.constants import PROSTHESIS_JOINT_NAMES
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint


DEFAULT_MOTION = "KIT/3/walk_6m_straight_line04_poses"
DEFAULT_DATASET_DIR = "data/teacher_rollouts/KIT_KINESIS_TRAINING_MOTIONS"
DEFAULT_CKPT = "outputs/prosthesis_distill/latest/checkpoints/checkpoint_distilled"
DEFAULT_OUT = "outputs/distill_only/diagnostics"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-path", default=DEFAULT_MOTION)
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--plot-steps", type=int, default=200)
    p.add_argument("--sign-test-tau", type=float, default=10.0)
    p.add_argument("--skip-policy", action="store_true", help="Only run action replay diagnostics, not checkpoint policy rollout")
    p.add_argument("--disable_muscles_mode", default=None)
    p.add_argument("--disable_muscles_include", default=None)
    p.add_argument("--disable_muscles_exclude", default=None)
    p.add_argument("--prosthesis_torque_limits", default=None)
    p.add_argument("--prosthesis_action_type", default=None, choices=[None, "torque", "pd_residual_torque"])
    p.add_argument("--residual_pd_kp", default=None)
    p.add_argument("--residual_pd_kd", default=None)
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def make_env_from_checkpoint(checkpoint: Path, motion_path: str, args: argparse.Namespace):
    config, agent_state, metadata = load_checkpoint(str(checkpoint))
    OmegaConf.set_struct(config, False)
    apply_prosthesis_overrides(
        config,
        disable_mode=args.disable_muscles_mode,
        disable_include=parse_optional_csv(args.disable_muscles_include),
        disable_exclude=parse_optional_csv(args.disable_muscles_exclude),
        torque_limits=parse_optional_vec4(args.prosthesis_torque_limits),
        action_type=args.prosthesis_action_type,
        residual_pd_kp=parse_optional_vec4(args.residual_pd_kp),
        residual_pd_kd=parse_optional_vec4(args.residual_pd_kd),
    )
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBodyProsthesisEnv"
    env_params["headless"] = True
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    prosthesis = dict(env_params.get("prosthesis", {}) or {})
    prosthesis["enabled"] = True
    prosthesis["control_mode"] = "train_policy"
    env_params["prosthesis"] = prosthesis
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass
    control_dt = apply_temporal_params(config)
    env = TaskFactory.get_factory_cls(config.experiment.task_factory.name).make(**{**env_params, **task_params})
    return config, agent_state, metadata, env, control_dt


def quat_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = [float(v) for v in q]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def current_obs(env) -> np.ndarray:
    obs, carry = env._create_observation(env.model, env.data, env._additional_carry)
    env._additional_carry = carry
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def mapping_rows(env, mapping: DistillMapping, data: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    tau = np.asarray(data["target_prosthesis_tau"], dtype=np.float32)
    action = np.asarray(data["target_prosthesis_action"], dtype=np.float32)
    for i, name in enumerate(PROSTHESIS_JOINT_NAMES):
        jid = int(env.prosthesis_mapping.prosthesis_joint_ids[i])
        aid = int(env.prosthesis_mapping.prosthesis_actuator_ids[i])
        rows.append(
            {
                "order": i,
                "joint_name": name,
                "qpos_id": int(env.model.jnt_qposadr[jid]),
                "qvel_id": int(env.model.jnt_dofadr[jid]),
                "dof_id": int(env.model.jnt_dofadr[jid]),
                "actuator_id": aid,
                "actuator_name": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
                "torque_limit": float(env.prosthesis_torque_limits[i]),
                "target_tau_mean": float(np.mean(tau[:, i])),
                "target_tau_std": float(np.std(tau[:, i])),
                "target_tau_min": float(np.min(tau[:, i])),
                "target_tau_max": float(np.max(tau[:, i])),
                "target_action_clipping_ratio": float(np.mean(np.abs(action[:, i]) >= 0.999)),
            }
        )
    return rows


def sign_test(env, tau_mag: float) -> list[dict]:
    rows = []
    base_qvel = np.asarray(env.data.qvel).copy()
    base_qpos = np.asarray(env.data.qpos).copy()
    qvel_ids = np.asarray(env.prosthesis_mapping.prosthesis_qvel_indices, dtype=np.int32)
    actuator_ids = np.asarray(env.prosthesis_mapping.prosthesis_actuator_ids, dtype=np.int32)
    for i, name in enumerate(PROSTHESIS_JOINT_NAMES):
        for sign in (1.0, -1.0):
            env.reset()
            env.data.qvel[:] = base_qvel
            env.data.qpos[:] = base_qpos
            mujoco.mj_forward(env.model, env.data)
            env.data.ctrl[:] = 0.0
            env.data.ctrl[actuator_ids[i]] = sign * tau_mag
            mujoco.mj_forward(env.model, env.data)
            qacc0 = np.asarray(env.data.qacc[qvel_ids], dtype=np.float64).copy()
            qvel_before = np.asarray(env.data.qvel[qvel_ids], dtype=np.float64).copy()
            mujoco.mj_step(env.model, env.data)
            qvel_after = np.asarray(env.data.qvel[qvel_ids], dtype=np.float64).copy()
            rows.append(
                {
                    "joint": name,
                    "applied_tau": sign * tau_mag,
                    "qacc_target_dof": float(qacc0[i]),
                    "qvel_delta_target_dof": float(qvel_after[i] - qvel_before[i]),
                    "qacc_all_4dof": qacc0.tolist(),
                    "qvel_delta_all_4dof": (qvel_after - qvel_before).tolist(),
                }
            )
    return rows


def replay_dataset_actions(env, data: dict[str, np.ndarray], mapping: DistillMapping, *, mode: str, steps: int):
    obs = env.reset()
    n = min(steps, int(data["obs_student"].shape[0]))
    traj = []
    for t in range(n):
        rem = np.asarray(data["target_remaining_muscle_action"][t], dtype=np.float32)
        if mode == "normalized":
            prost = np.asarray(data["target_prosthesis_action"][t], dtype=np.float32)
        elif mode == "tau_direct":
            prost = np.asarray(data["target_prosthesis_tau"][t], dtype=np.float32)
        elif mode == "zero_residual_pd":
            prost = np.zeros(4, dtype=np.float32)
        else:
            raise ValueError(mode)
        action = np.concatenate([rem, prost]).astype(np.float32)
        obs_l2 = float(np.linalg.norm(np.asarray(obs).reshape(-1) - data["obs_student"][t]))
        obs, reward, _absorbing, done, info = env.step(action)
        pobs = env.get_prosthesis_obs()
        grf_pair, contact_pair = contact_load_by_bodies(env.model, env.data, mapping.prosthesis_body_ids)
        roll, pitch = quat_roll_pitch(np.asarray(env.data.qpos[3:7], dtype=np.float64))
        q = np.asarray(env.data.qpos[np.asarray(mapping.prosthesis_qpos_indices)], dtype=np.float32)
        q_ref = np.asarray(data["q_prosthesis"][t], dtype=np.float32)
        disabled_force = float(info.get("disabled_muscle_force_norm", 0.0))
        all_force = float(np.linalg.norm(np.asarray(env.data.actuator_force, dtype=np.float64)) + 1e-8)
        traj.append(
            {
                "step": t,
                "mode": mode,
                "reward": float(np.asarray(reward).item()),
                "done": bool(done),
                "root_height": float(env.data.qpos[2]),
                "root_roll": roll,
                "root_pitch": pitch,
                "knee_q": float(q[0]),
                "ankle_q": float(q[1]),
                "subtalar_q": float(q[2]),
                "mtp_q": float(q[3]),
                "knee_tau": float(prost[0]),
                "ankle_tau": float(prost[1]),
                "subtalar_tau": float(prost[2]),
                "mtp_tau": float(prost[3]),
                "left_contact": float(contact_pair[0]),
                "right_contact": float(contact_pair[1]),
                "left_grf": float(grf_pair[0]),
                "right_grf": float(grf_pair[1]),
                "joint_error": float(np.linalg.norm(q - q_ref)),
                "action_norm": float(np.linalg.norm(action)),
                "obs_l2_vs_dataset": obs_l2,
                "disabled_muscle_ctrl_norm": float(info.get("disabled_muscle_ctrl_norm", 0.0)),
                "disabled_muscle_force_norm": disabled_force,
                "disabled_muscle_force_ratio": disabled_force / all_force,
                "prosthesis_tau_info_norm": float(np.linalg.norm(np.asarray(info.get("prosthesis_tau", np.zeros(4))))),
            }
        )
    return traj


def rollout_policy(env, config, agent_state, data: dict[str, np.ndarray], mapping: DistillMapping, steps: int):
    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    runner = PolicyRunner.from_agent_state(agent_conf, agent_state, env, deterministic=True, seed=0)
    obs = env.reset()
    obs_policy = runner.reset_obs(obs)
    rows = []
    n = min(steps, int(data["obs_student"].shape[0]))
    for t in range(n):
        action, _ = runner.act(obs_policy)
        obs, reward, _absorbing, done, info = env.step(action)
        obs_policy = runner.update_obs(obs)
        grf_pair, contact_pair = contact_load_by_bodies(env.model, env.data, mapping.prosthesis_body_ids)
        roll, pitch = quat_roll_pitch(np.asarray(env.data.qpos[3:7], dtype=np.float64))
        q = np.asarray(env.data.qpos[np.asarray(mapping.prosthesis_qpos_indices)], dtype=np.float32)
        q_ref = np.asarray(data["q_prosthesis"][t], dtype=np.float32)
        all_force = float(np.linalg.norm(np.asarray(env.data.actuator_force, dtype=np.float64)) + 1e-8)
        disabled_force = float(info.get("disabled_muscle_force_norm", 0.0))
        rows.append(
            {
                "step": t,
                "mode": "policy",
                "reward": float(np.asarray(reward).item()),
                "done": bool(done),
                "root_height": float(env.data.qpos[2]),
                "root_roll": roll,
                "root_pitch": pitch,
                "knee_q": float(q[0]),
                "ankle_q": float(q[1]),
                "subtalar_q": float(q[2]),
                "mtp_q": float(q[3]),
                "knee_tau": float(action[335]),
                "ankle_tau": float(action[336]),
                "subtalar_tau": float(action[337]),
                "mtp_tau": float(action[338]),
                "left_contact": float(contact_pair[0]),
                "right_contact": float(contact_pair[1]),
                "left_grf": float(grf_pair[0]),
                "right_grf": float(grf_pair[1]),
                "joint_error": float(np.linalg.norm(q - q_ref)),
                "action_norm": float(np.linalg.norm(action)),
                "obs_l2_vs_dataset": float(np.linalg.norm(np.asarray(obs).reshape(-1) - data["obs_student"][t])),
                "disabled_muscle_ctrl_norm": float(info.get("disabled_muscle_ctrl_norm", 0.0)),
                "disabled_muscle_force_norm": disabled_force,
                "disabled_muscle_force_ratio": disabled_force / all_force,
                "prosthesis_tau_info_norm": float(np.linalg.norm(np.asarray(info.get("prosthesis_tau", np.zeros(4))))),
            }
        )
    return rows


def summarize_traj(rows: list[dict]) -> dict:
    root = np.asarray([r["root_height"] for r in rows], dtype=np.float64)
    done = np.asarray([r["done"] for r in rows], dtype=bool)
    return {
        "steps": len(rows),
        "first_done_step": int(np.argmax(done)) if done.any() else None,
        "done_count": int(done.sum()),
        "root_height_start": float(root[0]) if root.size else 0.0,
        "root_height_end": float(root[-1]) if root.size else 0.0,
        "root_height_min": float(root.min()) if root.size else 0.0,
        "obs_l2_mean": float(np.mean([r["obs_l2_vs_dataset"] for r in rows])),
        "joint_error_mean": float(np.mean([r["joint_error"] for r in rows])),
        "disabled_ctrl_max": float(np.max([r["disabled_muscle_ctrl_norm"] for r in rows])),
        "disabled_force_norm_max": float(np.max([r["disabled_muscle_force_norm"] for r in rows])),
        "disabled_force_ratio_max": float(np.max([r["disabled_muscle_force_ratio"] for r in rows])),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(path: Path, rows_by_mode: dict[str, list[dict]], plot_steps: int) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
    for mode, rows in rows_by_mode.items():
        rows = rows[:plot_steps]
        x = np.asarray([r["step"] for r in rows])
        axes[0].plot(x, [r["root_height"] for r in rows], label=f"{mode} height")
        axes[0].plot(x, [r["root_roll"] for r in rows], linestyle="--", label=f"{mode} roll")
        axes[0].plot(x, [r["root_pitch"] for r in rows], linestyle=":", label=f"{mode} pitch")
        for key in ("knee_q", "ankle_q", "subtalar_q", "mtp_q"):
            axes[1].plot(x, [r[key] for r in rows], label=f"{mode} {key}")
        for key in ("knee_tau", "ankle_tau", "subtalar_tau", "mtp_tau"):
            axes[2].plot(x, [r[key] for r in rows], label=f"{mode} {key}")
        axes[3].plot(x, [r["left_contact"] for r in rows], label=f"{mode} left_contact")
        axes[3].plot(x, [r["right_contact"] for r in rows], linestyle="--", label=f"{mode} right_contact")
        axes[4].plot(x, [r["left_grf"] for r in rows], label=f"{mode} left_grf")
        axes[4].plot(x, [r["right_grf"] for r in rows], linestyle="--", label=f"{mode} right_grf")
        axes[5].plot(x, [r["joint_error"] for r in rows], label=f"{mode} joint_error")
        axes[6].plot(x, [r["action_norm"] for r in rows], label=f"{mode} action_norm")
    titles = [
        "root height / roll / pitch",
        "prosthesis q",
        "prosthesis tau/action channels",
        "contact flags",
        "GRF left/right",
        "4DOF joint error vs teacher dataset",
        "action norm",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("step (100 Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = resolve(args.dataset_dir)
    checkpoint = resolve(args.checkpoint)
    npz_path = dataset_dir / safe_motion_filename(args.motion_path)
    mapping = DistillMapping.load_json(dataset_dir / "mapping.json")
    with np.load(npz_path, allow_pickle=True) as npz:
        data = {k: np.asarray(npz[k]) for k in npz.files}

    config, agent_state, metadata, env, control_dt = make_env_from_checkpoint(checkpoint, args.motion_path, args)
    action_names = [env.model.actuator(int(aid)).name for aid in getattr(env, "_action_indices", range(env.model.nu))]
    action_order_ok = (
        list(action_names[: mapping.n_remaining_muscles]) == list(mapping.remaining_muscle_names)
        and list(action_names[-4:]) == list(mapping.prosthesis_motor_names)
        and list(mapping.prosthesis_joint_names) == list(PROSTHESIS_JOINT_NAMES)
    )

    rows_map = mapping_rows(env, mapping, data)
    sign_rows = sign_test(env, float(args.sign_test_tau))

    initial_obs = env.reset()
    obs0_l2 = float(np.linalg.norm(np.asarray(initial_obs).reshape(-1) - data["obs_student"][0]))
    obs0_linf = float(np.max(np.abs(np.asarray(initial_obs).reshape(-1) - data["obs_student"][0])))
    normalizer = {
        "len_obs_history": int(getattr(config.experiment, "len_obs_history", 1)),
        "split_goal": bool(getattr(config.experiment, "split_goal", False)),
        "has_run_stats": bool(hasattr(agent_state.train_state, "run_stats")),
        "run_stats_type": type(agent_state.train_state.run_stats).__name__,
        "note": "Network apply uses checkpoint run_stats; dataset stores raw obs_student arrays, no separate mean/std file is used by distill dataset.",
    }

    replay_normalized = replay_dataset_actions(env, data, mapping, mode="normalized", steps=args.steps)
    replay_tau = replay_dataset_actions(env, data, mapping, mode="tau_direct", steps=args.steps)
    policy_rows = [] if args.skip_policy else rollout_policy(env, config, agent_state, data, mapping, steps=args.steps)
    replay_zero_pd = (
        replay_dataset_actions(env, data, mapping, mode="zero_residual_pd", steps=args.steps)
        if getattr(env, "prosthesis_action_type", "torque") == "pd_residual_torque"
        else []
    )

    target_remaining = np.asarray(data["target_remaining_muscle_action"], dtype=np.float32)
    teacher_all = np.asarray(data["teacher_action_all_muscles"], dtype=np.float32)
    teacher_rem_from_all = teacher_all[:, np.asarray(mapping.teacher_remaining_action_indices, dtype=np.int32)]
    muscle_range = {
        "target_remaining_min": float(target_remaining.min()),
        "target_remaining_max": float(target_remaining.max()),
        "teacher_all_min": float(teacher_all.min()),
        "teacher_all_max": float(teacher_all.max()),
        "target_matches_teacher_remaining_max_abs": float(np.max(np.abs(target_remaining - teacher_rem_from_all))),
    }

    summary = {
        "motion_path": args.motion_path,
        "dataset_npz": str(npz_path),
        "checkpoint": str(checkpoint),
        "control_dt": control_dt,
        "action_order_ok": bool(action_order_ok),
        "expected_order": ["knee", "ankle", "subtalar", "mtp"],
        "student_last4_action_names": action_names[-4:],
        "mapping_joint_names": list(mapping.prosthesis_joint_names),
        "obs0_l2_dataset_vs_env_reset": obs0_l2,
        "obs0_linf_dataset_vs_env_reset": obs0_linf,
        "obs_normalizer": normalizer,
        "muscle_action_range": muscle_range,
        "replay_summaries": {
            "normalized": summarize_traj(replay_normalized),
            "tau_direct": summarize_traj(replay_tau),
            "zero_residual_pd": summarize_traj(replay_zero_pd) if replay_zero_pd else None,
            "policy": summarize_traj(policy_rows) if policy_rows else None,
        },
        "mapping": rows_map,
        "sign_test": sign_rows,
        "checkpoint_metadata": str(metadata),
    }

    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_csv(out_dir / "prosthesis_mapping.csv", rows_map)
    write_csv(out_dir / "sign_test.csv", sign_rows)
    write_csv(out_dir / "replay_normalized.csv", replay_normalized)
    write_csv(out_dir / "replay_tau_direct.csv", replay_tau)
    write_csv(out_dir / "replay_zero_residual_pd.csv", replay_zero_pd)
    write_csv(out_dir / "policy_rollout.csv", policy_rows)
    rows_by_mode = {"normalized_replay": replay_normalized, "tau_direct_replay": replay_tau}
    if replay_zero_pd:
        rows_by_mode["zero_residual_pd"] = replay_zero_pd
    if policy_rows:
        rows_by_mode["policy"] = policy_rows
    plot_diagnostics(
        out_dir / "first_2s_diagnostics.png",
        rows_by_mode,
        args.plot_steps,
    )
    print(json.dumps(summary, indent=2, default=str))
    env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
