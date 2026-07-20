#!/usr/bin/env python
"""Capture one official muscle rollout for joint-muscle hybrid replay (self-aligned reference)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import repo_root
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.joint_replay import save_joint_trajectory, write_replay_meta
from musclemimic.evaluation.joint_replay import JointTrajectory
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.muscle_replay import record_muscle_closed_loop_rollout, save_muscle_trajectory
from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", default="KIT/3/walk_6m_straight_line04_poses")
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument(
        "--checkpoint_path",
        default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Default: outputs/replay/joint_muscle_hybrid/<controller>/<SAFE>/",
    )
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=0, help="0 = full trajectory length")
    return p.parse_args()


def build_env(config, motion_path: str):
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBody"
    env_params["headless"] = True
    env_params.pop("prosthesis", None)
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
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
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    safe = safe_motion_name(args.motion_path)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root() / "outputs" / "replay" / "joint_muscle_hybrid" / args.controller_type / safe
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config, agent_state, metadata = load_checkpoint(args.checkpoint_path)
    OmegaConf.set_struct(config, False)
    apply_temporal_params(config)
    env = build_env(config, args.motion_path)
    try:
        agent_conf = PPOJax.init_agent_conf(env, config)
        agent_state = align_agent_state(agent_state, agent_conf)
        runner = PolicyRunner.from_agent_state(
            agent_conf,
            agent_state,
            env,
            deterministic=True,
            seed=int(args.eval_seed),
        )
        n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
        print(f"[capture hybrid reference] {args.motion_path} -> {out_dir} ({n_steps} steps)")

        muscle_traj, episode_return, done_count = record_muscle_closed_loop_rollout(
            env=env,
            policy=runner,
            n_steps=n_steps,
            record_path=None,
        )
        muscle_traj.motion_path = args.motion_path
        muscle_traj.source_controller = str(args.controller_type)
        muscle_traj.source_checkpoint = str(args.checkpoint_path)
        muscle_traj.replay_mode = "muscle_closed_loop_hybrid_reference"
        muscle_traj.metadata = {
            **(muscle_traj.metadata or {}),
            "capture_purpose": "joint_muscle_hybrid_self_aligned_reference",
            "alignment": "muscle.qpos[i] pre-step, muscle.qpos[i+1] post-step, actuator_ctrl[i] same rollout",
        }

        muscle_path = out_dir / "aligned_muscle_trajectory.npz"
        save_muscle_trajectory(muscle_path, muscle_traj)

        audit = audit_myofullbody_left_leg(env.model)
        left_q = np.asarray(audit.qpos_indices, dtype=np.int32)
        left_v = np.asarray(audit.qvel_indices, dtype=np.int32)
        post_qpos = muscle_traj.qpos[1:]
        post_qvel = muscle_traj.qvel[1:]
        left_reference = {
            "motion_path": args.motion_path,
            "dt": float(muscle_traj.dt),
            "left_joint_names": [j.name for j in audit.joints],
            "left_qpos": post_qpos[:, left_q].astype(np.float64),
            "left_qvel": post_qvel[:, left_v].astype(np.float64),
            "source_muscle_trajectory_npz": str(muscle_path),
        }
        left_ref_path = out_dir / "left_leg_joint_reference.npz"
        np.savez_compressed(left_ref_path, **left_reference)

        joint_traj = JointTrajectory(
            motion_path=args.motion_path,
            dt=float(muscle_traj.dt),
            qpos=post_qpos.astype(np.float64),
            qvel=post_qvel.astype(np.float64),
            source_controller=str(args.controller_type),
            source_checkpoint=str(args.checkpoint_path),
            replay_mode="derived_post_step_from_muscle_rollout",
            metadata={
                "derived_from": str(muscle_path),
                "note": "post-step states from the same muscle rollout; qpos[k] == muscle.qpos[k+1]",
            },
        )
        joint_path = out_dir / "aligned_joint_trajectory.npz"
        save_joint_trajectory(joint_path, joint_traj)

        meta = {
            "motion_path": args.motion_path,
            "controller_type": args.controller_type,
            "checkpoint_path": args.checkpoint_path,
            "aligned_muscle_trajectory_npz": str(muscle_path),
            "aligned_joint_trajectory_npz": str(joint_path),
            "left_leg_joint_reference_npz": str(left_ref_path),
            "n_muscle_frames": int(muscle_traj.n_frames),
            "n_hybrid_steps": int(muscle_traj.n_frames - 1),
            "dt": float(muscle_traj.dt),
            "episode_return": float(episode_return),
            "done_count": int(done_count),
            "alignment": muscle_traj.metadata.get("alignment"),
        }
        write_replay_meta(out_dir / "capture_meta.json", meta)
        print(f"Saved muscle reference: {muscle_path} ({muscle_traj.n_frames} frames)")
        print(f"Saved joint reference: {joint_path} ({joint_traj.n_frames} post-step frames)")
        print(f"Saved left-leg slice: {left_ref_path}")
        print(f"  return={episode_return:.3f} done_count={done_count}")
    finally:
        env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
