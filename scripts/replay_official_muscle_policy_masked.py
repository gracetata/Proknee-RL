#!/usr/bin/env python
"""Closed-loop official muscle policy with selected muscles disabled."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import repo_root
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.joint_replay import write_replay_meta
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.muscle_replay import (
    actuator_ids_for_muscle_names,
    color_disabled_muscle_tendons_blue,
    record_muscle_closed_loop_masked_rollout,
    save_muscle_trajectory,
)
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless
from loco_mujoco.task_factories import TaskFactory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", required=True)
    p.add_argument(
        "--mask-preset",
        default="foot1",
        choices=sorted(MUSCLE_MASK_PRESETS),
        help="Disabled muscle preset; remaining muscles are driven by the official policy.",
    )
    p.add_argument(
        "--checkpoint_path",
        default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2",
    )
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=0, help="0 = full trajectory length")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--cam-distance", type=float, default=6.0)
    p.add_argument("--cam-elevation", type=float, default=-20.0)
    p.add_argument("--cam-azimuth", type=float, default=90.0)
    p.add_argument("--skip_video", action="store_true")
    return p.parse_args()


def default_output_dir(controller_type: str, mask_preset: str, motion_path: str) -> Path:
    return (
        repo_root()
        / "outputs"
        / "replay"
        / "muscle_mimic"
        / controller_type
        / mask_preset
        / safe_motion_name(motion_path)
    )


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

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        args.controller_type, args.mask_preset, args.motion_path
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    disabled_names = MUSCLE_MASK_PRESETS[args.mask_preset]
    config, agent_state, metadata = load_checkpoint(args.checkpoint_path)
    OmegaConf.set_struct(config, False)
    control_dt = apply_temporal_params(config)
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

        disabled_actuators = actuator_ids_for_muscle_names(env.model, disabled_names)
        highlighted = color_disabled_muscle_tendons_blue(env.model, disabled_names)
        print(
            f"[masked muscle policy] {args.motion_path} preset={args.mask_preset} -> {out_dir}\n"
            f"disabled_muscles={list(disabled_names)} actuator_ids={disabled_actuators.tolist()} "
            f"highlighted_tendons={highlighted}"
        )

        n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
        fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
        raw_path = out_dir / f".muscle_policy_{args.mask_preset}_closed_loop_raw.mp4"
        video_path = web_mp4_path(out_dir / f"muscle_policy_{args.mask_preset}_closed_loop.mp4")
        traj_path = out_dir / f"muscle_policy_{args.mask_preset}_closed_loop_trajectory.npz"

        traj, episode_return, done_count = record_muscle_closed_loop_masked_rollout(
            env=env,
            policy=runner,
            disabled_actuators=disabled_actuators,
            n_steps=n_steps,
            width=int(args.width),
            height=int(args.height),
            fps=fps,
            cam_distance=float(args.cam_distance),
            cam_elevation=float(args.cam_elevation),
            cam_azimuth=float(args.cam_azimuth),
            record_path=None if args.skip_video else raw_path,
        )
        traj.motion_path = args.motion_path
        traj.source_controller = str(args.controller_type)
        traj.source_checkpoint = str(args.checkpoint_path)
        save_muscle_trajectory(traj_path, traj)

        if not args.skip_video:
            encode_web_mp4(raw_path, video_path, remove_src=True)
            print(f"Saved masked closed-loop video: {video_path}")

        meta = {
            "motion_path": args.motion_path,
            "controller_type": args.controller_type,
            "checkpoint_path": args.checkpoint_path,
            "replay_mode": f"muscle_closed_loop_{args.mask_preset}",
            "description": (
                "Official full-muscle policy closed-loop rollout with selected disabled muscles "
                "zeroed each step; remaining muscles are policy-controlled."
            ),
            "mask_preset": args.mask_preset,
            "disabled_muscle_names": list(disabled_names),
            "disabled_actuator_ids": disabled_actuators.tolist(),
            "highlighted_disabled_muscle_tendons": int(highlighted),
            "muscle_trajectory_npz": str(traj_path),
            "muscle_replay_video": str(video_path) if not args.skip_video else None,
            "n_frames": int(traj.n_frames),
            "duration_s": float(traj.duration_s),
            "dt": float(traj.dt),
            "policy_action_dim": int(traj.policy_action.shape[-1]),
            "actuator_ctrl_dim": int(traj.actuator_ctrl.shape[-1]),
            "episode_return": float(episode_return),
            "done_count": int(done_count),
            "distill_metadata": metadata,
        }
        write_replay_meta(out_dir / "replay_meta.json", meta)
        print(f"  frames={traj.n_frames} return={episode_return:.3f} done_count={done_count}")
        return 0
    finally:
        env.stop()


if __name__ == "__main__":
    raise SystemExit(main())
