#!/usr/bin/env python
"""Record distilled prosthesis policy replay (335 muscles + 4-DOF prosthesis)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import apply_prosthesis_overrides, parse_optional_vec4
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless

DEFAULT_MOTION = "KIT/3/walk_6m_straight_line04_poses"
DEFAULT_CHECKPOINT = (
    "/home/user/Workspace/musclemimic/musclemimic/outputs/prosthesis_distill/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "/home/user/Workspace/musclemimic/musclemimic/outputs/distill_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--motion-path", default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-steps", type=int, default=0, help="0 = full trajectory length")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--cam-distance", type=float, default=6.0)
    parser.add_argument("--cam-elevation", type=float, default=-20.0)
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--prosthesis-action-type", default=None, choices=[None, "torque", "pd_residual_torque"])
    parser.add_argument("--residual-pd-kp", default=None, help="knee,ankle,subtalar,mtp")
    parser.add_argument("--residual-pd-kd", default=None, help="knee,ankle,subtalar,mtp")
    parser.add_argument("--torque-slew-limit", default=None, help="scalar or knee,ankle,subtalar,mtp Nm/step")
    parser.add_argument("--tau-lowpass-alpha", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config, agent_state, metadata = load_checkpoint(args.checkpoint)
    OmegaConf.set_struct(config, False)
    slew = None
    if args.torque_slew_limit:
        parts = [float(x.strip()) for x in str(args.torque_slew_limit).split(",") if x.strip()]
        slew = tuple(parts) if len(parts) == 4 else float(parts[0])
    apply_prosthesis_overrides(
        config,
        action_type=args.prosthesis_action_type,
        residual_pd_kp=parse_optional_vec4(args.residual_pd_kp),
        residual_pd_kd=parse_optional_vec4(args.residual_pd_kd),
        torque_slew_limit=slew,
        tau_lowpass_alpha=args.tau_lowpass_alpha,
    )
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBodyProsthesisEnv"
    env_params["headless"] = True
    prosthesis = dict(env_params.get("prosthesis", {}) or {})
    prosthesis["enabled"] = True
    prosthesis["control_mode"] = "train_policy"
    env_params["prosthesis"] = prosthesis
    prosthesis_cfg = prosthesis
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [args.motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass

    control_dt = apply_temporal_params(config)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**{**env_params, **task_params})

    n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    runner = PolicyRunner.from_agent_state(
        agent_conf,
        agent_state,
        env,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    safe_motion = args.motion_path.replace("/", "_")
    raw_path = out_dir / f".{safe_motion}_raw.mp4"
    video_path = web_mp4_path(out_dir / f"{safe_motion}_distill_replay.mp4")
    fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))

    obs = env.reset()
    obs_policy = runner.reset_obs(obs)
    episode_return = 0.0
    done_count = 0
    tau_log: list[np.ndarray] = []

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prosthesis execution: residual_pd={prosthesis_cfg.get('residual_pd')} execution={prosthesis_cfg.get('execution')}")
    print(f"Motion: {args.motion_path} | steps={n_steps} | fps={fps} | duration≈{n_steps * control_dt:.2f}s")
    print(f"Recording -> {video_path}")

    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.cam_distance
    cam.elevation = args.cam_elevation
    cam.azimuth = args.cam_azimuth

    with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
        for step in range(n_steps):
            action, _value = runner.act(obs_policy)
            obs, reward, _absorbing, done, info = env.step(action)
            obs_policy = runner.update_obs(obs)
            episode_return += float(np.asarray(reward).item())
            if bool(done):
                done_count += 1
            tau_log.append(np.asarray(info.get("prosthesis_tau", np.zeros(4)), dtype=np.float32))
            cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            renderer.update_scene(env.data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()
    env.stop()
    encode_web_mp4(raw_path, video_path)

    tau_arr = np.stack(tau_log, axis=0) if tau_log else np.zeros((0, 4), dtype=np.float32)
    tau_stats = {}
    if tau_arr.size:
        names = ["knee", "ankle", "subtalar", "mtp"]
        for i, name in enumerate(names):
            col = tau_arr[:, i]
            tau_stats[f"{name}_tau_std"] = float(np.std(col))
            tau_stats[f"{name}_tau_abs_mean"] = float(np.mean(np.abs(col)))

    meta = {
        "checkpoint": str(args.checkpoint),
        "motion_path": args.motion_path,
        "n_steps": n_steps,
        "fps": fps,
        "duration_s": float(n_steps * control_dt),
        "episode_return": episode_return,
        "done_count": done_count,
        "prosthesis_residual_pd": prosthesis_cfg.get("residual_pd"),
        "prosthesis_execution": prosthesis_cfg.get("execution"),
        "prosthesis_tau_stats": tau_stats,
        "video": str(video_path),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "distill_metadata": metadata,
    }
    (out_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Saved video: {video_path}")
    print(f"Saved meta: {out_dir / 'replay_meta.json'}")
    print(f"episode_return={episode_return:.3f} done_count={done_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
