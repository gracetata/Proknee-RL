#!/usr/bin/env python
"""Record replay for masked-observation full-muscle distilled policy."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.obs_mask import MaskedObservationEnvView, MaskedObsSpec, apply_obs_mask, build_masked_obs_spec
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless

DEFAULT_MOTION = "KIT/3/walk_6m_straight_line04_poses"
DEFAULT_CHECKPOINT = (
    "/home/user/Workspace/musclemimic/musclemimic/outputs/full_muscle_masked_obs_distill/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "/home/user/Workspace/musclemimic/musclemimic/outputs/full_muscle_masked_obs_replay"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--motion-path", default=DEFAULT_MOTION)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-steps", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


def find_mask_spec(checkpoint: Path, env) -> MaskedObsSpec:
    candidates = [
        checkpoint / "masked_obs_spec.json",
        checkpoint.parent / "masked_obs_spec.json",
        checkpoint.parent.parent / "masked_obs_spec.json",
        checkpoint.parent / "distilled_metadata.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "masked_obs_spec.json":
            return MaskedObsSpec.load_json(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_payload = payload.get("masked_obs_spec")
        if spec_payload:
            return MaskedObsSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec_payload.items()})
    return build_masked_obs_spec(env)


def main() -> int:
    args = parse_args()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    config, agent_state, metadata = load_checkpoint(str(checkpoint))
    OmegaConf.set_struct(config, False)
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
    amass["rel_dataset_path"] = [args.motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass
    control_dt = apply_temporal_params(config)
    env = TaskFactory.get_factory_cls(config.experiment.task_factory.name).make(**{**env_params, **task_params})
    spec = find_mask_spec(checkpoint, env)
    env_view = MaskedObservationEnvView(env, spec)
    agent_conf = PPOJax.init_agent_conf(env_view, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    runner = PolicyRunner.from_agent_state(
        agent_conf,
        agent_state,
        env_view,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
    fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
    safe_motion = args.motion_path.replace("/", "_")
    raw_path = out_dir / f".{safe_motion}_raw.mp4"
    video_path = web_mp4_path(out_dir / f"{safe_motion}_masked_obs_full_muscle_replay.mp4")
    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 6.0
    cam.elevation = -20.0
    cam.azimuth = 90.0
    obs = env.reset()
    obs_policy = runner.reset_obs(apply_obs_mask(obs, spec))
    episode_return = 0.0
    done_count = 0
    with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
        for _step in range(n_steps):
            action, _value = runner.act(obs_policy)
            obs, reward, _absorbing, done, _info = env.step(action)
            obs_policy = runner.update_obs(apply_obs_mask(obs, spec))
            episode_return += float(np.asarray(reward).item())
            done_count += int(bool(done))
            cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            renderer.update_scene(env.data, camera=cam)
            writer.append_data(renderer.render())
    renderer.close()
    env.stop()
    encode_web_mp4(raw_path, video_path)
    meta = {
        "checkpoint": str(checkpoint),
        "motion_path": args.motion_path,
        "n_steps": n_steps,
        "fps": fps,
        "duration_s": float(n_steps * control_dt),
        "episode_return": episode_return,
        "done_count": done_count,
        "video": str(video_path),
        "masked_obs_dim": spec.masked_obs_dim,
        "raw_obs_dim": spec.raw_obs_dim,
        "distill_metadata": metadata,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Saved video: {video_path}")
    print(f"episode_return={episode_return:.3f} done_count={done_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
