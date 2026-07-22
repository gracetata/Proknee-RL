#!/usr/bin/env python3
"""Headless web-mp4 recording for torque-replay baseline (zero residual)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import _bootstrap  # noqa: F401

from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.ppo import load_training_config
from torque_replay_training.replay_env import TorqueReplayEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="Output .mp4 path (will become *_web.mp4)")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cam-distance", type=float, default=3.5)
    parser.add_argument("--cam-elevation", type=float, default=0.0)
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument(
        "--exact-baseline",
        action="store_true",
        help="Pass recorded prosthesis torque through without clip (validation-style)",
    )
    parser.add_argument(
        "--ignore-fall",
        action="store_true",
        help="Keep stepping for the full clip even after a fall is detected",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ppo_config, replay_config = load_training_config(args.config)
    replay_config = replace(
        replay_config,
        episode_steps=10**9,
        random_start=False,
        exact_baseline=bool(args.exact_baseline),
    )
    if args.ignore_fall:
        replay_config = replace(replay_config, fall_height=-1e6, fall_up_z=-1e6)

    output = web_mp4_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.parent / f".{output.stem}_raw.mp4"

    with TorqueReplayEnv(args.dataset, args.checkpoint, replay_config, seed=0) as env:
        obs, info = env.reset(seed=0, start_step=0, dataset_index=0)
        n_steps = int(env.dataset.n_steps)
        fps = int(round(1.0 / float(env.dataset.dt_control)))
        renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = args.cam_distance
        cam.elevation = args.cam_elevation
        cam.azimuth = args.cam_azimuth

        meta = {
            "dataset": str(Path(args.dataset).resolve()),
            "motion_path": env.dataset.motion_path,
            "n_steps": n_steps,
            "fps": fps,
            "replay_mode": replay_config.replay_mode,
            "exact_baseline": replay_config.exact_baseline,
            "healthy_kp": replay_config.healthy_kp,
            "healthy_kd": replay_config.healthy_kd,
            "ignore_fall": bool(args.ignore_fall),
            "fell": False,
            "fell_step": None,
            "video": str(output),
        }
        print(
            json.dumps(
                {
                    "status": "recording",
                    "motion": meta["motion_path"],
                    "steps": n_steps,
                    "fps": fps,
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
            cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            renderer.update_scene(env.data, camera=cam)
            writer.append_data(renderer.render())
            for step in range(n_steps):
                obs, _reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float64))
                if info.get("fell") and meta["fell_step"] is None:
                    meta["fell"] = True
                    meta["fell_step"] = int(step + 1)
                cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())
                if (step + 1) % 200 == 0:
                    print(
                        f"step={step + 1}/{n_steps} height={info.get('root_height'):.3f} "
                        f"fell={info.get('fell')}",
                        flush=True,
                    )
                if (terminated or truncated) and not args.ignore_fall:
                    break

        renderer.close()

    encode_web_mp4(raw_path, output)
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "saved", **meta}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
