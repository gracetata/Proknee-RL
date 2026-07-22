#!/usr/bin/env python3
"""Local-only MuJoCo viewer for replay baseline or a trained prosthesis policy."""

from __future__ import annotations

import argparse
import json
import os
import time

import jax.numpy as jnp
import numpy as np

import _bootstrap  # noqa: F401

from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.ppo import load_policy_checkpoint, load_training_config
from torque_replay_training.replay_env import TorqueReplayEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, nargs="+")
    parser.add_argument("--policy", help="Omit to visualize the zero-residual healthy baseline")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true", help="Step without opening a GUI")
    args = parser.parse_args()
    if not args.check_only and not os.environ.get("DISPLAY"):
        raise SystemExit("MuJoCo GUI requires a local DISPLAY; use --check-only on headless machines")
    if args.steps <= 0 or args.realtime_factor <= 0:
        raise SystemExit("--steps and --realtime-factor must be positive")

    _ppo_config, replay_config = load_training_config(args.config)
    model = params = metadata = None
    if args.policy:
        model, params, metadata = load_policy_checkpoint(args.policy)

    with TorqueReplayEnv(args.dataset, args.checkpoint, replay_config) as env:
        obs, _ = env.reset(seed=0, start_step=0, dataset_index=args.dataset_index)
        if metadata and int(metadata["observation_size"]) != env.observation_size:
            raise ValueError("policy/environment observation sizes differ")

        viewer = None
        if not args.check_only:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -15.0

        rewards = []
        fell = False
        length = 0
        try:
            while length < args.steps and (viewer is None or viewer.is_running()):
                started = time.perf_counter()
                if model is None:
                    action = np.zeros(env.action_size, dtype=np.float32)
                else:
                    mean, _log_std, _value = model.apply({"params": params}, jnp.asarray(obs)[None, :])
                    action = np.asarray(jnp.tanh(mean[0]), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                length += 1
                if viewer is not None:
                    viewer.cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                    viewer.sync()
                    delay = env.dataset.dt_control / args.realtime_factor - (time.perf_counter() - started)
                    if delay > 0:
                        time.sleep(delay)
                if terminated or truncated:
                    fell = bool(info["fell"])
                    break
        finally:
            if viewer is not None:
                viewer.close()

    print(
        json.dumps(
            {
                "mode": "policy" if args.policy else "healthy_baseline",
                "steps": length,
                "fell": fell,
                "return": float(np.sum(rewards)),
                "check_only": args.check_only,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
