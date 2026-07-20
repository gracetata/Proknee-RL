#!/usr/bin/env python
"""Smoke tests for the MyoFullBody prosthesis environment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np

import musclemimic.environments  # noqa: F401 - registers envs
from loco_mujoco.task_factories import RLFactory
from musclemimic.prosthesis.controllers import ReferencePDProsthesisController


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", default="/home/user/Workspace/musclemimic/outputs/prosthesis_debug")
    parser.add_argument("--mjx", action="store_true", help="Also run a minimal MJX reset/step smoke")
    return parser.parse_args()


def run_cpu(mode: str, steps: int, out_dir: Path):
    controller = ReferencePDProsthesisController() if mode == "eval_external_controller" else None
    env = RLFactory.make(
        "MyoFullBodyProsthesisEnv",
        disable_fingers=True,
        reward_type="NoReward",
        prosthesis={"enabled": True, "control_mode": mode},
        prosthesis_controller=controller,
    )
    obs = env.reset()
    assert obs.shape[0] == env.info.observation_space.shape[0]

    low = np.asarray(env.info.action_space.low, dtype=np.float32)
    high = np.asarray(env.info.action_space.high, dtype=np.float32)
    rows = []
    rng = np.random.default_rng(0)
    for step in range(steps):
        action = rng.uniform(low, high).astype(np.float32)
        obs, reward, absorbing, done, info = env.step(action)
        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"NaN/Inf observation at step {step}")
        disabled_norm = float(info.get("disabled_muscle_ctrl_norm", np.nan))
        if disabled_norm > 1e-8:
            raise RuntimeError(f"disabled ctrl not zero at step {step}: {disabled_norm}")
        tau = np.asarray(info.get("prosthesis_tau"), dtype=np.float32)
        if tau.shape != (4,) or not np.all(np.isfinite(tau)):
            raise RuntimeError(f"Invalid prosthesis tau at step {step}: {tau}")
        rows.append(
            {
                "mode": mode,
                "step": step,
                "reward": float(reward),
                "absorbing": bool(absorbing),
                "done": bool(done),
                "disabled_ctrl_norm": disabled_norm,
                "disabled_force_norm": float(info.get("disabled_muscle_force_norm", 0.0)),
                "tau_knee": float(tau[0]),
                "tau_ankle": float(tau[1]),
                "tau_subtalar": float(tau[2]),
                "tau_mtp": float(tau[3]),
            }
        )
        if done:
            obs = env.reset()
    csv_path = out_dir / f"smoke_{mode}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{mode}: OK action_dim={env.info.action_space.shape[0]} obs_dim={env.info.observation_space.shape[0]} log={csv_path}")


def run_mjx():
    env = RLFactory.make(
        "MjxMyoFullBodyProsthesisEnv",
        disable_fingers=True,
        reward_type="NoReward",
        mjx_backend="jax",
        num_envs=1,
        prosthesis={"enabled": True, "control_mode": "train_policy"},
    )
    state = env.mjx_reset(jax.random.key(0))
    action = np.zeros(env.info.action_space.shape[0], dtype=np.float32)
    state = env.mjx_step(state, action)
    obs = np.asarray(state.observation)
    if not np.all(np.isfinite(obs)):
        raise RuntimeError("MJX smoke produced NaN/Inf observation")
    print(f"mjx: OK action_dim={env.info.action_space.shape[0]} obs_dim={env.info.observation_space.shape[0]}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cpu("train_policy", args.steps, out_dir)
    run_cpu("eval_external_controller", args.steps, out_dir)
    if args.mjx:
        run_mjx()


if __name__ == "__main__":
    main()
