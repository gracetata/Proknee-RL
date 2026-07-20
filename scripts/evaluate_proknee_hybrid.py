#!/usr/bin/env python
"""Evaluate Stage 1/2 MuscleMimic-ProKnee checkpoints on oracle rollouts."""

from __future__ import annotations

import argparse
import pickle

import jax
import jax.numpy as jnp

from musclemimic.proknee import MuscleProKneeHybridEnv
from musclemimic.proknee.models import MuscleProKneeStudent, MuscleProKneeTeacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--policy", required=True, help="Stage1 or Stage2 .pt checkpoint")
    parser.add_argument("--kind", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--dataset-group", default="KIT_KINESIS_TESTING_MOTIONS")
    parser.add_argument("--motion-path", nargs="*", default=None, help="Optional explicit reference paths for smoke tests")
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> int:
    args = parse_args()
    ckpt = load_pickle(args.policy)
    env = MuscleProKneeHybridEnv(
        args.checkpoint,
        dataset_group=args.dataset_group,
        rel_dataset_path=args.motion_path,
        history_len=int(ckpt.get("history_len", 30)),
        deterministic_oracle=False,
        apply_teacher_action=False,
        target_mode=str(ckpt.get("target_mode", "residual")),
    )
    data = env.reset()
    if args.kind == "stage1":
        model = MuscleProKneeTeacher(action_dim=int(ckpt["action_dim"]))

        @jax.jit
        def predict(params, obs, priv, hist):
            del hist
            pred, _ = model.apply(params, obs, priv)
            return pred

    else:
        model = MuscleProKneeStudent(action_dim=int(ckpt["action_dim"]))

        @jax.jit
        def predict(params, obs, priv, hist):
            pred, _ = model.apply(params, obs, priv, hist, mode="student")
            return pred

    mse_sum = 0.0
    done_count = 0
    for _step in range(1, args.steps + 1):
        obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
        priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
        hist = jnp.asarray(data.proprio_hist[None, :, :], dtype=jnp.float32)
        target = jnp.asarray(data.oracle_target[None, :], dtype=jnp.float32)
        pred = predict(ckpt["params"], obs, priv, hist)
        mse_sum += float(jnp.mean(jnp.square(pred - target)))
        data = env.step()
        if data.done:
            done_count += 1
            data = env.reset()
    print(f"steps={args.steps} target_mse={mse_sum / max(args.steps, 1):.6f} resets={done_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
