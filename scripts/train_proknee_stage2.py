#!/usr/bin/env python
"""Stage 2 ProprioAdapt distillation for MuscleMimic-ProKnee."""

from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState

from musclemimic.proknee import MuscleProKneeHybridEnv
from musclemimic.proknee.models import MuscleProKneeStudent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--dataset-group", default="KIT_KINESIS_TRAINING_MOTIONS")
    parser.add_argument("--motion-path", nargs="*", default=None, help="Optional explicit reference paths for smoke tests")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs/proknee_stage2")
    parser.add_argument("--save-interval", type=int, default=1000)
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(path: str, state: TrainState, meta: dict):
    with open(path, "wb") as f:
        pickle.dump({**meta, "params": jax.device_get(state.params)}, f)


def merge_stage1_params(student_params, stage1_params):
    dst = unfreeze(student_params)
    src = unfreeze(stage1_params)
    for name in ("priv_mlp", "actor"):
        if name in src["params"]:
            dst["params"][name] = src["params"][name]
    return freeze(dst)


def zero_frozen_grads(grads):
    grad_dict = unfreeze(grads)
    for name in ("priv_mlp", "actor"):
        if name in grad_dict["params"]:
            grad_dict["params"][name] = jax.tree.map(jnp.zeros_like, grad_dict["params"][name])
    return freeze(grad_dict)


def tuple_from_checkpoint(ckpt: dict, key: str, default: tuple[float, ...] | None) -> tuple[float, ...] | None:
    value = ckpt.get(key, default)
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return tuple([float(value)] * 4)
    return tuple(float(x) for x in value)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = load_pickle(args.stage1)
    target_mode = str(ckpt.get("target_mode", "residual"))
    replacement_enabled = bool(ckpt.get("dagger_pd_override", False)) and target_mode == "qpos"
    env = MuscleProKneeHybridEnv(
        args.checkpoint,
        dataset_group=args.dataset_group,
        rel_dataset_path=args.motion_path,
        history_len=int(ckpt.get("history_len", 30)),
        deterministic_oracle=False,
        apply_teacher_action=replacement_enabled,
        target_mode=target_mode,
        pd_override=replacement_enabled,
        pd_kp=tuple_from_checkpoint(ckpt, "pd_kp", (300.0, 200.0, 200.0, 120.0)),
        pd_kd=tuple_from_checkpoint(ckpt, "pd_kd", (30.0, 20.0, 20.0, 12.0)),
        pd_torque_limit=tuple_from_checkpoint(ckpt, "pd_torque_limit", (120.0, 120.0, 120.0, 120.0)),
        pd_torque_slew_limit=tuple_from_checkpoint(ckpt, "pd_torque_slew_limit", None),
        oracle_torque_ff_scale=float(ckpt.get("oracle_torque_ff_scale", 0.0)),
        oracle_torque_ff_limit=tuple_from_checkpoint(ckpt, "oracle_torque_ff_limit", None),
        pd_gain_scale=float(ckpt.get("pd_gain_scale", 1.0)),
        max_prosthesis_qpos_step=tuple_from_checkpoint(ckpt, "max_prosthesis_qpos_step", None),
        prosthesis_muscle_scale=float(ckpt.get("prosthesis_muscle_scale", 1.0)),
    )
    data = env.reset()
    model = MuscleProKneeStudent(action_dim=int(ckpt["action_dim"]))
    params = model.init(
        jax.random.key(1),
        jnp.zeros((1, int(ckpt["obs_dim"])), dtype=jnp.float32),
        jnp.zeros((1, int(ckpt["priv_dim"])), dtype=jnp.float32),
        jnp.zeros((1, env.spec.history_len, env.spec.proprio_dim), dtype=jnp.float32),
        mode="student",
    )
    params = merge_stage1_params(params, ckpt["params"])
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(args.lr))

    @jax.jit
    def train_step(state, obs, priv, hist):
        def loss_fn(params):
            _teacher_action, teacher_latent = state.apply_fn(params, obs, priv, hist, mode="teacher")
            _student_action, student_latent = state.apply_fn(params, obs, priv, hist, mode="student")
            return jnp.mean(jnp.square(student_latent - jax.lax.stop_gradient(teacher_latent)))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = zero_frozen_grads(grads)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def predict_teacher_action(params, obs, priv, hist):
        action, _latent = state.apply_fn(params, obs, priv, hist, mode="teacher")
        return action

    print("Stage2 dimensions:")
    print(f"  obs_dim={ckpt['obs_dim']}, priv_dim={ckpt['priv_dim']}, proprio_dim={env.spec.proprio_dim}")
    print(
        "  stage1_env="
        f"target_mode={target_mode}, replacement_enabled={replacement_enabled}, "
        f"prosthesis_muscle_scale={env.prosthesis_muscle_scale}, "
        f"pd_kp={env.pd_kp.tolist()}, pd_kd={env.pd_kd.tolist()}, "
        f"pd_torque_limit={env.pd_torque_limit}, pd_gain_scale={env.pd_gain_scale}, "
        f"max_prosthesis_qpos_step={env.max_prosthesis_qpos_step}",
        flush=True,
    )
    print(f"  output_dir={args.output_dir}")

    meta = {
        **ckpt,
        "stage1": args.stage1,
        "stage2_replacement_aligned": replacement_enabled,
        "stage2_env_target_mode": target_mode,
        "stage2_env_prosthesis_muscle_scale": float(env.prosthesis_muscle_scale),
    }
    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
        priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
        hist = jnp.asarray(data.proprio_hist[None, :, :], dtype=jnp.float32)
        teacher_action = None
        if replacement_enabled:
            teacher_action = jax.device_get(predict_teacher_action(state.params, obs, priv, hist))[0]
        state, loss = train_step(state, obs, priv, hist)

        running += float(loss)
        data = env.step(teacher_action)
        if data.done:
            data = env.reset()

        if step % 100 == 0:
            print(f"step={step:07d} latent_mse={running/100:.6f} fps={step/(time.time()-t0+1e-6):.1f}", flush=True)
            running = 0.0
        if step % args.save_interval == 0:
            path = os.path.join(args.output_dir, f"stage2_step_{step}.pt")
            save_checkpoint(path, state, {**meta, "step": step})
            print(f"saved {path}")

    final_path = os.path.join(args.output_dir, "stage2_final.pt")
    save_checkpoint(final_path, state, {**meta, "step": args.steps})
    print(f"saved {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
