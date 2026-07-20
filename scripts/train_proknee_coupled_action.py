#!/usr/bin/env python
"""Closed-loop action fine-tuning for MuscleMimic-ProKnee.

This is a separate experimental route.  It does not replace the existing
qpos+torque-residual Stage1 trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState

from musclemimic.distill.config import motion_list_from_group
from musclemimic.proknee import MuscleProKneeCoupledActionEnv
from musclemimic.proknee.action_models import CoupledActionPolicy


@dataclass(frozen=True)
class CoupledActionMeta:
    step: int
    checkpoint: str
    dataset_group: str
    val_dataset_group: str
    motion_path: list[str] | None
    val_motion_path: list[str] | None
    obs_dim: int
    priv_dim: int
    prosthesis_action_dim: int
    body_residual_dim: int
    body_residual_adapter: bool
    action_mode: str
    action_torque_limit: tuple[float, ...]
    action_torque_slew_limit: tuple[float, ...] | None
    prosthesis_muscle_scale: float
    body_residual_scale: float
    mask_preset: str
    obs_mode: str
    discount: float
    clip_eps: float
    best_eval_score: float | None


def parse_args():
    parser = argparse.ArgumentParser(description="Train closed-loop prosthesis action policy")
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--dataset-group", default="KIT_KINESIS_TRAINING_MOTIONS")
    parser.add_argument("--val-dataset-group", default="KIT_KINESIS_TESTING_MOTIONS")
    parser.add_argument("--motion-path", nargs="*", default=None)
    parser.add_argument("--val-motion-path", nargs="*", default=None)
    parser.add_argument("--warm-start", default=None, help="Optional Stage1 or coupled-action checkpoint")
    parser.add_argument(
        "--warm-start-prosthesis-head",
        choices=["auto", "none"],
        default="auto",
        help="When warm-starting from Stage1, map torque_actor -> prosthesis_actor.",
    )
    parser.add_argument("--output-dir", default="/home/user/Workspace/musclemimic/outputs/proknee_coupled_action")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--action-mode", choices=["torque", "impedance_residual"], default="torque")
    parser.add_argument("--action-torque-limit-list", default="110,65,45,22")
    parser.add_argument("--action-torque-slew-limit-list", default="10,5,3,1.5")
    parser.add_argument("--prosthesis-muscle-scale", type=float, default=0.0)
    parser.add_argument(
        "--mask-preset",
        default="strict19",
        help="Disabled muscle preset when prosthesis_muscle_scale=0 (use knee15 for knee15 experiments).",
    )
    parser.add_argument(
        "--obs-mode",
        choices=["easy", "pose_only"],
        default="easy",
        help="easy keeps ctrl summary + priv ctrl; pose_only removes muscle/ctrl inputs.",
    )
    parser.add_argument("--body-residual-adapter", action="store_true")
    parser.add_argument("--body-residual-scale", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=2048)
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="TensorBoard log directory (default: <output-dir>/tensorboard).",
    )
    parser.add_argument("--disable-tensorboard", action="store_true")
    return parser.parse_args()


class TensorboardLogger:
    def __init__(self, log_dir: str | None):
        self.writer = None
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        try:
            from tensorboardX import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requires tensorboardX. Install with `uv add tensorboardX` "
                "or pass --disable-tensorboard."
            ) from exc
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging enabled at: {log_dir}", flush=True)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), int(step))

    def add_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for key, value in values.items():
            self.add_scalar(f"{prefix}/{key}", value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def count_motions(args, *, validation: bool = False) -> int | None:
    rel_paths = args.val_motion_path if validation else args.motion_path
    if rel_paths:
        return len(rel_paths)
    group = args.val_dataset_group if validation else args.dataset_group
    if not group:
        return None
    return len(motion_list_from_group(group))


def parse_vec(text: str | None, n: int) -> tuple[float, ...] | None:
    if text is None or str(text).lower() == "none":
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(vals)}")
    return tuple(vals)


def make_env(args, *, validation: bool = False) -> MuscleProKneeCoupledActionEnv:
    return MuscleProKneeCoupledActionEnv(
        args.checkpoint,
        dataset_group=args.val_dataset_group if validation else args.dataset_group,
        rel_dataset_path=args.val_motion_path if validation else args.motion_path,
        deterministic_oracle=True,
        action_mode=args.action_mode,
        action_torque_limit=parse_vec(args.action_torque_limit_list, 4),
        action_torque_slew_limit=parse_vec(args.action_torque_slew_limit_list, 4),
        prosthesis_muscle_scale=args.prosthesis_muscle_scale,
        body_residual_scale=args.body_residual_scale if args.body_residual_adapter else 0.0,
        mask_preset=args.mask_preset,
        obs_mode=args.obs_mode,
    )


def merge_compatible_params(initialized_params, checkpoint_params):
    def merge(dst, src):
        if isinstance(dst, dict) and isinstance(src, dict):
            out = dict(dst)
            for key, value in src.items():
                if key in out:
                    out[key] = merge(out[key], value)
            return out
        if getattr(dst, "shape", None) == getattr(src, "shape", None):
            return src
        return dst

    return freeze(merge(unfreeze(initialized_params), unfreeze(checkpoint_params)))


def transplant_stage1_torque_actor(initialized_params, checkpoint_params):
    """Map a Stage1 torque head into the coupled action prosthesis actor.

    Stage1 checkpoints store the physical feedforward-torque head as
    ``params/torque_actor``.  CoupledActionPolicy stores the normalized action
    head as ``params/prosthesis_actor`` but internally divides the raw output by
    ``action_scale`` before tanh, so copying the torque head gives a meaningful
    initial action instead of a random saturated policy.
    """

    out = unfreeze(initialized_params)
    src = unfreeze(checkpoint_params)
    out_params = out.get("params", out)
    src_params = src.get("params", src)
    if "prosthesis_actor" not in out_params or "torque_actor" not in src_params:
        return freeze(out)

    def compatible(dst, src_value):
        if isinstance(dst, dict) and isinstance(src_value, dict):
            return all(key in dst and compatible(dst[key], value) for key, value in src_value.items())
        return getattr(dst, "shape", None) == getattr(src_value, "shape", None)

    if compatible(out_params["prosthesis_actor"], src_params["torque_actor"]):
        out_params["prosthesis_actor"] = src_params["torque_actor"]
        print("[coupled_action] warm-start: mapped Stage1 torque_actor -> prosthesis_actor", flush=True)
    else:
        print("[coupled_action] warm-start: skipped torque_actor mapping due to shape mismatch", flush=True)
    return freeze(out)


def init_state(model, env, args) -> TrainState:
    params = model.init(
        jax.random.key(args.seed),
        jnp.zeros((1, env.spec.obs_dim), dtype=jnp.float32),
        jnp.zeros((1, env.spec.priv_dim), dtype=jnp.float32),
        body_residual=bool(args.body_residual_adapter),
    )
    if args.warm_start:
        with open(args.warm_start, "rb") as f:
            ckpt = pickle.load(f)
        params = merge_compatible_params(params, ckpt["params"])
        if args.warm_start_prosthesis_head == "auto":
            params = transplant_stage1_torque_actor(params, ckpt["params"])
    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def gaussian_log_prob(action, mean, log_std):
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(jnp.square((action - mean) / std) + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)))


def split_actions(action: np.ndarray, prosthesis_dim: int, body_dim: int):
    prosthesis = action[:prosthesis_dim]
    body = None if body_dim == 0 else action[prosthesis_dim : prosthesis_dim + body_dim]
    return prosthesis, body


def build_sample(model, *, body_residual: bool):
    @jax.jit
    def sample(params, obs, priv, rng):
        out = model.apply(params, obs, priv, body_residual=body_residual)
        rng_p, rng_b = jax.random.split(rng)
        p_std = jnp.exp(out["prosthesis_log_std"])
        p_action = out["prosthesis_mean"] + p_std * jax.random.normal(rng_p, out["prosthesis_mean"].shape)
        p_logp = gaussian_log_prob(p_action, out["prosthesis_mean"], out["prosthesis_log_std"])
        if body_residual and out["body_mean"].shape[-1] > 0:
            b_std = jnp.exp(out["body_log_std"])
            b_action = out["body_mean"] + b_std * jax.random.normal(rng_b, out["body_mean"].shape)
            b_logp = gaussian_log_prob(b_action, out["body_mean"], out["body_log_std"])
            action = jnp.concatenate([p_action, b_action], axis=-1)
            logp = p_logp + b_logp
        else:
            action = p_action
            logp = p_logp
        return action, logp

    return sample


def build_mean_action(model, *, body_residual: bool):
    @jax.jit
    def mean_action(params, obs, priv):
        out = model.apply(params, obs, priv, body_residual=body_residual)
        if body_residual and out["body_mean"].shape[-1] > 0:
            return jnp.concatenate([out["prosthesis_mean"], out["body_mean"]], axis=-1)
        return out["prosthesis_mean"]

    return mean_action


def discounted_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + float(gamma) * running * (1.0 - float(dones[i]))
        out[i] = running
    return out


def build_train_step(model, *, body_residual: bool, clip_eps: float, entropy_weight: float):
    @jax.jit
    def train_step(state, obs, priv, actions, old_logp, advantages):
        def loss_fn(params):
            out = model.apply(params, obs, priv, body_residual=body_residual)
            p_action = actions[:, : out["prosthesis_mean"].shape[-1]]
            logp = gaussian_log_prob(p_action, out["prosthesis_mean"], out["prosthesis_log_std"])
            entropy = gaussian_entropy(out["prosthesis_log_std"])
            if body_residual and out["body_mean"].shape[-1] > 0:
                b_action = actions[:, out["prosthesis_mean"].shape[-1] :]
                logp = logp + gaussian_log_prob(b_action, out["body_mean"], out["body_log_std"])
                entropy = entropy + gaussian_entropy(out["body_log_std"])
            ratio = jnp.exp(logp - old_logp)
            unclipped = ratio * advantages
            clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            pg_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            loss = pg_loss - entropy_weight * entropy
            return loss, {
                "pg_loss": pg_loss,
                "entropy": entropy,
                "ratio_mean": jnp.mean(ratio),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, metrics

    return train_step


def collect_rollout(env, data, state, sample_fn, args, rng_key):
    obs_buf = []
    priv_buf = []
    action_buf = []
    logp_buf = []
    reward_buf = []
    done_buf = []
    term_sums: dict[str, float] = {}
    done_count = 0
    episode_lengths: list[int] = []
    cur_ep_len = 0
    for _ in range(args.rollout_steps):
        obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
        priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
        rng_key, act_key = jax.random.split(rng_key)
        action, logp = sample_fn(state.params, obs, priv, act_key)
        action_np = np.asarray(action)[0]
        p_action, b_action = split_actions(action_np, env.action_dim, env.body_residual_dim if args.body_residual_adapter else 0)
        data = env.step_action(p_action, b_action)
        obs_buf.append(np.asarray(obs)[0])
        priv_buf.append(np.asarray(priv)[0])
        action_buf.append(action_np.astype(np.float32))
        logp_buf.append(float(np.asarray(logp)[0]))
        reward_buf.append(float(data.reward))
        done_buf.append(float(data.done))
        cur_ep_len += 1
        for key, value in data.reward_terms.items():
            term_sums[key] = term_sums.get(key, 0.0) + float(value)
        if data.done:
            done_count += 1
            episode_lengths.append(cur_ep_len)
            cur_ep_len = 0
            data = env.reset()
    if cur_ep_len:
        episode_lengths.append(cur_ep_len)
    return data, rng_key, {
        "obs": jnp.asarray(np.stack(obs_buf), dtype=jnp.float32),
        "priv": jnp.asarray(np.stack(priv_buf), dtype=jnp.float32),
        "actions": jnp.asarray(np.stack(action_buf), dtype=jnp.float32),
        "old_logp": jnp.asarray(np.asarray(logp_buf), dtype=jnp.float32),
        "rewards": np.asarray(reward_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "term_means": {k: v / max(args.rollout_steps, 1) for k, v in term_sums.items()},
        "done_count": done_count,
        "done_rate": done_count / max(args.rollout_steps, 1),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float(args.rollout_steps),
    }


def evaluate(env, state, mean_fn, args):
    data = env.reset()
    rewards = []
    resets = 0
    survival = 0
    lengths = []
    tracking = []
    torque_err = []
    for _ in range(args.eval_steps):
        obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
        priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
        action = np.asarray(mean_fn(state.params, obs, priv))[0]
        p_action, b_action = split_actions(action, env.action_dim, env.body_residual_dim if args.body_residual_adapter else 0)
        data = env.step_action(p_action, b_action)
        rewards.append(float(data.reward))
        tracking.append(float(data.info.get("prosthesis_tracking_mae", 0.0)))
        applied = np.asarray(data.info.get("prosthesis_applied_torque_mean", np.zeros(4)), dtype=np.float64)
        oracle = np.asarray(data.info.get("prosthesis_oracle_torque_target", np.zeros(4)), dtype=np.float64)
        torque_err.append(float(np.mean(np.abs(applied - oracle))))
        survival += 1
        if data.done:
            resets += 1
            lengths.append(survival)
            survival = 0
            data = env.reset()
    if survival:
        lengths.append(survival)
    avg_survival = float(np.mean(lengths)) if lengths else float(args.eval_steps)
    score = resets * 10.0 + float(np.mean(tracking)) * 20.0 + float(np.mean(torque_err)) * 0.02 - 0.01 * avg_survival
    return {
        "score": float(score),
        "reward": float(np.mean(rewards)),
        "resets": resets,
        "avg_survival": avg_survival,
        "tracking_mae": float(np.mean(tracking)),
        "torque_mae": float(np.mean(torque_err)),
    }


def save_checkpoint(path: str, state: TrainState, meta: CoupledActionMeta):
    with open(path, "wb") as f:
        pickle.dump({**asdict(meta), "params": jax.device_get(state.params)}, f)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir = None if args.disable_tensorboard else (args.tensorboard_dir or os.path.join(args.output_dir, "tensorboard"))
    tb = TensorboardLogger(tb_dir)
    train_motion_count = count_motions(args, validation=False)
    val_motion_count = count_motions(args, validation=True)
    env = make_env(args, validation=False)
    val_env = make_env(args, validation=True)
    model = CoupledActionPolicy(
        prosthesis_action_dim=env.action_dim,
        body_residual_dim=env.body_residual_dim if args.body_residual_adapter else 0,
        action_scale=parse_vec(args.action_torque_limit_list, 4),
    )
    state = init_state(model, env, args)
    sample_fn = build_sample(model, body_residual=bool(args.body_residual_adapter))
    mean_fn = build_mean_action(model, body_residual=bool(args.body_residual_adapter))
    train_step = build_train_step(
        model,
        body_residual=bool(args.body_residual_adapter),
        clip_eps=float(args.clip_eps),
        entropy_weight=float(args.entropy_weight),
    )

    meta_base = {
        "checkpoint": args.checkpoint,
        "dataset_group": args.dataset_group,
        "val_dataset_group": args.val_dataset_group,
        "motion_path": args.motion_path,
        "val_motion_path": args.val_motion_path,
        "obs_dim": env.spec.obs_dim,
        "priv_dim": env.spec.priv_dim,
        "prosthesis_action_dim": env.action_dim,
        "body_residual_dim": env.body_residual_dim if args.body_residual_adapter else 0,
        "body_residual_adapter": bool(args.body_residual_adapter),
        "action_mode": args.action_mode,
        "action_torque_limit": parse_vec(args.action_torque_limit_list, 4),
        "action_torque_slew_limit": parse_vec(args.action_torque_slew_limit_list, 4),
        "prosthesis_muscle_scale": float(args.prosthesis_muscle_scale),
        "body_residual_scale": float(args.body_residual_scale if args.body_residual_adapter else 0.0),
        "mask_preset": str(args.mask_preset),
        "obs_mode": str(args.obs_mode),
        "discount": float(args.discount),
        "clip_eps": float(args.clip_eps),
    }
    print("Coupled action training:")
    print(f"  obs_dim={env.spec.obs_dim} priv_dim={env.spec.priv_dim} body_residual_dim={meta_base['body_residual_dim']}")
    print(
        f"  mask_preset={env.mask_preset} disabled_muscles={len(env.disabled_muscle_names)} "
        f"obs_mode={env.obs_mode} prosthesis_muscle_scale={env.prosthesis_muscle_scale}"
    )
    print(f"  output_dir={args.output_dir}")
    if train_motion_count is not None:
        print(f"  train_motions={train_motion_count} val_motions={val_motion_count}")
    if tb_dir:
        print(f"  tensorboard_dir={tb_dir}")

    run_config = {
        **meta_base,
        "steps": int(args.steps),
        "rollout_steps": int(args.rollout_steps),
        "update_epochs": int(args.update_epochs),
        "lr": float(args.lr),
        "entropy_weight": float(args.entropy_weight),
        "eval_interval": int(args.eval_interval),
        "eval_steps": int(args.eval_steps),
        "train_motion_count": train_motion_count,
        "val_motion_count": val_motion_count,
        "tensorboard_dir": tb_dir,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    data = env.reset()
    rng_key = jax.random.key(args.seed)
    best_score: float | None = None
    step = 0
    t0 = time.time()
    while step < args.steps:
        data, rng_key, batch = collect_rollout(env, data, state, sample_fn, args, rng_key)
        returns = discounted_returns(batch["rewards"], batch["dones"], args.discount)
        adv = returns - float(np.mean(returns))
        adv = adv / (float(np.std(adv)) + 1e-6)
        advantages = jnp.asarray(adv, dtype=jnp.float32)
        for _ in range(args.update_epochs):
            state, loss, metrics = train_step(
                state,
                batch["obs"],
                batch["priv"],
                batch["actions"],
                batch["old_logp"],
                advantages,
            )
        step += int(args.rollout_steps)
        fps = step / (time.time() - t0 + 1e-6)
        terms = " ".join(f"{k}={v:.4f}" for k, v in sorted(batch["term_means"].items()))
        train_reward = float(np.mean(batch["rewards"]))
        print(
            f"step={step:08d} reward={train_reward:.4f} "
            f"loss={float(loss):.4f} pg={float(metrics['pg_loss']):.4f} "
            f"entropy={float(metrics['entropy']):.3f} ratio={float(metrics['ratio_mean']):.3f} fps={fps:.1f} {terms}",
            flush=True,
        )
        tb.add_scalar("train/reward_mean", train_reward, step)
        tb.add_scalar("train/loss", float(loss), step)
        tb.add_scalar("train/pg_loss", float(metrics["pg_loss"]), step)
        tb.add_scalar("train/entropy", float(metrics["entropy"]), step)
        tb.add_scalar("train/ratio_mean", float(metrics["ratio_mean"]), step)
        tb.add_scalar("train/fps", fps, step)
        tb.add_scalar("train/done_rate", float(batch["done_rate"]), step)
        tb.add_scalar("train/mean_episode_length", float(batch["mean_episode_length"]), step)
        tb.add_scalars("train/reward_terms", batch["term_means"], step)
        if step % args.eval_interval < args.rollout_steps:
            val = evaluate(val_env, state, mean_fn, args)
            print(
                f"[eval] step={step:08d} score={val['score']:.4f} reward={val['reward']:.4f} "
                f"avg_survival={val['avg_survival']:.1f} resets={val['resets']} "
                f"tracking={val['tracking_mae']:.4f} torque_mae={val['torque_mae']:.4f}",
                flush=True,
            )
            tb.add_scalar("eval/score", val["score"], step)
            tb.add_scalar("eval/reward_mean", val["reward"], step)
            tb.add_scalar("eval/avg_survival", val["avg_survival"], step)
            tb.add_scalar("eval/resets", val["resets"], step)
            tb.add_scalar("eval/tracking_mae", val["tracking_mae"], step)
            tb.add_scalar("eval/torque_mae", val["torque_mae"], step)
            if best_score is None or val["score"] < best_score:
                best_score = val["score"]
                save_checkpoint(
                    os.path.join(args.output_dir, "coupled_action_best.pt"),
                    state,
                    CoupledActionMeta(step=step, best_eval_score=best_score, **meta_base),
                )
                print(f"saved best {os.path.join(args.output_dir, 'coupled_action_best.pt')}", flush=True)
        if step % args.save_interval < args.rollout_steps:
            save_checkpoint(
                os.path.join(args.output_dir, f"coupled_action_step_{step}.pt"),
                state,
                CoupledActionMeta(step=step, best_eval_score=best_score, **meta_base),
            )
    save_checkpoint(
        os.path.join(args.output_dir, "coupled_action_final.pt"),
        state,
        CoupledActionMeta(step=step, best_eval_score=best_score, **meta_base),
    )
    print(f"saved {os.path.join(args.output_dir, 'coupled_action_final.pt')}")
    tb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

