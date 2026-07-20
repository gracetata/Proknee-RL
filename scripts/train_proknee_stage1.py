#!/usr/bin/env python
"""Stage 1 Teacher training for MuscleMimic-ProKnee.

This is the full Stage-1 trainer scaffold:
- frozen mm-10m-2 generates the full-body oracle rollout;
- the Teacher receives global obs + privileged info;
- the target is the oracle next-step left knee/ankle joint residual;
- training is batched, with optional DAgger-style Teacher execution;
- validation evaluates target MSE on a held-out reference set.

The default action space is 4D joint residual:
    knee_angle_l, ankle_angle_l, subtalar_angle_l, mtp_angle_l
"""

from __future__ import annotations

import argparse
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

from musclemimic.proknee import MuscleProKneeHybridEnv
from musclemimic.proknee.models import MuscleProKneeTeacher


@dataclass(frozen=True)
class Stage1Meta:
    step: int
    obs_dim: int
    priv_dim: int
    action_dim: int
    history_len: int
    checkpoint: str
    dataset_group: str
    val_dataset_group: str
    motion_path: list[str] | None
    val_motion_path: list[str] | None
    audit_joints: list[str]
    target_mode: str
    prosthesis_muscle_scale: float
    teacher_exec_ratio: float
    dagger_pd_override: bool
    pd_kp: tuple[float, ...]
    pd_kd: tuple[float, ...]
    pd_torque_limit: tuple[float, ...]
    pd_torque_slew_limit: tuple[float, ...] | None
    oracle_torque_ff_scale: float
    oracle_torque_ff_limit: tuple[float, ...] | None
    pd_gain_scale: float
    max_prosthesis_qpos_step: tuple[float, ...] | None
    qvel_loss_weight: float
    smoothness_loss_weight: float
    command_step_loss_weight: float
    implied_qvel_mag_weight: float
    torque_head: bool
    torque_loss_weight: float
    torque_joint_weights: tuple[float, ...]
    predicted_torque_ff_scale: float
    torque_target_mode: str
    torque_exec_ratio: float
    command_step_joint_weights: tuple[float, ...]
    implied_qvel_mag_joint_weights: tuple[float, ...]
    physical_eval: bool
    physical_eval_steps: int
    physical_qvel_joint_weights: tuple[float, ...]
    physical_toe_z_weight: float
    physical_root_height_min: float
    physical_root_height_weight: float
    physical_root_up_weight: float
    physical_contact_weight: float
    physical_torque_oracle_weight: float
    control_dt: float


def parse_args():
    parser = argparse.ArgumentParser(description="Train MuscleMimic-ProKnee Stage 1 Teacher")
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--dataset-group", default="KIT_KINESIS_TRAINING_MOTIONS")
    parser.add_argument("--val-dataset-group", default="KIT_KINESIS_TESTING_MOTIONS")
    parser.add_argument("--motion-path", nargs="*", default=None, help="Optional explicit train references for smoke tests")
    parser.add_argument("--val-motion-path", nargs="*", default=None, help="Optional explicit validation references")
    parser.add_argument("--steps", type=int, default=100_000, help="Number of oracle samples/env steps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", default=None, help="Optional Stage1 checkpoint to continue training from")
    parser.add_argument("--teacher-exec-ratio", type=float, default=0.0,
                        help="Probability of executing Teacher action in HybridEnv. 0 keeps oracle-only stable rollout.")
    parser.add_argument("--teacher-noise-std", type=float, default=0.0)
    parser.add_argument("--target-mode", choices=["residual", "qpos"], default="residual")
    parser.add_argument("--qvel-loss-weight", type=float, default=1e-4)
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.01)
    parser.add_argument(
        "--command-step-loss-weight",
        type=float,
        default=0.0,
        help="Penalize large qpos target jumps from the current prosthesis qpos.",
    )
    parser.add_argument(
        "--implied-qvel-mag-weight",
        type=float,
        default=0.0,
        help="Penalize large implied prosthesis qvel from predicted qpos targets.",
    )
    parser.add_argument(
        "--command-step-joint-weights",
        default="1,1,1,1",
        help="Per-joint weights for command-step loss: knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--implied-qvel-mag-joint-weights",
        default="1,1,1,1",
        help="Per-joint weights for implied-qvel magnitude loss: knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--torque-head",
        action="store_true",
        help="Train an additional Stage1 torque feedforward head supervised by oracle prosthesis qfrc_actuator.",
    )
    parser.add_argument(
        "--torque-loss-weight",
        type=float,
        default=0.0,
        help="Weight for MSE(predicted_torque, oracle_prosthesis_torque).",
    )
    parser.add_argument(
        "--torque-joint-weights",
        default="1,1,1,1",
        help="Per-joint weights for torque-head loss: knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--predicted-torque-ff-scale",
        type=float,
        default=1.0,
        help="Scale predicted torque head before using it as feedforward in teacher rollout/physical eval.",
    )
    parser.add_argument(
        "--torque-target-mode",
        choices=["oracle", "pd_residual"],
        default="pd_residual",
        help=(
            "Torque-head label. 'oracle' learns raw oracle qfrc_actuator; "
            "'pd_residual' learns the feedforward residual so PD+FF matches oracle torque."
        ),
    )
    parser.add_argument(
        "--torque-exec-ratio",
        type=float,
        default=1.0,
        help=(
            "When teacher actions are executed, probability of using predicted torque FF. "
            "Otherwise the rollout uses oracle-equivalent FF, enabling a torque curriculum."
        ),
    )
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--dagger-pd-override", action="store_true", help="Use PD override for teacher rollout samples")
    parser.add_argument("--joint-kp", default="300,200,200,120")
    parser.add_argument("--joint-kd", default="30,20,20,12")
    parser.add_argument("--pd-torque-limit", type=float, default=120.0)
    parser.add_argument(
        "--pd-torque-limit-list",
        default=None,
        help="Per-joint torque limit for knee, ankle, subtalar, mtp; overrides --pd-torque-limit.",
    )
    parser.add_argument(
        "--pd-torque-slew-limit-list",
        default=None,
        help="Optional per-substep torque change cap for knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--oracle-torque-ff-scale",
        type=float,
        default=0.0,
        help="Scale oracle qfrc_actuator on prosthesis DOFs as torque feedforward during PD rollout.",
    )
    parser.add_argument(
        "--oracle-torque-ff-limit-list",
        default=None,
        help="Optional per-joint absolute cap for oracle torque feedforward: knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--pd-gain-scale",
        type=float,
        default=1.0,
        help="Multiply PD kp/kd in prosthesis-replacement steps (often <1 reduces jitter in sim).",
    )
    parser.add_argument(
        "--max-prosthesis-qpos-step",
        type=float,
        default=None,
        help="Optional max |Δq| per step (rad) on commanded prosthesis targets; reduces PD chatter.",
    )
    parser.add_argument(
        "--max-prosthesis-qpos-step-list",
        default=None,
        help="Per-joint max |Δq| per step (rad), overriding --max-prosthesis-qpos-step.",
    )
    parser.add_argument(
        "--prosthesis-muscle-scale",
        type=float,
        default=1.0,
        help="Scale [0,1] on prosthesis-boundary muscles during DAgger PD rollout (curriculum toward motor-only).",
    )
    parser.add_argument(
        "--conflict-muscle-scale",
        type=float,
        default=None,
        help="Deprecated: use --prosthesis-muscle-scale. If set, overrides --prosthesis-muscle-scale.",
    )
    parser.add_argument("--deterministic-oracle", action="store_true")
    parser.add_argument("--output-dir", default="outputs/proknee_stage1")
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument(
        "--physical-eval",
        action="store_true",
        help="Select best checkpoint with closed-loop prosthesis-replacement validation.",
    )
    parser.add_argument("--physical-eval-steps", type=int, default=512)
    parser.add_argument(
        "--physical-qvel-joint-weights",
        default="1,1,1,1",
        help="Per-joint qvel weights in physical checkpoint selection: knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--physical-toe-z-weight",
        type=float,
        default=0.0,
        help="Penalty weight on mean toe site height in physical checkpoint selection.",
    )
    parser.add_argument(
        "--physical-root-height-min",
        type=float,
        default=0.85,
        help="Minimum desired pelvis/root height in physical checkpoint selection.",
    )
    parser.add_argument(
        "--physical-root-height-weight",
        type=float,
        default=0.0,
        help="Penalty weight for pelvis/root height falling below --physical-root-height-min.",
    )
    parser.add_argument(
        "--physical-root-up-weight",
        type=float,
        default=0.0,
        help="Penalty weight for pelvis/root tilt away from upright in physical checkpoint selection.",
    )
    parser.add_argument(
        "--physical-contact-weight",
        type=float,
        default=0.0,
        help="Penalty weight for low contact count during physical checkpoint selection.",
    )
    parser.add_argument(
        "--physical-torque-oracle-weight",
        type=float,
        default=0.0,
        help="Penalty weight on |applied prosthesis torque - oracle muscle torque| during physical eval.",
    )
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_gain_list(text: str, fallback: float, n: int) -> tuple[float, ...]:
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    except Exception:
        vals = []
    if len(vals) != n:
        vals = [float(fallback)] * n
    return tuple(vals)


def parse_optional_gain_list(text: str | None, fallback: float | None, n: int) -> tuple[float, ...] | None:
    if text is None:
        if fallback is None:
            return None
        return tuple([float(fallback)] * n)
    return parse_gain_list(text, fallback if fallback is not None else 0.0, n)


def save_checkpoint(
    path: str,
    state: TrainState,
    meta: Stage1Meta,
    best_val_mse: float | None,
    best_val_score: float | None = None,
):
    payload = {
        **asdict(meta),
        "best_val_mse": best_val_mse,
        "best_val_score": best_val_score,
        "params": jax.device_get(state.params),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def resolve_prosthesis_muscle_scale(args) -> float:
    if getattr(args, "conflict_muscle_scale", None) is not None:
        print(
            "[train_proknee_stage1] WARNING: --conflict-muscle-scale is deprecated; "
            "prefer --prosthesis-muscle-scale.",
            flush=True,
        )
        return float(args.conflict_muscle_scale)
    return float(args.prosthesis_muscle_scale)


def make_env(args, *, validation: bool = False, physical: bool = False) -> MuscleProKneeHybridEnv:
    pd_kp = parse_gain_list(args.joint_kp, 250.0, 4)
    pd_kd = parse_gain_list(args.joint_kd, 25.0, 4)
    max_qpos_step = parse_optional_gain_list(args.max_prosthesis_qpos_step_list, args.max_prosthesis_qpos_step, 4)
    torque_limit = parse_optional_gain_list(args.pd_torque_limit_list, args.pd_torque_limit, 4)
    torque_slew_limit = parse_optional_gain_list(args.pd_torque_slew_limit_list, None, 4)
    p_scale = resolve_prosthesis_muscle_scale(args)
    execute_teacher = (args.teacher_exec_ratio > 0.0 and not validation) or physical
    pd_override = args.dagger_pd_override and (not validation or physical)
    return MuscleProKneeHybridEnv(
        args.checkpoint,
        dataset_group=args.val_dataset_group if validation else args.dataset_group,
        rel_dataset_path=args.val_motion_path if validation else args.motion_path,
        history_len=args.history_len,
        deterministic_oracle=args.deterministic_oracle,
        apply_teacher_action=execute_teacher,
        target_mode=args.target_mode,
        pd_override=pd_override,
        pd_kp=pd_kp,
        pd_kd=pd_kd,
        pd_torque_limit=torque_limit,
        pd_torque_slew_limit=torque_slew_limit,
        oracle_torque_ff_scale=args.oracle_torque_ff_scale,
        oracle_torque_ff_limit=parse_optional_gain_list(args.oracle_torque_ff_limit_list, None, 4),
        torque_feedforward_target_mode=args.torque_target_mode,
        pd_gain_scale=args.pd_gain_scale,
        max_prosthesis_qpos_step=max_qpos_step,
        prosthesis_muscle_scale=p_scale,
        conflict_muscle_scale=None,
    )


def init_state(model, env, args) -> TrainState:
    params = model.init(
        jax.random.key(args.seed),
        jnp.zeros((1, env.spec.obs_dim), dtype=jnp.float32),
        jnp.zeros((1, env.spec.priv_dim), dtype=jnp.float32),
        return_torque=bool(args.torque_head),
    )
    if args.resume:
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        params = merge_compatible_params(params, ckpt["params"])
    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.lr))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def merge_compatible_params(initialized_params, checkpoint_params):
    """Load old checkpoints while leaving newly added heads initialized."""

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


def build_train_step(*, torque_head: bool = False):
    @jax.jit
    def train_step(
        state,
        obs,
        priv,
        target,
        current_qpos,
        target_qvel,
        qvel_weight,
        smoothness_weight,
        command_step_weight,
        implied_qvel_mag_weight,
        command_step_joint_weights,
        implied_qvel_mag_joint_weights,
        torque_target,
        torque_loss_weight,
        torque_joint_weights,
        control_dt,
    ):
        def loss_fn(params):
            if torque_head:
                pred, pred_torque, latent = state.apply_fn(params, obs, priv, return_torque=True)
            else:
                pred, latent = state.apply_fn(params, obs, priv)
                pred_torque = jnp.zeros_like(torque_target)
            diff = pred - target
            target_loss = jnp.mean(jnp.square(diff))
            torque_diff = pred_torque - torque_target
            torque_loss = jnp.mean(jnp.square(torque_diff) * torque_joint_weights)
            implied_qvel = (pred - current_qpos) / jnp.maximum(control_dt, 1e-6)
            qvel_loss = jnp.mean(jnp.square(implied_qvel - target_qvel))
            command_step_loss = jnp.mean(jnp.square(pred - current_qpos) * command_step_joint_weights)
            implied_qvel_mag_loss = jnp.mean(jnp.square(implied_qvel) * implied_qvel_mag_joint_weights)
            if pred.shape[0] > 1:
                pred_delta = pred[1:] - pred[:-1]
                target_delta = target[1:] - target[:-1]
                smoothness_loss = jnp.mean(jnp.square(pred_delta - target_delta))
            else:
                smoothness_loss = jnp.asarray(0.0, dtype=pred.dtype)
            loss = (
                target_loss
                + qvel_weight * qvel_loss
                + smoothness_weight * smoothness_loss
                + command_step_weight * command_step_loss
                + implied_qvel_mag_weight * implied_qvel_mag_loss
                + torque_loss_weight * torque_loss
            )
            mae = jnp.mean(jnp.abs(diff))
            torque_mae = jnp.mean(jnp.abs(torque_diff))
            latent_norm = jnp.mean(jnp.linalg.norm(latent, axis=-1))
            return loss, {
                "mae": mae,
                "latent_norm": latent_norm,
                "target_loss": target_loss,
                "qvel_loss": qvel_loss,
                "smoothness_loss": smoothness_loss,
                "command_step_loss": command_step_loss,
                "implied_qvel_mag_loss": implied_qvel_mag_loss,
                "torque_loss": torque_loss,
                "torque_mae": torque_mae,
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, metrics

    return train_step


def build_predict(model, *, torque_head: bool = False):
    @jax.jit
    def predict(params, obs, priv):
        if torque_head:
            pred, pred_torque, _latent = model.apply(params, obs, priv, return_torque=True)
            return pred, pred_torque
        pred, _latent = model.apply(params, obs, priv)
        return pred, jnp.zeros_like(pred)

    return predict


def oracle_equivalent_torque_target(env, step_data, target_qpos: np.ndarray, mode: str) -> np.ndarray:
    """Return the torque-head label for the current sample.

    In ``pd_residual`` mode the head learns only the feedforward residual needed
    on top of the PD tracking term, so the total motor-layer torque can match
    the original muscle-generated generalized force.
    """

    oracle_torque = np.asarray(step_data.oracle_torque_target, dtype=np.float32)
    if mode == "oracle":
        target = np.asarray(oracle_torque, dtype=np.float64)
    elif mode == "pd_residual":
        q = np.asarray(step_data.info["qpos_before"][env.audit.qpos_indices], dtype=np.float64)
        qd = np.asarray(step_data.info["qvel_before"][env.audit.qvel_indices], dtype=np.float64)
        kp = np.asarray(env.pd_kp, dtype=np.float64) * float(env.pd_gain_scale)
        kd = np.asarray(env.pd_kd, dtype=np.float64) * float(env.pd_gain_scale)
        pd_term = kp * (np.asarray(target_qpos, dtype=np.float64) - q) - kd * qd
        target = np.asarray(oracle_torque, dtype=np.float64) - pd_term
    else:
        raise ValueError(f"Unknown torque target mode: {mode}")

    # Train only executable feedforward torques.  Without this clip the head is
    # supervised on values that deployment will later saturate away, especially
    # on the knee during support.
    ff_limit = getattr(env, "oracle_torque_ff_limit", None)
    if ff_limit is not None:
        limit = np.asarray(ff_limit, dtype=np.float64)
        target = np.clip(target, -limit, limit)
    return target.astype(np.float32)


def collect_batch(env, step_data, state, predict_fn, args, rng: np.random.Generator):
    obs_buf, priv_buf, target_buf, qpos_buf, qvel_buf, torque_buf = [], [], [], [], [], []
    for _ in range(args.batch_size):
        teacher_action = None
        teacher_torque = None
        if args.teacher_exec_ratio > 0.0 and rng.random() < args.teacher_exec_ratio:
            pred, pred_torque = predict_fn(
                state.params,
                jnp.asarray(step_data.obs[None, :], dtype=jnp.float32),
                jnp.asarray(step_data.priv_info[None, :], dtype=jnp.float32),
            )
            teacher_action = np.asarray(pred)[0]
            if args.torque_head and rng.random() < float(np.clip(args.torque_exec_ratio, 0.0, 1.0)):
                teacher_torque = float(args.predicted_torque_ff_scale) * np.asarray(pred_torque)[0]
            if args.teacher_noise_std > 0:
                teacher_action = teacher_action + rng.normal(0.0, args.teacher_noise_std, teacher_action.shape)

        step_data = env.step(teacher_action, teacher_torque)
        obs_buf.append(step_data.obs)
        priv_buf.append(step_data.priv_info)
        target_buf.append(step_data.oracle_target)
        qpos_buf.append(step_data.info["qpos_before"][env.audit.qpos_indices])
        qvel_buf.append(step_data.info["qvel_after"][env.audit.qvel_indices])
        torque_buf.append(oracle_equivalent_torque_target(env, step_data, step_data.oracle_target, args.torque_target_mode))
        if step_data.done:
            step_data = env.reset()

    return (
        step_data,
        jnp.asarray(np.stack(obs_buf), dtype=jnp.float32),
        jnp.asarray(np.stack(priv_buf), dtype=jnp.float32),
        jnp.asarray(np.stack(target_buf), dtype=jnp.float32),
        jnp.asarray(np.stack(qpos_buf), dtype=jnp.float32),
        jnp.asarray(np.stack(qvel_buf), dtype=jnp.float32),
        jnp.asarray(np.stack(torque_buf), dtype=jnp.float32),
    )


def evaluate(env, state, model, n_steps: int):
    predict = build_predict(model, torque_head=False)
    data = env.reset()
    mse_sum, mae_sum, qvel_mse_sum, resets = 0.0, 0.0, 0.0, 0
    for _ in range(n_steps):
        data = env.step()
        obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
        priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
        target = jnp.asarray(data.oracle_target[None, :], dtype=jnp.float32)
        pred, _pred_torque = predict(state.params, obs, priv)
        diff = np.asarray(pred - target)
        mse_sum += float(np.mean(diff**2))
        mae_sum += float(np.mean(np.abs(diff)))
        current_qpos = data.info["qpos_before"][env.audit.qpos_indices]
        target_qvel = data.info["qvel_after"][env.audit.qvel_indices]
        implied_qvel = (np.asarray(pred)[0] - current_qpos) / 0.01
        qvel_mse_sum += float(np.mean((implied_qvel - target_qvel) ** 2))
        if data.done:
            resets += 1
            data = env.reset()
    denom = max(n_steps, 1)
    return {"mse": mse_sum / denom, "mae": mae_sum / denom, "qvel_mse": qvel_mse_sum / denom, "resets": resets}


def evaluate_physical(
    env,
    state,
    model,
    n_steps: int,
    control_dt: float,
    qvel_joint_weights: tuple[float, ...],
    toe_z_weight: float,
    root_height_min: float,
    root_height_weight: float,
    root_up_weight: float,
    contact_weight: float,
    torque_oracle_weight: float,
    torque_head: bool = False,
    predicted_torque_ff_scale: float = 1.0,
):
    predict = build_predict(model, torque_head=torque_head)
    data = env.reset()
    mse_sum = mae_sum = qvel_mse_sum = 0.0
    qvel_ma_sum = tracking_mae_sum = torque_abs_sum = 0.0
    weighted_qvel_ma_sum = toe_z_sum = 0.0
    root_height_shortfall_sum = root_tilt_sum = contact_shortfall_sum = 0.0
    torque_oracle_mae_sum = 0.0
    qvel_weights = np.asarray(qvel_joint_weights, dtype=np.float64)
    resets = 0
    survival = 0
    episode_lengths: list[int] = []
    for _ in range(n_steps):
        obs = jnp.asarray(data.obs[None, :], dtype=jnp.float32)
        priv = jnp.asarray(data.priv_info[None, :], dtype=jnp.float32)
        target = jnp.asarray(data.oracle_target[None, :], dtype=jnp.float32)
        pred, pred_torque = predict(state.params, obs, priv)
        pred_np = np.asarray(pred)[0]
        diff = np.asarray(pred - target)
        mse_sum += float(np.mean(diff**2))
        mae_sum += float(np.mean(np.abs(diff)))
        current_qpos_full = data.info.get("qpos_before", np.asarray(env.env.data.qpos, dtype=np.float32))
        target_qvel_full = data.info.get("qvel_after", np.asarray(env.env.data.qvel, dtype=np.float32))
        current_qpos = current_qpos_full[env.audit.qpos_indices]
        target_qvel = target_qvel_full[env.audit.qvel_indices]
        implied_qvel = (pred_np - current_qpos) / max(float(control_dt), 1e-6)
        qvel_mse_sum += float(np.mean((implied_qvel - target_qvel) ** 2))

        torque_np = np.asarray(pred_torque)[0] if torque_head else None
        if torque_np is not None:
            torque_np = torque_np * float(predicted_torque_ff_scale)
        data = env.step(pred_np, torque_np)
        info = data.info
        qvel_ma_sum += float(info.get("prosthesis_qvel_ma", 0.0))
        qvel4 = np.asarray(info.get("prosthesis_qvel", np.zeros(4, dtype=np.float32)), dtype=np.float64)
        weighted_qvel_ma_sum += float(np.mean(np.abs(qvel4) * qvel_weights))
        toe_z_sum += float(info.get("prosthesis_toe_z", 0.0))
        root_height = float(info.get("prosthesis_root_height", env.env.data.qpos[2]))
        root_up_z = float(info.get("prosthesis_root_up_z", 1.0))
        ncon = float(info.get("prosthesis_ncon", info.get("ncon", 0.0)))
        root_height_shortfall_sum += max(float(root_height_min) - root_height, 0.0)
        root_tilt_sum += max(1.0 - root_up_z, 0.0)
        contact_shortfall_sum += max(1.0 - ncon, 0.0)
        tracking_mae_sum += float(info.get("prosthesis_tracking_mae", 0.0))
        torque_abs_sum += float(info.get("prosthesis_pd_torque_abs_mean", 0.0))
        applied_torque = np.asarray(info.get("prosthesis_applied_torque_mean", np.zeros(4)), dtype=np.float64)
        oracle_torque = np.asarray(info.get("prosthesis_oracle_torque_target", np.zeros(4)), dtype=np.float64)
        torque_oracle_mae_sum += float(np.mean(np.abs(applied_torque - oracle_torque)))
        survival += 1
        if data.done:
            resets += 1
            episode_lengths.append(survival)
            survival = 0
            data = env.reset()
    if survival > 0:
        episode_lengths.append(survival)
    denom = max(n_steps, 1)
    avg_survival = float(np.mean(episode_lengths)) if episode_lengths else float(n_steps)
    qvel_ma = qvel_ma_sum / denom
    weighted_qvel_ma = weighted_qvel_ma_sum / denom
    toe_z = toe_z_sum / denom
    root_height_shortfall = root_height_shortfall_sum / denom
    root_tilt = root_tilt_sum / denom
    contact_shortfall = contact_shortfall_sum / denom
    tracking_mae = tracking_mae_sum / denom
    torque_abs_mean = torque_abs_sum / denom
    torque_oracle_mae = torque_oracle_mae_sum / denom
    # Lower is better. Resets dominate; qvel/tracking terms break ties toward calmer physical rollouts.
    score = (
        resets * 10.0
        + weighted_qvel_ma
        + 20.0 * tracking_mae
        + 0.01 * torque_abs_mean
        + float(toe_z_weight) * toe_z
        + float(root_height_weight) * root_height_shortfall
        + float(root_up_weight) * root_tilt
        + float(contact_weight) * contact_shortfall
        + float(torque_oracle_weight) * torque_oracle_mae
        - 0.01 * avg_survival
    )
    return {
        "mse": mse_sum / denom,
        "mae": mae_sum / denom,
        "qvel_mse": qvel_mse_sum / denom,
        "resets": resets,
        "avg_survival": avg_survival,
        "qvel_ma": qvel_ma,
        "weighted_qvel_ma": weighted_qvel_ma,
        "toe_z": toe_z,
        "root_height_shortfall": root_height_shortfall,
        "root_tilt": root_tilt,
        "contact_shortfall": contact_shortfall,
        "tracking_mae": tracking_mae,
        "torque_abs_mean": torque_abs_mean,
        "torque_oracle_mae": torque_oracle_mae,
        "score": float(score),
    }


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_env = make_env(args, validation=False)
    val_env = make_env(args, validation=True)
    physical_val_env = make_env(args, validation=True, physical=True) if args.physical_eval else None
    step_data = train_env.reset()

    model = MuscleProKneeTeacher(action_dim=train_env.action_dim)
    state = init_state(model, train_env, args)
    train_step = build_train_step(torque_head=bool(args.torque_head))
    predict = build_predict(model, torque_head=bool(args.torque_head))

    pd_kp_meta = parse_gain_list(args.joint_kp, 250.0, 4)
    pd_kd_meta = parse_gain_list(args.joint_kd, 25.0, 4)
    max_qpos_step_meta = parse_optional_gain_list(args.max_prosthesis_qpos_step_list, args.max_prosthesis_qpos_step, 4)
    torque_limit_meta = parse_optional_gain_list(args.pd_torque_limit_list, args.pd_torque_limit, 4)
    torque_slew_limit_meta = parse_optional_gain_list(args.pd_torque_slew_limit_list, None, 4)
    oracle_torque_ff_limit_meta = parse_optional_gain_list(args.oracle_torque_ff_limit_list, None, 4)
    command_step_joint_weights_meta = parse_gain_list(args.command_step_joint_weights, 1.0, 4)
    implied_qvel_mag_joint_weights_meta = parse_gain_list(args.implied_qvel_mag_joint_weights, 1.0, 4)
    torque_joint_weights_meta = parse_gain_list(args.torque_joint_weights, 1.0, 4)
    physical_qvel_joint_weights_meta = parse_gain_list(args.physical_qvel_joint_weights, 1.0, 4)
    p_scale_meta = resolve_prosthesis_muscle_scale(args)
    meta_base = {
        "obs_dim": train_env.spec.obs_dim,
        "priv_dim": train_env.spec.priv_dim,
        "action_dim": train_env.action_dim,
        "history_len": train_env.spec.history_len,
        "checkpoint": args.checkpoint,
        "dataset_group": args.dataset_group,
        "val_dataset_group": args.val_dataset_group,
        "motion_path": args.motion_path,
        "val_motion_path": args.val_motion_path,
        "audit_joints": [j.name for j in train_env.audit.joints],
        "target_mode": args.target_mode,
        "prosthesis_muscle_scale": p_scale_meta,
        "teacher_exec_ratio": float(args.teacher_exec_ratio),
        "dagger_pd_override": bool(args.dagger_pd_override),
        "pd_kp": pd_kp_meta,
        "pd_kd": pd_kd_meta,
        "pd_torque_limit": torque_limit_meta,
        "pd_torque_slew_limit": torque_slew_limit_meta,
        "oracle_torque_ff_scale": float(args.oracle_torque_ff_scale),
        "oracle_torque_ff_limit": oracle_torque_ff_limit_meta,
        "pd_gain_scale": float(args.pd_gain_scale),
        "max_prosthesis_qpos_step": max_qpos_step_meta,
        "qvel_loss_weight": float(args.qvel_loss_weight),
        "smoothness_loss_weight": float(args.smoothness_loss_weight),
        "command_step_loss_weight": float(args.command_step_loss_weight),
        "implied_qvel_mag_weight": float(args.implied_qvel_mag_weight),
        "torque_head": bool(args.torque_head),
        "torque_loss_weight": float(args.torque_loss_weight),
        "torque_joint_weights": torque_joint_weights_meta,
        "predicted_torque_ff_scale": float(args.predicted_torque_ff_scale),
        "torque_target_mode": args.torque_target_mode,
        "torque_exec_ratio": float(args.torque_exec_ratio),
        "command_step_joint_weights": command_step_joint_weights_meta,
        "implied_qvel_mag_joint_weights": implied_qvel_mag_joint_weights_meta,
        "physical_eval": bool(args.physical_eval),
        "physical_eval_steps": int(args.physical_eval_steps),
        "physical_qvel_joint_weights": physical_qvel_joint_weights_meta,
        "physical_toe_z_weight": float(args.physical_toe_z_weight),
        "physical_root_height_min": float(args.physical_root_height_min),
        "physical_root_height_weight": float(args.physical_root_height_weight),
        "physical_root_up_weight": float(args.physical_root_up_weight),
        "physical_contact_weight": float(args.physical_contact_weight),
        "physical_torque_oracle_weight": float(args.physical_torque_oracle_weight),
        "control_dt": float(args.control_dt),
    }

    print("Stage1 full training:")
    print(f"  obs_dim={train_env.spec.obs_dim}, priv_dim={train_env.spec.priv_dim}, action_dim={train_env.action_dim}")
    print(f"  dataset_group={args.dataset_group}, val_dataset_group={args.val_dataset_group}")
    print(f"  target_mode={args.target_mode}")
    print(
        f"  batch_size={args.batch_size}, steps={args.steps}, teacher_exec_ratio={args.teacher_exec_ratio}, "
        f"prosthesis_muscle_scale={p_scale_meta}, dagger_pd_override={args.dagger_pd_override}",
        flush=True,
    )
    print(f"  output_dir={args.output_dir}")

    best_val_mse: float | None = None
    best_val_score: float | None = None
    running_loss, running_mae, samples_since_log = 0.0, 0.0, 0
    t0 = time.time()
    step = 0
    while step < args.steps:
        step_data, obs, priv, target, current_qpos, target_qvel, torque_target = collect_batch(
            train_env, step_data, state, predict, args, rng
        )
        batch_n = int(obs.shape[0])
        step += batch_n
        state, loss, metrics = train_step(
            state,
            obs,
            priv,
            target,
            current_qpos,
            target_qvel,
            jnp.asarray(args.qvel_loss_weight, dtype=jnp.float32),
            jnp.asarray(args.smoothness_loss_weight, dtype=jnp.float32),
            jnp.asarray(args.command_step_loss_weight, dtype=jnp.float32),
            jnp.asarray(args.implied_qvel_mag_weight, dtype=jnp.float32),
            jnp.asarray(command_step_joint_weights_meta, dtype=jnp.float32),
            jnp.asarray(implied_qvel_mag_joint_weights_meta, dtype=jnp.float32),
            torque_target,
            jnp.asarray(args.torque_loss_weight, dtype=jnp.float32),
            jnp.asarray(torque_joint_weights_meta, dtype=jnp.float32),
            jnp.asarray(args.control_dt, dtype=jnp.float32),
        )
        running_loss += float(loss) * batch_n
        running_mae += float(metrics["mae"]) * batch_n
        samples_since_log += batch_n

        if step % args.log_interval < batch_n:
            fps = step / (time.time() - t0 + 1e-6)
            print(
                f"step={step:08d} loss={running_loss/max(samples_since_log,1):.6f} "
                f"mae={running_mae/max(samples_since_log,1):.6f} "
                f"target_loss={float(metrics['target_loss']):.6f} "
                f"qvel_loss={float(metrics['qvel_loss']):.6f} "
                f"smooth={float(metrics['smoothness_loss']):.6f} "
                f"cmd_step={float(metrics['command_step_loss']):.6f} "
                f"qvel_mag={float(metrics['implied_qvel_mag_loss']):.6f} "
                f"torque_loss={float(metrics['torque_loss']):.6f} "
                f"torque_mae={float(metrics['torque_mae']):.6f} "
                f"latent_norm={float(metrics['latent_norm']):.3f} fps={fps:.1f}",
                flush=True,
            )
            running_loss = running_mae = 0.0
            samples_since_log = 0

        if step % args.eval_interval < batch_n:
            val = evaluate(val_env, state, model, args.eval_steps)
            print(
                f"[eval] step={step:08d} val_mse={val['mse']:.6f} "
                f"val_mae={val['mae']:.6f} val_qvel_mse={val['qvel_mse']:.6f} resets={val['resets']}",
                flush=True,
            )
            selected_score = val["mse"]
            if args.physical_eval and physical_val_env is not None:
                phys = evaluate_physical(
                    physical_val_env,
                    state,
                    model,
                    args.physical_eval_steps,
                    args.control_dt,
                    physical_qvel_joint_weights_meta,
                    args.physical_toe_z_weight,
                    args.physical_root_height_min,
                    args.physical_root_height_weight,
                    args.physical_root_up_weight,
                    args.physical_contact_weight,
                    args.physical_torque_oracle_weight,
                    bool(args.torque_head),
                    float(args.predicted_torque_ff_scale),
                )
                selected_score = phys["score"]
                print(
                    f"[physical_eval] step={step:08d} score={phys['score']:.6f} "
                    f"avg_survival={phys['avg_survival']:.1f} resets={phys['resets']} "
                    f"qvel_ma={phys['qvel_ma']:.6f} weighted_qvel_ma={phys['weighted_qvel_ma']:.6f} "
                    f"toe_z={phys['toe_z']:.6f} tracking_mae={phys['tracking_mae']:.6f} "
                    f"root_h_short={phys['root_height_shortfall']:.6f} root_tilt={phys['root_tilt']:.6f} "
                    f"contact_short={phys['contact_shortfall']:.6f} "
                    f"torque_abs={phys['torque_abs_mean']:.6f} "
                    f"torque_oracle_mae={phys['torque_oracle_mae']:.6f}",
                    flush=True,
                )
            if best_val_score is None or selected_score < best_val_score:
                best_val_mse = val["mse"]
                best_val_score = float(selected_score)
                best_path = os.path.join(args.output_dir, "stage1_best.pt")
                save_checkpoint(best_path, state, Stage1Meta(step=step, **meta_base), best_val_mse, best_val_score)
                print(f"saved best {best_path}")

        if step % args.save_interval < batch_n:
            path = os.path.join(args.output_dir, f"stage1_step_{step}.pt")
            save_checkpoint(path, state, Stage1Meta(step=step, **meta_base), best_val_mse, best_val_score)
            print(f"saved {path}")

    final_path = os.path.join(args.output_dir, "stage1_final.pt")
    save_checkpoint(final_path, state, Stage1Meta(step=step, **meta_base), best_val_mse, best_val_score)
    print(f"saved {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
