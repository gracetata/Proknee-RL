#!/usr/bin/env python
"""Supervised distillation for MyoFullBodyProsthesisEnv policies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.distill.checkpoint import save_distilled_checkpoint, save_loss_curve
from musclemimic.distill.config import (
    apply_prosthesis_overrides,
    load_fullbody_config,
    make_env,
    parse_optional_vec4,
    repo_root,
)
from musclemimic.distill.dataset import ProsthesisDistillDataset
from musclemimic.distill.mapping import DistillMapping, build_distill_mapping, validate_distill_dimensions
from musclemimic.runner.eval_utils import load_checkpoint

DEFAULT_TEACHER_CHECKPOINT = "/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2"
DEFAULT_DATASET_DIR = "data/teacher_rollouts/KIT_KINESIS_TRAINING_MOTIONS"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--extra_dataset_dir", action="append", default=[])
    parser.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    parser.add_argument(
        "--mask-preset",
        default=None,
        choices=["strict19", "knee15", "distal11", "foot5", "foot1"],
        help="Must match the preset used when collecting teacher rollouts.",
    )
    parser.add_argument(
        "--prosthesis_action_type",
        default=None,
        choices=[None, "torque", "pd_residual_torque"],
        help="Must match the 4-DOF action type used when collecting teacher rollouts.",
    )
    parser.add_argument("--residual_pd_kp", default=None, help="PD gains for pd_residual_torque action type")
    parser.add_argument("--residual_pd_kd", default=None, help="Damping gains for pd_residual_torque action type")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_muscle", type=float, default=1.0)
    parser.add_argument("--lambda_prosthesis", type=float, default=10.0)
    parser.add_argument("--lambda_tau", type=float, default=0.0, help="Weight for normalized prosthesis torque MSE")
    parser.add_argument("--init", choices=["random", "official_trunk_init", "checkpoint"], default="official_trunk_init")
    parser.add_argument("--init_checkpoint", default=None, help="Resume/fine-tune from a distilled checkpoint when --init checkpoint")
    parser.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--output_dir", default="outputs/prosthesis_distill")
    parser.add_argument("--wandb", choices=["disabled", "online"], default="disabled")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument(
        "--val_dataset_dir",
        action="append",
        default=[],
        help="Optional extra validation-only dataset dirs (e.g. latest DAgger round). Uses split=all.",
    )
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def assert_jax_gpu(required: bool) -> None:
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    has_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    if required and not has_gpu:
        raise RuntimeError("GPU required for distillation but jax.devices() shows no GPU. Unset JAX_PLATFORMS=cpu.")


def merge_compatible_params(student, teacher):
    warnings = []

    def merge(path, s, t):
        if isinstance(s, dict) and isinstance(t, dict):
            return {k: merge(path + (k,), s[k], t[k]) if k in t else s[k] for k in s}
        if hasattr(s, "shape") and hasattr(t, "shape") and tuple(s.shape) == tuple(t.shape):
            return t
        if hasattr(s, "shape") and hasattr(t, "shape"):
            warnings.append((".".join(path), tuple(s.shape), tuple(t.shape)))
        return s

    return merge(tuple(), student, teacher), warnings


def load_mapping_for_dataset(dataset_dir: Path, teacher_env, student_env) -> DistillMapping:
    mapping_path = dataset_dir / "mapping.json"
    if mapping_path.exists():
        mapping = DistillMapping.load_json(mapping_path)
    else:
        mapping = build_distill_mapping(teacher_env, student_env)
    env_mapping = build_distill_mapping(teacher_env, student_env)
    if mapping.student_obs_dim != env_mapping.student_obs_dim:
        raise ValueError(
            f"Dataset mapping student_obs_dim={mapping.student_obs_dim} != env {env_mapping.student_obs_dim}"
        )
    if mapping.n_remaining_muscles != env_mapping.n_remaining_muscles:
        raise ValueError(
            f"Dataset mapping n_remaining={mapping.n_remaining_muscles} != env {env_mapping.n_remaining_muscles}"
        )
    return mapping


def init_student_state(cfg, env, args):
    agent_conf = PPOJax.init_agent_conf(env, cfg)
    rng = jax.random.PRNGKey(args.seed)
    init_vars = agent_conf.network.init(rng, jnp.zeros(env.info.observation_space.shape, dtype=jnp.float32))
    params = init_vars["params"]
    run_stats = init_vars["run_stats"]
    if args.init == "official_trunk_init":
        teacher_cfg, teacher_state, _ = load_checkpoint(args.teacher_checkpoint)
        teacher_params = teacher_state.train_state.params
        params, warnings = merge_compatible_params(params, teacher_params)
        if warnings:
            print(f"[official_trunk_init] skipped {len(warnings)} incompatible leaves")
            for name, s_shape, t_shape in warnings[:20]:
                print(f"  {name}: student={s_shape} teacher={t_shape}")
        try:
            run_stats, rs_warnings = merge_compatible_params(run_stats, teacher_state.train_state.run_stats)
            if rs_warnings:
                print(f"[official_trunk_init] skipped {len(rs_warnings)} incompatible run_stats leaves")
        except Exception as exc:
            print(f"[official_trunk_init] warning: could not merge run_stats: {exc}")
        _ = teacher_cfg
    elif args.init == "checkpoint":
        if not args.init_checkpoint:
            raise ValueError("--init checkpoint requires --init_checkpoint")
        _cfg, agent_state, _meta = load_checkpoint(str(resolve_path(args.init_checkpoint)))
        params, warnings = merge_compatible_params(params, agent_state.train_state.params)
        if warnings:
            print(f"[init_checkpoint] skipped {len(warnings)} incompatible param leaves")
        try:
            run_stats, rs_warnings = merge_compatible_params(run_stats, agent_state.train_state.run_stats)
            if rs_warnings:
                print(f"[init_checkpoint] skipped {len(rs_warnings)} incompatible run_stats leaves")
        except Exception as exc:
            print(f"[init_checkpoint] warning: could not merge run_stats: {exc}")
    tx = optax.adamw(args.lr)
    state = TrainState.create(apply_fn=agent_conf.network.apply, params=params, run_stats=run_stats, tx=tx)
    return agent_conf, state


def make_train_step(network, tx, n_remaining: int, lambda_muscle: float, lambda_prosthesis: float, lambda_tau: float):
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params, run_stats):
            y, updates = network.apply(
                {"params": params, "run_stats": run_stats},
                batch["obs"],
                mutable=["run_stats"],
            )
            pi, _value = y
            pred = pi.mean()
            pred_remaining = pred[:, :n_remaining]
            pred_prosthesis = pred[:, n_remaining : n_remaining + 4]
            muscle_mse = jnp.mean((pred_remaining - batch["target_remaining"]) ** 2)
            prosthesis_mse = jnp.mean((pred_prosthesis - batch["target_prosthesis"]) ** 2)
            residual_tau_pred = pred_prosthesis * batch["torque_limits"]
            tau_pred = jnp.clip(batch["tau_pd"] + residual_tau_pred, -batch["torque_limits"], batch["torque_limits"])
            tau_err_norm = (tau_pred - batch["target_tau"]) / jnp.maximum(batch["torque_limits"], 1e-6)
            tau_mse = jnp.mean(tau_err_norm**2)
            loss = lambda_muscle * muscle_mse + lambda_prosthesis * prosthesis_mse + lambda_tau * tau_mse
            metrics = {
                "total_loss": loss,
                "muscle_mse": muscle_mse,
                "prosthesis_mse": prosthesis_mse,
                "prosthesis_tau_mse": tau_mse,
            }
            return loss, (metrics, updates["run_stats"])

        (loss, (metrics, run_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state.run_stats)
        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return state.replace(params=params, opt_state=opt_state, run_stats=run_stats, step=state.step + 1), {**metrics, "total_loss": loss}

    return train_step


def eval_loss(network, state, dataset, batch_size, n_remaining, lambda_muscle, lambda_prosthesis, lambda_tau, torque_limits):
    losses = []
    for batch in dataset.iter_batches(batch_size, shuffle=False):
        obs = jnp.asarray(batch["obs"])
        y, _updates = network.apply({"params": state.params, "run_stats": state.run_stats}, obs, mutable=["run_stats"])
        pi, _ = y
        pred = pi.mean()
        mr = jnp.mean((pred[:, :n_remaining] - jnp.asarray(batch["target_remaining"])) ** 2)
        mp = jnp.mean((pred[:, n_remaining : n_remaining + 4] - jnp.asarray(batch["target_prosthesis"])) ** 2)
        limits = jnp.asarray(torque_limits)
        residual_tau_pred = pred[:, n_remaining : n_remaining + 4] * limits
        tau_pred = jnp.clip(jnp.asarray(batch["tau_pd"]) + residual_tau_pred, -limits, limits)
        tau_err_norm = (tau_pred - jnp.asarray(batch["target_tau"])) / jnp.maximum(limits, 1e-6)
        tm = jnp.mean(tau_err_norm**2)
        losses.append(np.asarray([lambda_muscle * mr + lambda_prosthesis * mp + lambda_tau * tm, mr, mp, tm], dtype=np.float64))
    return np.mean(np.vstack(losses), axis=0)


def main() -> int:
    args = parse_args()
    assert_jax_gpu(args.require_gpu)

    cfg = load_fullbody_config(args.config_name)
    if args.mask_preset or args.prosthesis_action_type or args.residual_pd_kp or args.residual_pd_kd:
        cfg = apply_prosthesis_overrides(
            cfg,
            disable_preset=args.mask_preset,
            action_type=args.prosthesis_action_type,
            residual_pd_kp=parse_optional_vec4(args.residual_pd_kp),
            residual_pd_kd=parse_optional_vec4(args.residual_pd_kd),
        )
    OmegaConf.set_struct(cfg, False)
    dataset_dirs = [resolve_path(args.dataset_dir), *[resolve_path(p) for p in args.extra_dataset_dir]]
    train_ds = ProsthesisDistillDataset(dataset_dirs, "train", args.val_fraction, args.seed, args.max_files, args.max_frames)
    val_ds = ProsthesisDistillDataset(dataset_dirs, "val", args.val_fraction, args.seed, args.max_files, args.max_frames)
    val_dagger_ds = None
    if args.val_dataset_dir:
        val_dagger_dirs = [resolve_path(p) for p in args.val_dataset_dir]
        val_dagger_ds = ProsthesisDistillDataset(
            val_dagger_dirs, "all", 0.0, args.seed, args.max_files, args.max_frames
        )

    sample_motion_path = None
    with np.load(train_ds.files[0], allow_pickle=True) as sample:
        raw = sample["motion_path"]
        sample_motion_path = str(raw.item()) if raw.shape == () else str(raw[0])

    env = make_env(
        cfg,
        env_name="MyoFullBodyProsthesisEnv",
        motion_paths=[sample_motion_path],
        use_mujoco=True,
        fixed_start=True,
    )
    teacher_env = make_env(
        cfg,
        env_name="MyoFullBody",
        motion_paths=[sample_motion_path],
        use_mujoco=True,
        fixed_start=True,
    )
    mapping = load_mapping_for_dataset(dataset_dirs[0], teacher_env, env)
    validate_distill_dimensions(
        mapping,
        obs_dim=train_ds.obs_dim,
        n_remaining=train_ds.n_remaining_muscles,
    )
    env_obs_dim = int(env.info.observation_space.shape[0])
    if train_ds.obs_dim != env_obs_dim:
        raise ValueError(f"Dataset obs dim {train_ds.obs_dim} != imitation env obs dim {env_obs_dim}")
    if train_ds.n_remaining_muscles != mapping.n_remaining_muscles:
        raise ValueError(f"Dataset remaining dim {train_ds.n_remaining_muscles} != mapping {mapping.n_remaining_muscles}")
    if int(train_ds.data["target_prosthesis"].shape[-1]) != 4:
        raise ValueError("Expected prosthesis target dim 4")

    agent_conf, state = init_student_state(cfg, env, args)
    tx = state.tx
    train_step = make_train_step(
        agent_conf.network,
        tx,
        mapping.n_remaining_muscles,
        float(args.lambda_muscle),
        float(args.lambda_prosthesis),
        float(args.lambda_tau),
    )
    torque_limits = np.asarray(mapping.prosthesis_torque_limits, dtype=np.float32)

    run_dir = resolve_path(args.output_dir) / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    mapping.save_json(run_dir / "mapping.json")

    wb = None
    if args.wandb == "online":
        import wandb

        wb = wandb.init(project="prosthesis-distill", config=vars(args))

    rows = []
    for epoch in range(1, int(args.epochs) + 1):
        metric_accum = []
        for batch in train_ds.iter_batches(args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch_jax = {
                "obs": jnp.asarray(batch["obs"]),
                "target_remaining": jnp.asarray(batch["target_remaining"]),
                "target_prosthesis": jnp.asarray(batch["target_prosthesis"]),
                "target_tau": jnp.asarray(batch["target_tau"]),
                "tau_pd": jnp.asarray(batch["tau_pd"]),
                "torque_limits": jnp.asarray(torque_limits),
            }
            state, metrics = train_step(state, batch_jax)
            metric_accum.append({k: float(v) for k, v in metrics.items()})
        mean_metrics = {k: float(np.mean([m[k] for m in metric_accum])) for k in metric_accum[0]}
        val_loss, val_muscle, val_prosthesis, val_tau = eval_loss(
            agent_conf.network,
            state,
            val_ds,
            args.batch_size,
            mapping.n_remaining_muscles,
            float(args.lambda_muscle),
            float(args.lambda_prosthesis),
            float(args.lambda_tau),
            torque_limits,
        )
        row = {
            "epoch": epoch,
            **mean_metrics,
            "val_loss": float(val_loss),
            "val_muscle_mse": float(val_muscle),
            "val_prosthesis_mse": float(val_prosthesis),
            "val_prosthesis_tau_mse": float(val_tau),
            "prosthesis_action_clip_ratio": float(np.mean(np.abs(train_ds.data["target_prosthesis"]) >= 0.999)),
        }
        if val_dagger_ds is not None:
            dagger_loss, dagger_muscle, dagger_prosthesis, dagger_tau = eval_loss(
                agent_conf.network,
                state,
                val_dagger_ds,
                args.batch_size,
                mapping.n_remaining_muscles,
                float(args.lambda_muscle),
                float(args.lambda_prosthesis),
                float(args.lambda_tau),
                torque_limits,
            )
            row.update(
                {
                    "val_dagger_loss": float(dagger_loss),
                    "val_dagger_muscle_mse": float(dagger_muscle),
                    "val_dagger_prosthesis_mse": float(dagger_prosthesis),
                    "val_dagger_prosthesis_tau_mse": float(dagger_tau),
                }
            )
        rows.append(row)
        if wb is not None:
            wb.log(row, step=epoch)
        print(
            f"epoch={epoch:03d} loss={row['total_loss']:.6f} muscle={row['muscle_mse']:.6f} "
            f"prosthesis={row['prosthesis_mse']:.6f} val={row['val_loss']:.6f}"
            + (
                f" val_dagger={row['val_dagger_loss']:.6f}"
                if "val_dagger_loss" in row
                else ""
            )
        )

    save_loss_curve(run_dir / "loss_curve.csv", rows)
    metadata = {
        "learning_rate": args.lr,
        "distill_optimizer_step": int(state.step),
        "lambda_muscle": args.lambda_muscle,
        "lambda_prosthesis": args.lambda_prosthesis,
        "lambda_tau": args.lambda_tau,
        "dataset_dir": str(dataset_dirs[0]),
        "extra_dataset_dirs": [str(p) for p in dataset_dirs[1:]],
        "val_dataset_dirs": [str(resolve_path(p)) for p in args.val_dataset_dir],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "init": args.init,
        "last_metrics": rows[-1] if rows else {},
    }
    ckpt_path = save_distilled_checkpoint(
        checkpoint_dir=ckpt_dir,
        agent_conf=agent_conf,
        params=state.params,
        run_stats=state.run_stats,
        opt_state=state.opt_state,
        step=int(state.step),
        metadata=metadata,
    )
    latest = resolve_path(args.output_dir) / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        latest.write_text(str(run_dir), encoding="utf-8")
    (run_dir / "checkpoint_path.txt").write_text(str(ckpt_path), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"checkpoint": ckpt_path, "metadata": metadata}, indent=2), encoding="utf-8")
    if wb is not None:
        wb.finish()
    print(f"Saved distilled checkpoint: {ckpt_path}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
