#!/usr/bin/env python
"""Supervised split-action prosthesis distillation.

The actor has explicit remaining-muscle and prosthesis heads, then concatenates
them into the MyoFullBodyProsthesisEnv action order.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.distill.checkpoint import save_distilled_checkpoint, save_loss_curve
from musclemimic.distill.config import load_fullbody_config, make_env, repo_root
from musclemimic.distill.dataset import SplitActionProsthesisDataset
from musclemimic.distill.mapping import DistillMapping, build_distill_mapping, validate_distill_dimensions
from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint
from train_prosthesis_distillation import (
    DEFAULT_TEACHER_CHECKPOINT,
    assert_jax_gpu,
    eval_loss,
    init_student_state,
    make_train_step,
    merge_compatible_params,
)


DEFAULT_DATASET_DIR = "data/split_action_prosthesis_distill/KIT_KINESIS_TRAINING_MOTIONS"
DEFAULT_OUTPUT_DIR = "outputs/split_action_prosthesis_distill"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--extra_dataset_dir", action="append", default=[])
    parser.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_prosthesis", type=float, default=10.0)
    parser.add_argument("--init", choices=["random", "official_trunk_init", "checkpoint"], default="official_trunk_init")
    parser.add_argument("--init_checkpoint", default=None, help="Initialize from a split-action distilled checkpoint")
    parser.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb", choices=["disabled", "online"], default="disabled")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def load_mapping_for_dataset(dataset_dir: Path, teacher_env, student_env) -> DistillMapping:
    mapping_path = dataset_dir / "mapping.json"
    mapping = DistillMapping.load_json(mapping_path) if mapping_path.exists() else build_distill_mapping(teacher_env, student_env)
    env_mapping = build_distill_mapping(teacher_env, student_env)
    if mapping.student_obs_dim != env_mapping.student_obs_dim:
        raise ValueError(f"Dataset mapping student_obs_dim={mapping.student_obs_dim} != env {env_mapping.student_obs_dim}")
    if mapping.n_remaining_muscles != env_mapping.n_remaining_muscles:
        raise ValueError(f"Dataset mapping n_remaining={mapping.n_remaining_muscles} != env {env_mapping.n_remaining_muscles}")
    return mapping


def enable_split_action_actor(cfg, mapping: DistillMapping) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.split_action_actor = {
        "enabled": True,
        "remaining_action_dim": int(mapping.n_remaining_muscles),
        "prosthesis_action_dim": int(len(mapping.prosthesis_joint_names)),
    }


def init_split_student_state(cfg, env, args, mapping: DistillMapping):
    enable_split_action_actor(cfg, mapping)
    agent_conf = PPOJax.init_agent_conf(env, cfg)
    rng = jax.random.PRNGKey(args.seed)
    init_vars = agent_conf.network.init(rng, jnp.zeros(env.info.observation_space.shape, dtype=jnp.float32))
    params = init_vars["params"]
    run_stats = init_vars["run_stats"]
    if args.init == "checkpoint":
        if not args.init_checkpoint:
            raise ValueError("--init checkpoint requires --init_checkpoint")
        ckpt_path = resolve_path(args.init_checkpoint)
        _cfg, agent_state, _meta = load_checkpoint(str(ckpt_path))
        params, warnings = merge_compatible_params(params, agent_state.train_state.params)
        if warnings:
            print(f"[init_checkpoint] skipped {len(warnings)} incompatible param leaves")
        try:
            run_stats, rs_warnings = merge_compatible_params(run_stats, agent_state.train_state.run_stats)
            if rs_warnings:
                print(f"[init_checkpoint] skipped {len(rs_warnings)} incompatible run_stats leaves")
        except Exception as exc:
            print(f"[init_checkpoint] warning: could not merge run_stats: {exc}")
    elif args.init == "official_trunk_init":
        agent_conf, state = init_student_state(cfg, env, args)
        return agent_conf, state
    import optax
    tx = optax.adamw(args.lr)
    state = TrainState.create(apply_fn=agent_conf.network.apply, params=params, run_stats=run_stats, tx=tx)
    return agent_conf, state


def main() -> int:
    args = parse_args()
    assert_jax_gpu(args.require_gpu)

    cfg = load_fullbody_config(args.config_name)
    OmegaConf.set_struct(cfg, False)
    dataset_dirs = [resolve_path(args.dataset_dir), *[resolve_path(p) for p in args.extra_dataset_dir]]
    train_ds = SplitActionProsthesisDataset(dataset_dirs, "train", args.val_fraction, args.seed, args.max_files, args.max_frames)
    val_ds = SplitActionProsthesisDataset(dataset_dirs, "val", args.val_fraction, args.seed, args.max_files, args.max_frames)

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
    validate_distill_dimensions(mapping, obs_dim=train_ds.obs_dim, n_remaining=train_ds.n_remaining_muscles)
    if train_ds.obs_dim != int(env.info.observation_space.shape[0]):
        raise ValueError(f"Dataset obs dim {train_ds.obs_dim} != imitation env obs dim {env.info.observation_space.shape[0]}")
    if int(train_ds.data["target_prosthesis"].shape[-1]) != len(mapping.prosthesis_joint_names):
        raise ValueError("Expected prosthesis target dim 4")

    enable_split_action_actor(cfg, mapping)
    agent_conf, state = init_split_student_state(cfg, env, args, mapping)
    train_step = make_train_step(agent_conf.network, state.tx, mapping.n_remaining_muscles, float(args.lambda_prosthesis))
    torque_limits = np.asarray(mapping.prosthesis_torque_limits, dtype=np.float32)

    run_dir = resolve_path(args.output_dir) / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    mapping.save_json(run_dir / "mapping.json")

    wb = None
    if args.wandb == "online":
        import wandb

        wb = wandb.init(project="split-action-prosthesis-distill", config=vars(args))

    rows = []
    for epoch in range(1, int(args.epochs) + 1):
        metric_accum = []
        for batch in train_ds.iter_batches(args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch_jax = {
                "obs": jax.numpy.asarray(batch["obs"]),
                "target_remaining": jax.numpy.asarray(batch["target_remaining"]),
                "target_prosthesis": jax.numpy.asarray(batch["target_prosthesis"]),
                "target_tau": jax.numpy.asarray(batch["target_tau"]),
                "torque_limits": jax.numpy.asarray(torque_limits),
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
            float(args.lambda_prosthesis),
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
        rows.append(row)
        if wb is not None:
            wb.log(row, step=epoch)
        print(
            f"epoch={epoch:03d} loss={row['total_loss']:.6f} muscle={row['muscle_mse']:.6f} "
            f"prosthesis={row['prosthesis_mse']:.6f} val={row['val_loss']:.6f}"
        )

    save_loss_curve(run_dir / "loss_curve.csv", rows)
    metadata = {
        "split_action": True,
        "learning_rate": args.lr,
        "distill_optimizer_step": int(state.step),
        "lambda_prosthesis": args.lambda_prosthesis,
        "dataset_dirs": [str(p) for p in dataset_dirs],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "init": args.init,
        "n_remaining_muscles": int(mapping.n_remaining_muscles),
        "prosthesis_action_dim": int(len(mapping.prosthesis_joint_names)),
        "disabled_muscle_names": list(mapping.disabled_muscle_names),
        "prosthesis_joint_names": list(mapping.prosthesis_joint_names),
        "prosthesis_torque_limits": list(mapping.prosthesis_torque_limits),
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
    print(f"Saved split-action distilled checkpoint: {ckpt_path}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
