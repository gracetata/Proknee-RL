#!/usr/bin/env python
"""Train a policy that outputs only remaining-muscle actions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze, unfreeze
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.distill.checkpoint import save_distilled_checkpoint, save_loss_curve
from musclemimic.distill.config import load_fullbody_config, make_env, repo_root
from musclemimic.distill.mapping import DistillMapping, build_distill_mapping
from musclemimic.distill.obs_mask import MaskedObsSpec, build_masked_obs_spec
from musclemimic.distill.remaining_only import RemainingOnlyDataset, RemainingOnlyEnvView
from musclemimic.runner.eval_utils import load_checkpoint
from train_prosthesis_distillation import assert_jax_gpu, merge_compatible_params

DEFAULT_DATASET_DIR = "data/remaining_only_reference4dof/KIT_KINESIS_TRAINING_MOTIONS"
DEFAULT_INIT_CHECKPOINT = (
    "outputs/full_muscle_masked_obs_dagger1_focused_round3_distill/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "outputs/remaining_only_reference4dof_distill"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--extra_dataset_dir", action="append", default=[])
    p.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--init_checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--wandb", choices=["disabled", "online"], default="disabled")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require_gpu", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def load_mapping(dataset_dir: Path, teacher_env, student_env) -> DistillMapping:
    mapping_path = dataset_dir / "mapping.json"
    mapping = DistillMapping.load_json(mapping_path) if mapping_path.exists() else build_distill_mapping(teacher_env, student_env)
    env_mapping = build_distill_mapping(teacher_env, student_env)
    if mapping.n_remaining_muscles != env_mapping.n_remaining_muscles:
        raise ValueError(f"Dataset n_remaining={mapping.n_remaining_muscles} != env {env_mapping.n_remaining_muscles}")
    return mapping


def load_mask_spec(dataset_dir: Path, env) -> MaskedObsSpec:
    path = dataset_dir / "masked_obs_spec.json"
    spec = MaskedObsSpec.load_json(path) if path.exists() else build_masked_obs_spec(env)
    if spec.masked_obs_dim != len(spec.keep_indices):
        raise ValueError("Invalid masked obs spec")
    return spec


def make_train_step(network, tx):
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
            muscle_mse = jnp.mean((pred - batch["target_remaining"]) ** 2)
            return muscle_mse, ({"total_loss": muscle_mse, "muscle_mse": muscle_mse}, updates["run_stats"])

        (loss, (metrics, run_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state.run_stats)
        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return state.replace(params=params, opt_state=opt_state, run_stats=run_stats, step=state.step + 1), {
            **metrics,
            "total_loss": loss,
        }

    return train_step


def eval_loss(network, state, dataset, batch_size):
    losses = []
    for batch in dataset.iter_batches(batch_size, shuffle=False):
        obs = jnp.asarray(batch["obs"])
        y, _updates = network.apply({"params": state.params, "run_stats": state.run_stats}, obs, mutable=["run_stats"])
        pi, _ = y
        pred = pi.mean()
        losses.append(float(jnp.mean((pred - jnp.asarray(batch["target_remaining"])) ** 2)))
    return float(np.mean(losses)) if losses else 0.0


def init_remaining_head_from_fullmuscle(params, source_params, mapping: DistillMapping):
    """Slice a 354-dim full-muscle actor head into the 335-dim remaining-muscle head."""
    target = unfreeze(params)
    source = unfreeze(source_params)
    rem_idx = np.asarray(mapping.teacher_remaining_action_indices, dtype=np.int32)
    try:
        src_actor = source["actor"]
        dst_actor = target["actor"]
        src_kernel = np.asarray(src_actor["output"]["kernel"])
        src_bias = np.asarray(src_actor["output"]["bias"])
        if src_kernel.shape[-1] == mapping.teacher_action_dim and dst_actor["output"]["kernel"].shape[-1] == mapping.n_remaining_muscles:
            dst_actor["output"]["kernel"] = jnp.asarray(src_kernel[:, rem_idx])
            dst_actor["output"]["bias"] = jnp.asarray(src_bias[rem_idx])
            print(
                "[init_checkpoint] initialized remaining-only actor.output "
                f"from full-muscle columns ({src_kernel.shape[-1]} -> {len(rem_idx)})"
            )
        if "log_std" in source and "log_std" in target:
            src_log_std = np.asarray(source["log_std"])
            if src_log_std.shape[0] == mapping.teacher_action_dim and target["log_std"].shape[0] == mapping.n_remaining_muscles:
                target["log_std"] = jnp.asarray(src_log_std[rem_idx])
                print("[init_checkpoint] initialized remaining-only log_std from full-muscle log_std")
    except KeyError as exc:
        print(f"[init_checkpoint] warning: could not slice remaining head: missing {exc}")
    return freeze(target)


def init_input_prefix_from_checkpoint(params, source_params, source_obs_dim: int):
    """Copy old observation rows into widened input kernels and keep new feature rows random."""
    target = unfreeze(params)
    source = unfreeze(source_params)
    copied = 0

    def visit(dst_node, src_node):
        nonlocal copied
        if not isinstance(dst_node, dict) or not isinstance(src_node, dict):
            return
        for key, dst_value in dst_node.items():
            if key not in src_node:
                continue
            src_value = src_node[key]
            if isinstance(dst_value, dict) and isinstance(src_value, dict):
                visit(dst_value, src_value)
                continue
            dst_arr = np.asarray(dst_value)
            src_arr = np.asarray(src_value)
            if (
                dst_arr.ndim == 2
                and src_arr.ndim == 2
                and src_arr.shape[0] == int(source_obs_dim)
                and dst_arr.shape[0] > src_arr.shape[0]
                and dst_arr.shape[1] == src_arr.shape[1]
            ):
                merged = dst_arr.copy()
                merged[: src_arr.shape[0], :] = src_arr
                dst_node[key] = jnp.asarray(merged)
                copied += 1

    visit(target, source)
    if copied:
        print(f"[init_checkpoint] copied old observation rows into {copied} widened input kernels")
    return freeze(target)


def init_state(cfg, env_view, args, mapping: DistillMapping, source_obs_dim: int):
    agent_conf = PPOJax.init_agent_conf(env_view, cfg)
    rng = jax.random.PRNGKey(args.seed)
    init_vars = agent_conf.network.init(rng, jnp.zeros(env_view.info.observation_space.shape, dtype=jnp.float32))
    params = init_vars["params"]
    run_stats = init_vars["run_stats"]
    if args.init_checkpoint:
        ckpt = resolve_path(args.init_checkpoint)
        _cfg, agent_state, _meta = load_checkpoint(str(ckpt))
        source_params = agent_state.train_state.params
        params, warnings = merge_compatible_params(params, source_params)
        params = init_input_prefix_from_checkpoint(params, source_params, source_obs_dim)
        params = init_remaining_head_from_fullmuscle(params, source_params, mapping)
        if warnings:
            print(f"[init_checkpoint] skipped {len(warnings)} incompatible param leaves")
            for name, s_shape, t_shape in warnings[:20]:
                print(f"  {name}: student={s_shape} checkpoint={t_shape}")
        try:
            run_stats, rs_warnings = merge_compatible_params(run_stats, agent_state.train_state.run_stats)
            if rs_warnings:
                print(f"[init_checkpoint] skipped {len(rs_warnings)} incompatible run_stats leaves")
        except Exception as exc:
            print(f"[init_checkpoint] warning: could not merge run_stats: {exc}")
    tx = optax.adamw(float(args.lr))
    state = TrainState.create(apply_fn=agent_conf.network.apply, params=params, run_stats=run_stats, tx=tx)
    return agent_conf, state


def main() -> int:
    args = parse_args()
    assert_jax_gpu(args.require_gpu)
    cfg = load_fullbody_config(args.config_name)
    OmegaConf.set_struct(cfg, False)
    dataset_dirs = [resolve_path(args.dataset_dir), *[resolve_path(p) for p in args.extra_dataset_dir]]
    train_ds = RemainingOnlyDataset(dataset_dirs, "train", args.val_fraction, args.seed, args.max_files, args.max_frames)
    val_ds = RemainingOnlyDataset(dataset_dirs, "val", args.val_fraction, args.seed, args.max_files, args.max_frames)

    with np.load(train_ds.files[0], allow_pickle=True) as sample:
        raw = sample["motion_path"]
        sample_motion_path = str(raw.item()) if raw.shape == () else str(raw[0])

    env = make_env(cfg, env_name="MyoFullBody", motion_paths=[sample_motion_path], use_mujoco=True, fixed_start=True)
    student_env = make_env(
        cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[sample_motion_path], use_mujoco=True, fixed_start=True
    )
    mapping = load_mapping(dataset_dirs[0], env, student_env)
    spec = load_mask_spec(dataset_dirs[0], env)
    if train_ds.n_remaining_muscles != mapping.n_remaining_muscles:
        raise ValueError(
            f"Dataset remaining dim {train_ds.n_remaining_muscles} != mapping {mapping.n_remaining_muscles}"
        )
    env_view = RemainingOnlyEnvView(env, spec, mapping.n_remaining_muscles)
    expected_obs_dim = int(env_view.info.observation_space.shape[0])
    if train_ds.obs_dim != expected_obs_dim:
        raise ValueError(f"Dataset obs dim {train_ds.obs_dim} != policy obs dim {expected_obs_dim}")
    agent_conf, state = init_state(cfg, env_view, args, mapping, spec.masked_obs_dim)
    train_step = make_train_step(agent_conf.network, state.tx)

    run_dir = resolve_path(args.output_dir) / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    mapping.save_json(run_dir / "mapping.json")
    spec.save_json(run_dir / "masked_obs_spec.json")

    wb = None
    if args.wandb == "online":
        import wandb

        wb = wandb.init(project="remaining-only-reference4dof-distill", config=vars(args))

    rows = []
    for epoch in range(1, int(args.epochs) + 1):
        metric_accum = []
        for batch in train_ds.iter_batches(args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch_jax = {
                "obs": jnp.asarray(batch["obs"]),
                "target_remaining": jnp.asarray(batch["target_remaining"]),
            }
            state, metrics = train_step(state, batch_jax)
            metric_accum.append({k: float(v) for k, v in metrics.items()})
        mean_metrics = {k: float(np.mean([m[k] for m in metric_accum])) for k in metric_accum[0]}
        val_mse = eval_loss(agent_conf.network, state, val_ds, args.batch_size)
        row = {"epoch": epoch, **mean_metrics, "val_muscle_mse": float(val_mse)}
        rows.append(row)
        if wb is not None:
            wb.log(row, step=epoch)
        print(f"epoch={epoch:03d} loss={row['total_loss']:.6f} val={row['val_muscle_mse']:.6f}")

    metadata = {
        "distill_type": "remaining_only_reference4dof",
        "learning_rate": float(args.lr),
        "distill_optimizer_step": int(state.step),
        "dataset_dirs": [str(p) for p in dataset_dirs],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "init_checkpoint": str(args.init_checkpoint),
        "n_remaining_muscles": int(mapping.n_remaining_muscles),
        "policy_obs_dim": int(expected_obs_dim),
        "masked_obs_dim": int(spec.masked_obs_dim),
        "prosthesis_state_feature_dim": int(expected_obs_dim - spec.masked_obs_dim),
        "disabled_muscle_names": list(mapping.disabled_muscle_names),
        "prosthesis_joint_names": list(mapping.prosthesis_joint_names),
        "masked_obs_spec": spec.__dict__,
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
    save_loss_curve(run_dir / "loss_curve.csv", rows)
    (run_dir / "checkpoint_path.txt").write_text(str(ckpt_path), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"checkpoint": ckpt_path, "metadata": metadata}, indent=2), encoding="utf-8")
    if wb is not None:
        wb.finish()
    env.stop()
    student_env.stop()
    print(f"Saved remaining-only distilled checkpoint: {ckpt_path}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
