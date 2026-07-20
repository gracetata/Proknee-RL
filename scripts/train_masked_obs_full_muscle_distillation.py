#!/usr/bin/env python
"""Supervised distillation: masked observation -> full 354-muscle teacher action."""

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
from musclemimic.distill.config import load_fullbody_config, make_env, repo_root
from musclemimic.distill.full_muscle_dataset import FullMuscleMaskedDistillDataset
from musclemimic.distill.obs_mask import MaskedObservationEnvView, MaskedObsSpec, build_masked_obs_spec
from musclemimic.runner.eval_utils import load_checkpoint

DEFAULT_TEACHER_CHECKPOINT = "/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2"
DEFAULT_DATASET_DIR = "data/full_muscle_masked_obs_rollouts/KIT_KINESIS_TRAINING_MOTIONS"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    p.add_argument(
        "--extra_dataset_dir",
        action="append",
        default=[],
        help="Additional rollout directory to append to the training/validation set. Repeat for multiple DAgger rounds.",
    )
    p.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init", choices=["random", "official_trunk_init"], default="official_trunk_init")
    p.add_argument(
        "--init_checkpoint",
        default=None,
        help="Optional distilled checkpoint to continue from. Overrides --init when provided.",
    )
    p.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    p.add_argument("--output_dir", default="outputs/full_muscle_masked_obs_distill")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require_gpu", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def assert_jax_gpu(required: bool) -> None:
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    has_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    if required and not has_gpu:
        raise RuntimeError("GPU required for distillation but jax.devices() shows no GPU. Use --no-require_gpu for smoke tests.")


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


def init_student_state(cfg, env_view, args):
    agent_conf = PPOJax.init_agent_conf(env_view, cfg)
    rng = jax.random.PRNGKey(args.seed)
    init_vars = agent_conf.network.init(rng, jnp.zeros(env_view.info.observation_space.shape, dtype=jnp.float32))
    params = init_vars["params"]
    run_stats = init_vars["run_stats"]
    if args.init_checkpoint:
        _student_cfg, student_state, _ = load_checkpoint(args.init_checkpoint)
        params, warnings = merge_compatible_params(params, student_state.train_state.params)
        if warnings:
            print(f"[init_checkpoint] skipped {len(warnings)} incompatible leaves")
            for name, s_shape, t_shape in warnings[:20]:
                print(f"  {name}: student={s_shape} checkpoint={t_shape}")
        try:
            run_stats, rs_warnings = merge_compatible_params(run_stats, student_state.train_state.run_stats)
            if rs_warnings:
                print(f"[init_checkpoint] skipped {len(rs_warnings)} incompatible run_stats leaves")
        except Exception as exc:
            print(f"[init_checkpoint] warning: could not merge run_stats: {exc}")
    elif args.init == "official_trunk_init":
        _teacher_cfg, teacher_state, _ = load_checkpoint(args.teacher_checkpoint)
        params, warnings = merge_compatible_params(params, teacher_state.train_state.params)
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
    tx = optax.adamw(args.lr)
    state = TrainState.create(apply_fn=agent_conf.network.apply, params=params, run_stats=run_stats, tx=tx)
    return agent_conf, state


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
            action_mse = jnp.mean((pred - batch["target_action"]) ** 2)
            return action_mse, ({"total_loss": action_mse, "action_mse": action_mse}, updates["run_stats"])

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
        y, _updates = network.apply(
            {"params": state.params, "run_stats": state.run_stats},
            jnp.asarray(batch["obs"]),
            mutable=["run_stats"],
        )
        pi, _ = y
        losses.append(float(jnp.mean((pi.mean() - jnp.asarray(batch["target_action"])) ** 2)))
    return float(np.mean(losses)) if losses else 0.0


def main() -> int:
    args = parse_args()
    assert_jax_gpu(args.require_gpu)
    cfg = load_fullbody_config(args.config_name)
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.env_params.env_name = "MyoFullBody"
    cfg.experiment.env_params.pop("prosthesis", None)
    dataset_dir = resolve_path(args.dataset_dir)
    dataset_dirs = [dataset_dir] + [resolve_path(path) for path in args.extra_dataset_dir]
    train_ds = FullMuscleMaskedDistillDataset(dataset_dirs, "train", args.val_fraction, args.seed, args.max_files, args.max_frames)
    val_ds = FullMuscleMaskedDistillDataset(dataset_dirs, "val", args.val_fraction, args.seed, args.max_files, args.max_frames)

    with np.load(train_ds.files[0], allow_pickle=True) as sample:
        raw = sample["motion_path"]
        sample_motion_path = str(raw.item()) if raw.shape == () else str(raw[0])

    env = make_env(cfg, env_name="MyoFullBody", motion_paths=[sample_motion_path], use_mujoco=True, fixed_start=True)
    spec_path = dataset_dir / "masked_obs_spec.json"
    spec = MaskedObsSpec.load_json(spec_path) if spec_path.exists() else build_masked_obs_spec(env)
    env_view = MaskedObservationEnvView(env, spec)
    if train_ds.obs_dim != spec.masked_obs_dim:
        raise ValueError(f"Dataset obs dim {train_ds.obs_dim} != mask spec {spec.masked_obs_dim}")
    if train_ds.action_dim != int(env.info.action_space.shape[0]):
        raise ValueError(f"Dataset action dim {train_ds.action_dim} != env action dim {env.info.action_space.shape[0]}")

    agent_conf, state = init_student_state(cfg, env_view, args)
    train_step = make_train_step(agent_conf.network, state.tx)
    run_dir = resolve_path(args.output_dir) / datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    spec.save_json(run_dir / "masked_obs_spec.json")

    rows = []
    for epoch in range(1, int(args.epochs) + 1):
        metric_accum = []
        for batch in train_ds.iter_batches(args.batch_size, shuffle=True, seed=args.seed + epoch):
            state, metrics = train_step(
                state,
                {
                    "obs": jnp.asarray(batch["obs"]),
                    "target_action": jnp.asarray(batch["target_action"]),
                },
            )
            metric_accum.append({k: float(v) for k, v in metrics.items()})
        mean_metrics = {k: float(np.mean([m[k] for m in metric_accum])) for k in metric_accum[0]}
        val_action_mse = eval_loss(agent_conf.network, state, val_ds, args.batch_size)
        row = {"epoch": epoch, **mean_metrics, "val_loss": val_action_mse, "val_action_mse": val_action_mse}
        rows.append(row)
        print(f"epoch={epoch:03d} loss={row['total_loss']:.6f} val={row['val_loss']:.6f}")

    save_loss_curve(run_dir / "loss_curve.csv", rows)
    metadata = {
        "distill_type": "full_muscle_masked_obs",
        "learning_rate": args.lr,
        "distill_optimizer_step": int(state.step),
        "dataset_dir": str(dataset_dir),
        "extra_dataset_dir": [str(path) for path in dataset_dirs[1:]],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "init": "init_checkpoint" if args.init_checkpoint else args.init,
        "init_checkpoint": args.init_checkpoint,
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
    (run_dir / "checkpoint_path.txt").write_text(str(ckpt_path), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"checkpoint": ckpt_path, "metadata": metadata}, indent=2), encoding="utf-8")
    env.stop()
    print(f"Saved distilled checkpoint: {ckpt_path}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
