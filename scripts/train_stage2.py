#!/usr/bin/env python3
"""
Stage 2: Hora-faithful supervised latent distillation for ProKnee.

Trains ONLY the adapt_tconv module to predict latent from proprio_hist.
Everything else (backbone, actor, critic, priv_mlp) is frozen from Stage 1.
Pure MSE loss on latent — NO PPO.

Usage:
    # CPU test (defaults to checkpoints/stage0 and checkpoints/stage1)
    python scripts/train_stage2.py --device cpu --num-envs 4

    # GPU training
    PYTHONUNBUFFERED=1 nohup python scripts/train_stage2.py \
        --device cuda:0 --num-envs 4096 \
        > outputs/stage2_train.log 2>&1 &
"""
import os
import sys
import argparse
import datetime
import shutil
import time
import logging
import numpy as np

if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

# Isaac Gym MUST be imported before torch
from isaacgym import gymapi  # noqa: F401

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_base import ProKneeBase
from proknee_hora.algo.proprio_adapt import ProprioAdapt
from proknee_hora.envs.constants import (
    OBS_DIM, TEACHER_PRIV_INFO_DIM, STUDENT_PROPRIO_DIM,
    LATENT_DIM, PROPRIO_HISTORY_LEN,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('train_stage2')

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     ProKnee Stage 2: Hora-Faithful Latent Distillation      ║
║     adapt_tconv(proprio_hist) → latent ≈ priv_mlp(priv)     ║
║     Pure MSE loss — NO PPO — Only adapt_tconv trained        ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 supervised latent distillation")
    p.add_argument("--body-policy", type=str,
                    default="outputs/checkpoints/stage0/stage0_amp_walk_5050.pth",
                    help="Stage 0 body policy checkpoint")
    p.add_argument("--teacher-ckpt", type=str,
                    default="outputs/checkpoints/stage1/best.pth",
                    help="Stage 1 teacher checkpoint")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=int(5e8), help="Max agent steps")
    p.add_argument("--save-interval", type=int, default=int(5e7), help="Save every N agent steps")
    p.add_argument("--log-interval", type=int, default=int(1e6), help="Log to TB every N agent steps")
    p.add_argument("--episode-length", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    print(BANNER)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "outputs", f"stage2_padapt_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Config] device={args.device}, num_envs={args.num_envs}")
    print(f"[Config] lr={args.lr}, max_steps={args.max_steps:.0e}")
    print(f"[Config] body_policy={args.body_policy}")
    print(f"[Config] teacher_ckpt={args.teacher_ckpt}")
    print(f"[Config] output_dir={out_dir}")
    print(f"[Config] obs_dim={OBS_DIM}, priv_info_dim={TEACHER_PRIV_INFO_DIM}, "
          f"proprio_dim={STUDENT_PROPRIO_DIM}, latent_dim={LATENT_DIM}, "
          f"hist_len={PROPRIO_HISTORY_LEN}")
    print()

    # ── Create environment ──────────────────────────────────────────
    print("[1/3] Creating ProKneeBase environment...")
    t0 = time.time()
    env = ProKneeBase(
        num_envs=args.num_envs,
        device=args.device,
        headless=True,
        body_policy_checkpoint=args.body_policy,
        prosthesis_only=True,
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=args.episode_length,
    )
    print(f"  Environment created in {time.time() - t0:.1f}s")
    print(f"  num_obs={env.num_obs}, num_actions={env.num_actions}")

    # ── Create ProprioAdapt trainer ─────────────────────────────────
    print("[2/3] Creating ProprioAdapt trainer...")
    trainer = ProprioAdapt(
        env=env,
        output_dir=out_dir,
        device=args.device,
        lr=args.lr,
        max_agent_steps=args.max_steps,
        save_interval_steps=args.save_interval,
        log_interval_steps=args.log_interval,
    )

    # ── Load Stage 1 teacher weights ────────────────────────────────
    print("[3/3] Loading Stage 1 teacher model...")
    trainer.restore_stage1(args.teacher_ckpt)

    # Verify parameter freeze
    n_adapt = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    n_frozen = n_total - n_adapt
    print(f"  Total params:     {n_total:,}")
    print(f"  Frozen params:    {n_frozen:,}")
    print(f"  Trainable (tconv): {n_adapt:,}")

    print()
    print("=" * 60)
    print("  Training started!")
    print(f"  Target: {args.max_steps:.0e} agent steps ({args.num_envs} envs)")
    print(f"  TensorBoard: tensorboard --logdir {os.path.join(out_dir, 'tb')}")
    print(f"  Interrupt: Ctrl+C or kill <PID>")
    print("=" * 60)
    print()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")

    # ── Final save ──────────────────────────────────────────────────
    final_path = trainer.save(os.path.join(trainer.nn_dir, 'final'))
    print(f"\n  Final checkpoint: {final_path}")

    # Copy best to canonical location
    canonical_dir = os.path.join(ROOT, "outputs", "checkpoints", "stage2")
    os.makedirs(canonical_dir, exist_ok=True)
    best_src = os.path.join(trainer.nn_dir, 'best.pth')
    if os.path.exists(best_src):
        shutil.copy2(best_src, os.path.join(canonical_dir, "best.pth"))
        print(f"  Best checkpoint → {canonical_dir}/best.pth")

    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Agent steps:   {trainer.agent_steps:,}")
    print(f"  Best reward:   {trainer.best_rewards:.2f}")
    print(f"  Checkpoints:   {trainer.nn_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
