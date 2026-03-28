#!/usr/bin/env python3
"""
Stage 1: Teacher Policy Training (Standalone Script)

Trains a prosthetic knee teacher policy with privileged information.
The frozen Stage 0 body policy controls 24 healthy DOFs,
while the teacher learns to control the left knee (1 DOF).
Ankle joints are locked (passive).

Usage:
    # CPU test (small scale)
    python scripts/train_stage1.py --device cpu --num-envs 16 --max-epochs 10

    # GPU training (full scale)
    python scripts/train_stage1.py --device cuda:0 --num-envs 4096 --max-epochs 3000
"""

# ── Isaac Gym MUST be imported before torch ──────────────────────────
import isaacgym  # noqa: F401

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from proknee_hora.envs.proknee_teacher import ProKneeTeacher
from proknee_hora.algo.models import ProKneePolicy
from proknee_hora.algo.ppo import PPO


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Teacher Training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--body-policy", type=str,
                        default="outputs/checkpoints/stage0/stage0_amp_walk_5050.pth")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    # PPO hyperparams
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--mini-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--e-clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--init-noise-std", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Banner ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  ProKnee Stage 1: Teacher Policy Training")
    print("=" * 60)
    print(f"  Device:       {args.device}")
    print(f"  Num envs:     {args.num_envs}")
    print(f"  Max epochs:   {args.max_epochs}")
    print(f"  Body policy:  {args.body_policy}")
    print(f"  LR:           {args.lr}")
    print(f"  Horizon:      {args.horizon}")
    print(f"  Minibatch:    {args.minibatch_size}")
    print("=" * 60)

    # ── Output directory ──────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            PROJECT_ROOT, "outputs", f"stage1_teacher_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"  Output dir:   {args.output_dir}")

    # ── Create environment ────────────────────────────────────────────
    print("\n[Stage 1] Creating ProKneeTeacher environment...")
    t0 = time.time()
    env = ProKneeTeacher(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        body_policy_checkpoint=args.body_policy,
    )
    print(f"[Stage 1] Environment created in {time.time()-t0:.1f}s")
    print(f"  num_actions={env.num_actions}, num_obs={env.num_obs}")

    # ── Create policy ─────────────────────────────────────────────────
    print("[Stage 1] Creating ProKneePolicy (teacher mode)...")
    from proknee_hora.envs.constants import TEACHER_PRIV_INFO_DIM, STUDENT_PROPRIO_DIM, LATENT_DIM
    policy = ProKneePolicy(
        obs_dim=env.num_obs,           # 10 (proprio, deployment-available)
        action_dim=env.num_actions,    # 1 (knee only)
        priv_info_dim=TEACHER_PRIV_INFO_DIM,  # 113
        proprio_dim=STUDENT_PROPRIO_DIM,      # 10
        history_len=30,
        latent_dim=LATENT_DIM,                # 32
        hidden_dims=[256, 128, 64],
        mode='teacher',
    ).to(args.device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {num_params:,}")

    # ── Create PPO trainer ────────────────────────────────────────────
    trainer = PPO(
        policy=policy,
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        e_clip=args.e_clip,
        entropy_coef=args.entropy_coef,
        value_loss_coef=1.0,
        max_grad_norm=1.0,
        horizon_length=args.horizon,
        minibatch_size=args.minibatch_size,
        mini_epochs=args.mini_epochs,
        device=args.device,
        use_priv_info=True,
        priv_info_dim=TEACHER_PRIV_INFO_DIM,
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        print(f"[Stage 1] Resuming from {args.resume}")
        trainer.load(args.resume)
        start_epoch = trainer.epoch + 1
        print(f"  Resuming from epoch {start_epoch}")

    # ── Reset environment ─────────────────────────────────────────────
    print("[Stage 1] Resetting environment...")
    env.reset()

    # ── TensorBoard ───────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[Stage 1] TensorBoard: tensorboard --logdir {tb_dir}")
    except ImportError:
        writer = None
        print("[Stage 1] TensorBoard not available")

    # ── Training loop ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training started!")
    print("=" * 60)
    best_reward = float('-inf')
    train_start = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        # Anneal action noise: 0.5 → 0.1 over first 2000 epochs
        target_std = max(0.1, 0.5 * (1 - epoch / 2000))
        with torch.no_grad():
            policy.actor_critic.log_std.fill_(np.log(target_std))

        # Collect rollout
        rollout_stats = trainer.collect_rollout()
        # Update policy
        update_stats = trainer.update()
        trainer.epoch = epoch

        reward_mean = rollout_stats['reward_mean']
        avg_ep_len = rollout_stats['avg_ep_len']
        policy_loss = update_stats['policy_loss']
        value_loss = update_stats['value_loss']
        entropy = update_stats['entropy']

        # TensorBoard
        if writer:
            writer.add_scalar("reward/mean", reward_mean, epoch)
            writer.add_scalar("reward/avg_ep_len", avg_ep_len, epoch)
            writer.add_scalar("reward/action_std", target_std, epoch)
            writer.add_scalar("loss/policy", policy_loss, epoch)
            writer.add_scalar("loss/value", value_loss, epoch)
            writer.add_scalar("loss/entropy", entropy, epoch)

        # Logging
        if epoch % args.log_interval == 0:
            elapsed = time.time() - train_start
            eps = (epoch - start_epoch + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  Epoch {epoch:5d}/{args.max_epochs} | "
                f"reward={reward_mean:7.3f} | "
                f"ep_len={avg_ep_len:6.1f} | "
                f"std={target_std:.3f} | "
                f"p_loss={policy_loss:.4f} | "
                f"v_loss={value_loss:.4f} | "
                f"ent={entropy:.4f} | "
                f"{eps:.1f} ep/s"
            )

        # Save best
        if reward_mean > best_reward:
            best_reward = reward_mean
            trainer.save(os.path.join(ckpt_dir, "best.pth"))

        # Periodic save
        if epoch > 0 and epoch % args.save_interval == 0:
            trainer.save(os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

    # ── Final save ────────────────────────────────────────────────────
    trainer.save(os.path.join(ckpt_dir, "final.pth"))
    if writer:
        writer.close()

    elapsed = time.time() - train_start
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Total time:   {elapsed/3600:.2f} hours")
    print(f"  Best reward:  {best_reward:.4f}")
    print(f"  Final ckpt:   {ckpt_dir}/final.pth")
    print(f"  Best ckpt:    {ckpt_dir}/best.pth")
    print("=" * 60)

    # Copy best to standard location
    import shutil
    std_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "stage1", "best.pth")
    os.makedirs(os.path.dirname(std_path), exist_ok=True)
    shutil.copy2(os.path.join(ckpt_dir, "best.pth"), std_path)
    print(f"  Copied best → {std_path}")


if __name__ == "__main__":
    main()
