#!/usr/bin/env python3
"""
Stage 1: Teacher Policy Training via DAgger-style Behavioral Cloning.

Instead of PPO, directly clone the body policy's knee action using supervised
learning. The teacher rolls out its OWN actions (not the body policy's) so
it learns to recover from its own distribution of states — key DAgger insight.

The body policy provides the ground truth knee action for each state.

Usage:
    # CPU test
    python scripts/train_stage1_dagger.py --device cpu --num-envs 16 --max-epochs 10

    # GPU training
    nohup python scripts/train_stage1_dagger.py --device cuda:0 --num-envs 4096 --max-epochs 5000 \
        > outputs/stage1_dagger.log 2>&1 &
"""

# ── Isaac Gym MUST be imported before torch ──────────────────────────
import isaacgym  # noqa: F401

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from proknee_hora.envs.proknee_base import ProKneeBase, load_body_policy
from proknee_hora.envs.constants import (
    OBS_DIM, TEACHER_PRIV_INFO_DIM, STUDENT_PROPRIO_DIM,
    LATENT_DIM, PROPRIO_HISTORY_LEN, ACTIVE_PROSTHESIS_JOINTS,
    PROSTHESIS_ACTION_DIM,
)
from proknee_hora.algo.models import ProKneePolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 DAgger Training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--steps-per-epoch", type=int, default=32,
                        help="Env steps per epoch (like PPO horizon)")
    parser.add_argument("--body-policy", type=str,
                        default="outputs/checkpoints/stage0/stage0_amp_walk_5050.pth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3,
                        help="Exploration noise std (annealed to 0)")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Banner ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  ProKnee Stage 1: DAgger-style Behavioral Cloning")
    print("=" * 60)
    print(f"  Device:       {args.device}")
    print(f"  Num envs:     {args.num_envs}")
    print(f"  Max epochs:   {args.max_epochs}")
    print(f"  Steps/epoch:  {args.steps_per_epoch}")
    print(f"  Body policy:  {args.body_policy}")
    print(f"  LR:           {args.lr}")
    print(f"  Noise std:    {args.noise_std}")
    print("=" * 60)

    # ── Output directory ──────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            PROJECT_ROOT, "outputs", f"stage1_dagger_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"  Output dir:   {args.output_dir}")

    # ── Create environment ────────────────────────────────────────────
    print("\n[Stage 1] Creating environment...")
    t0 = time.time()
    env = ProKneeBase(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        body_policy_checkpoint=args.body_policy,
        prosthesis_only=True,
        episode_length=300,
    )
    print(f"[Stage 1] Environment created in {time.time()-t0:.1f}s")

    # ── Load body policy (for ground truth knee action) ───────────────
    body_policy = load_body_policy(args.body_policy, args.device)
    body_policy.eval()
    print("[Stage 1] Body policy loaded for target knee action")

    # ── Create teacher policy ─────────────────────────────────────────
    print("[Stage 1] Creating ProKneePolicy (teacher mode)...")
    policy = ProKneePolicy(
        obs_dim=OBS_DIM,
        action_dim=PROSTHESIS_ACTION_DIM,
        priv_info_dim=TEACHER_PRIV_INFO_DIM,
        proprio_dim=STUDENT_PROPRIO_DIM,
        history_len=PROPRIO_HISTORY_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64],
        mode='teacher',
    ).to(args.device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {num_params:,}")

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(ckpt['policy_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[Stage 1] Resumed from epoch {start_epoch}")

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

    # ── Training loop ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training started!")
    print("=" * 60)
    best_eval_len = 0
    train_start = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        # Anneal exploration noise: noise_std → 0 over first 2000 epochs
        noise_std = max(0.0, args.noise_std * (1 - epoch / 2000))

        policy.train()
        epoch_loss = 0.0
        epoch_steps = 0
        ep_lens_sum = 0.0
        ep_count = 0

        for step in range(args.steps_per_epoch):
            # Get observations
            obs_dict = env.get_observations()
            obs = obs_dict['obs']
            priv_info = obs_dict['priv_info']

            # Teacher forward pass
            output = policy(obs, priv_info=priv_info)
            teacher_action = output['action_mean']  # (N, 1)

            # Get ground truth: body policy's knee action for current state
            with torch.no_grad():
                full_obs = env._compute_full_body_obs()
                body_action_full = body_policy(full_obs)
                target_knee = body_action_full[:, ACTIVE_PROSTHESIS_JOINTS]  # (N, 1)

            # Supervised loss
            loss = nn.functional.mse_loss(teacher_action, target_knee.detach())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            # Step environment with teacher's action + exploration noise
            with torch.no_grad():
                exec_action = teacher_action.detach()
                if noise_std > 0:
                    exec_action = exec_action + torch.randn_like(exec_action) * noise_std
                exec_action = exec_action.clamp(-1.0, 1.0)

            obs_dict, reward, done, info = env.step(exec_action)
            if 'finished_ep_len' in info:
                ep_lens_sum += info['finished_ep_len'].sum().item()
                ep_count += info['finished_ep_len'].shape[0]

        # Compute epoch stats
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_ep_len = ep_lens_sum / max(ep_count, 1)

        # TensorBoard
        if writer:
            writer.add_scalar("loss/mse", avg_loss, epoch)
            writer.add_scalar("reward/avg_ep_len", avg_ep_len, epoch)
            writer.add_scalar("reward/noise_std", noise_std, epoch)

        # Logging
        if epoch % args.log_interval == 0:
            elapsed = time.time() - train_start
            eps = (epoch - start_epoch + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  Epoch {epoch:5d}/{args.max_epochs} | "
                f"loss={avg_loss:.6f} | "
                f"ep_len={avg_ep_len:6.1f} | "
                f"noise={noise_std:.3f} | "
                f"{eps:.1f} ep/s"
            )

        # Save best (based on avg_ep_len)
        if avg_ep_len > best_eval_len and ep_count > 0:
            best_eval_len = avg_ep_len
            _save_ckpt(policy, optimizer, epoch, ckpt_dir, "best.pth")

        # Periodic save
        if epoch > 0 and epoch % args.save_interval == 0:
            _save_ckpt(policy, optimizer, epoch, ckpt_dir, f"epoch_{epoch}.pth")

    # ── Final save ────────────────────────────────────────────────────
    _save_ckpt(policy, optimizer, args.max_epochs - 1, ckpt_dir, "final.pth")
    if writer:
        writer.close()

    elapsed = time.time() - train_start
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Total time:    {elapsed/3600:.2f} hours")
    print(f"  Best ep_len:   {best_eval_len:.1f}")
    print(f"  Final ckpt:    {ckpt_dir}/final.pth")
    print(f"  Best ckpt:     {ckpt_dir}/best.pth")
    print("=" * 60)

    # Copy best to standard location
    import shutil
    std_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "stage1", "best.pth")
    os.makedirs(os.path.dirname(std_path), exist_ok=True)
    # Use best if available, otherwise final
    src_ckpt = os.path.join(ckpt_dir, "best.pth")
    if not os.path.exists(src_ckpt):
        src_ckpt = os.path.join(ckpt_dir, "final.pth")
    shutil.copy2(src_ckpt, std_path)
    print(f"  Copied best → {std_path}")


def _save_ckpt(policy, optimizer, epoch, ckpt_dir, filename):
    path = os.path.join(ckpt_dir, filename)
    torch.save({
        'epoch': epoch,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


if __name__ == "__main__":
    main()
