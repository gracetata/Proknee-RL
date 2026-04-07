#!/usr/bin/env python3
"""
Stage 1 Unified: Teacher Policy Training via DAgger with Velocity-Conditioned Body Policy.

Uses ProKneeUnifiedEnv with the unified velocity-conditioned body policy (Phase 11+).
Instead of per-motion body policies, a single unified policy handles all velocities.

Does NOT modify the original train_stage1_dagger.py or train_stage1_multi.py.

Usage:
    # CPU test
    python scripts/train_stage1_unified.py --device cpu --num-envs 16 --max-epochs 10

    # GPU training
    PYTHONUNBUFFERED=1 nohup python scripts/train_stage1_unified.py \
        --device cuda:0 --num-envs 4096 --max-epochs 8000 \
        > outputs/stage1_unified_train.log 2>&1 &
"""

# ── Isaac Gym MUST be imported before torch ──────────────────────────
import isaacgym  # noqa: F401

import os
import sys
import argparse
import time
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Fix numpy deprecations
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from proknee_hora.envs.proknee_unified import ProKneeUnifiedEnv
from proknee_hora.envs.constants import (
    LATENT_DIM, PROPRIO_HISTORY_LEN, PROSTHESIS_ACTION_DIM,
    ACTIVE_PROSTHESIS_JOINTS,
)
from proknee_hora.envs.constants_unified import (
    OBS_DIM_UNIFIED,
    TEACHER_PRIV_INFO_DIM_UNIFIED,
    STUDENT_PROPRIO_DIM_UNIFIED,
    VELOCITY_LEVELS,
)
from proknee_hora.algo.models import ProKneePolicy


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  ProKnee Stage 1 Unified: Velocity-Conditioned DAgger        ║
║  Single body policy → unified velocity control (0~2.5 m/s)   ║
║  obs=16D, priv=114D (113D base + 1D velocity_cmd)            ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Unified DAgger Training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=8000)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--body-policy", type=str,
                        default="outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth",
                        help="HumanMimic Unified Stage0 checkpoint")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--vel-switch-prob", type=float, default=0.005,
                        help="Per-step probability of velocity command switch")
    parser.add_argument("--vel-switch-interval", type=int, default=100,
                        help="Minimum steps before velocity can switch")
    return parser.parse_args()


def main():
    args = parse_args()

    print(BANNER)
    print(f"  Device:       {args.device}")
    print(f"  Num envs:     {args.num_envs}")
    print(f"  Max epochs:   {args.max_epochs}")
    print(f"  Steps/epoch:  {args.steps_per_epoch}")
    print(f"  Body policy:  {args.body_policy}")
    print(f"  LR:           {args.lr}")
    print(f"  Noise std:    {args.noise_std}")
    print(f"  Vel switch:   prob={args.vel_switch_prob}, interval={args.vel_switch_interval}")
    print("=" * 60)

    # Check body policy exists
    if not os.path.exists(args.body_policy):
        print(f"ERROR: Body policy not found: {args.body_policy}")
        print("Run Stage 0 Unified training first, then save checkpoint to this path.")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(PROJECT_ROOT, "outputs", f"stage1_unified_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"  Output dir:   {args.output_dir}")

    # ── Create unified environment ────────────────────────────────────
    print("\n[Stage 1 Unified] Creating ProKneeUnifiedEnv...")
    t0 = time.time()
    env = ProKneeUnifiedEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        body_policy_checkpoint=args.body_policy,
        episode_length=300,
        velocity_switch_prob=args.vel_switch_prob,
        velocity_switch_interval=args.vel_switch_interval,
    )
    print(f"[Stage 1 Unified] Environment created in {time.time()-t0:.1f}s")

    # ── Create teacher policy ─────────────────────────────────────────
    obs_dim = OBS_DIM_UNIFIED              # 16
    priv_dim = TEACHER_PRIV_INFO_DIM_UNIFIED  # 114
    proprio_dim = STUDENT_PROPRIO_DIM_UNIFIED  # 16

    print(f"[Stage 1 Unified] Creating ProKneePolicy (teacher mode)")
    print(f"  obs_dim:      {obs_dim}")
    print(f"  priv_dim:     {priv_dim}")
    print(f"  proprio_dim:  {proprio_dim}")

    policy = ProKneePolicy(
        obs_dim=obs_dim,
        action_dim=PROSTHESIS_ACTION_DIM,
        priv_info_dim=priv_dim,
        proprio_dim=proprio_dim,
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
        print(f"[Stage 1 Unified] Resumed from epoch {start_epoch}")

    # ── Reset environment ─────────────────────────────────────────────
    print("[Stage 1 Unified] Resetting environment...")
    env.reset()

    # ── TensorBoard ───────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[Stage 1 Unified] TensorBoard: tensorboard --logdir {tb_dir}")
    except ImportError:
        writer = None

    # ── Training loop ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training started!")
    print("=" * 60)
    best_eval_len = 0
    train_start = time.time()

    # Per-velocity tracking
    vel_ep_lens = {v: [] for v in VELOCITY_LEVELS}

    for epoch in range(start_epoch, args.max_epochs):
        noise_std = max(0.0, args.noise_std * (1 - epoch / 3000))

        policy.train()
        epoch_loss = 0.0
        epoch_steps = 0
        ep_lens_sum = 0.0
        ep_count = 0

        for step in range(args.steps_per_epoch):
            obs_dict = env.get_observations()
            obs = obs_dict['obs']       # 16D
            priv_info = obs_dict['priv_info']  # 114D (includes velocity_cmd)

            # Teacher forward pass
            output = policy(obs, priv_info=priv_info)
            teacher_action = output['action_mean']

            # Get ground truth from unified body policy
            with torch.no_grad():
                full_obs = env._compute_full_body_obs()  # 105D
                body_action_full = env._get_body_action(full_obs)  # 28D
                target = body_action_full[:, ACTIVE_PROSTHESIS_JOINTS]  # 4D

            # Supervised loss
            loss = nn.functional.mse_loss(teacher_action, target.detach())

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

        # ── Epoch statistics ──────────────────────────────────────────
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_ep_len = ep_lens_sum / max(ep_count, 1) if ep_count > 0 else 0.0

        # Log to TensorBoard
        if writer:
            writer.add_scalar('loss/mse', avg_loss, epoch)
            writer.add_scalar('reward/avg_ep_len', avg_ep_len, epoch)
            writer.add_scalar('train/noise_std', noise_std, epoch)
            # Log velocity distribution
            vel_cmd = env.velocity_cmd
            for v in VELOCITY_LEVELS:
                frac = (vel_cmd == v).float().mean().item()
                writer.add_scalar(f'velocity/frac_{v:.1f}', frac, epoch)

        # Console output
        if epoch % args.log_interval == 0:
            elapsed = time.time() - train_start
            fps = (epoch - start_epoch + 1) * args.steps_per_epoch * args.num_envs / elapsed
            print(f"Epoch {epoch:5d} | Loss {avg_loss:.6f} | Avg ep_len {avg_ep_len:6.1f} | "
                  f"Noise {noise_std:.3f} | FPS {fps:.0f}")

        # Save checkpoint
        if epoch % args.save_interval == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'avg_ep_len': avg_ep_len,
            }, ckpt_path)
            print(f"  [Checkpoint] Saved to {ckpt_path}")

        # Save best model
        if avg_ep_len > best_eval_len and epoch > 100:
            best_eval_len = avg_ep_len
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'avg_ep_len': avg_ep_len,
            }, best_path)
            print(f"  [Best] New best ep_len: {best_eval_len:.1f} at epoch {epoch}")

    # ── Final save ────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final.pth")
    torch.save({
        'epoch': args.max_epochs - 1,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\n[Stage 1 Unified] Training complete! Final checkpoint: {final_path}")

    # Copy best to canonical location
    canonical_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "stage1_unified")
    os.makedirs(canonical_dir, exist_ok=True)
    best_src = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(best_src):
        shutil.copy2(best_src, os.path.join(canonical_dir, "best.pth"))
        print(f"[Stage 1 Unified] Best checkpoint → {canonical_dir}/best.pth")

    print(f"\n[Stage 1 Unified] Best ep_len achieved: {best_eval_len:.1f}")


if __name__ == "__main__":
    main()
