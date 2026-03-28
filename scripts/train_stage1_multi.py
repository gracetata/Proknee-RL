#!/usr/bin/env python3
"""
Stage 1 Multi-Motion: Teacher Policy Training via DAgger.

Extends the single-motion DAgger training to support multiple motion types
(walk, run, dance, stand). Uses ProKneeMultiMotionEnv with per-motion
body policies providing ground truth.

Does NOT modify the original train_stage1_dagger.py.

Usage:
    # CPU test
    python scripts/train_stage1_multi.py --device cpu --num-envs 16 --max-epochs 10

    # GPU training (all 4 motions)
    nohup python scripts/train_stage1_multi.py --device cuda:0 --num-envs 4096 --max-epochs 8000 \
        > outputs/stage1_multi_train.log 2>&1 &

    # GPU training (walk + run only)
    nohup python scripts/train_stage1_multi.py --device cuda:0 --num-envs 4096 --max-epochs 6000 \
        --motions walk run > outputs/stage1_multi_wr.log 2>&1 &
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

from proknee_hora.envs.proknee_multi_motion import ProKneeMultiMotionEnv
from proknee_hora.envs.proknee_base import load_body_policy
from proknee_hora.envs.constants_multi import (
    OBS_DIM_MULTI, OBS_DIM_REALISTIC,
    TEACHER_PRIV_INFO_DIM, TEACHER_PRIV_INFO_DIM_MULTI,
    STUDENT_PROPRIO_DIM_MULTI, STUDENT_PROPRIO_DIM_REALISTIC,
    LATENT_DIM, PROPRIO_HISTORY_LEN, PROSTHESIS_ACTION_DIM,
    MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND,
    MOTION_NAMES, ACTIVE_PROSTHESIS_JOINTS,
)
from proknee_hora.algo.models import ProKneePolicy


MOTION_NAME_TO_ID = {v: k for k, v in MOTION_NAMES.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Multi-Motion DAgger Training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=8000)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--motions", nargs='+', default=['walk', 'run', 'dance', 'stand'],
                        help="Motion types to train on")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--switch-motion", action="store_true",
                        help="Enable mid-episode motion switching for transition training")
    parser.add_argument("--blend-steps", type=int, default=15,
                        help="Transition blending steps (0 to disable)")
    parser.add_argument("--no-motion-in-obs", action="store_true",
                        help="Realistic mode: motion one-hot in priv_info, not obs (obs=16D, priv=117D)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve motion IDs
    enabled_motions = []
    for name in args.motions:
        if name.lower() in MOTION_NAME_TO_ID:
            enabled_motions.append(MOTION_NAME_TO_ID[name.lower()])
        else:
            print(f"WARNING: Unknown motion type '{name}', skipping")
    if not enabled_motions:
        print("ERROR: No valid motion types specified")
        sys.exit(1)

    motion_names = [MOTION_NAMES[m] for m in enabled_motions]

    # Determine observation mode
    motion_in_obs = not args.no_motion_in_obs
    if motion_in_obs:
        obs_dim = OBS_DIM_MULTI              # 20
        priv_dim = TEACHER_PRIV_INFO_DIM     # 113
        proprio_dim = STUDENT_PROPRIO_DIM_MULTI  # 20
    else:
        obs_dim = OBS_DIM_REALISTIC          # 16
        priv_dim = TEACHER_PRIV_INFO_DIM_MULTI  # 117
        proprio_dim = STUDENT_PROPRIO_DIM_REALISTIC  # 16

    # ── Banner ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  ProKnee Stage 1 Multi-Motion: DAgger Training")
    print("=" * 60)
    print(f"  Device:       {args.device}")
    print(f"  Num envs:     {args.num_envs}")
    print(f"  Max epochs:   {args.max_epochs}")
    print(f"  Steps/epoch:  {args.steps_per_epoch}")
    print(f"  Motions:      {motion_names}")
    obs_mode = "realistic (obs=16D, priv=117D)" if not motion_in_obs else "legacy (obs=20D, priv=113D)"
    print(f"  Obs mode:     {obs_mode}")
    print(f"  obs_dim:      {obs_dim}")
    print(f"  priv_dim:     {priv_dim}")
    print(f"  LR:           {args.lr}")
    print(f"  Noise std:    {args.noise_std}")
    print("=" * 60)

    # ── Output directory ──────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        motion_tag = '_'.join(args.motions)
        args.output_dir = os.path.join(
            PROJECT_ROOT, "outputs", f"stage1_multi_{motion_tag}_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"  Output dir:   {args.output_dir}")

    # ── Create multi-motion environment ───────────────────────────────
    print("\n[Stage 1 Multi] Creating multi-motion environment...")
    t0 = time.time()
    env = ProKneeMultiMotionEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        enabled_motions=enabled_motions,
        episode_length=300,
        switch_motion_in_episode=args.switch_motion,
        transition_blend_steps=args.blend_steps,
        motion_in_obs=motion_in_obs,
    )
    print(f"[Stage 1 Multi] Environment created in {time.time()-t0:.1f}s")
    print(f"  Motion distribution: {env.get_motion_distribution()}")

    # ── Create teacher policy (with multi-motion obs_dim) ─────────────
    print("[Stage 1 Multi] Creating ProKneePolicy (teacher mode)...")
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
        print(f"[Stage 1 Multi] Resumed from epoch {start_epoch}")

    # ── Reset environment ─────────────────────────────────────────────
    print("[Stage 1 Multi] Resetting environment...")
    env.reset()

    # ── TensorBoard ───────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[Stage 1 Multi] TensorBoard: tensorboard --logdir {tb_dir}")
    except ImportError:
        writer = None

    # ── Training loop ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training started!")
    print("=" * 60)
    best_eval_len = 0
    train_start = time.time()

    # Per-motion episode length tracking
    per_motion_ep_lens = {mid: [] for mid in enabled_motions}

    for epoch in range(start_epoch, args.max_epochs):
        noise_std = max(0.0, args.noise_std * (1 - epoch / 3000))

        policy.train()
        epoch_loss = 0.0
        epoch_steps = 0
        ep_lens_sum = 0.0
        ep_count = 0

        for step in range(args.steps_per_epoch):
            obs_dict = env.get_observations()
            obs = obs_dict['obs']       # 20D
            priv_info = obs_dict['priv_info']  # 113D

            # Teacher forward pass
            output = policy(obs, priv_info=priv_info)
            teacher_action = output['action_mean']

            # Get ground truth from per-motion body policies
            with torch.no_grad():
                full_obs = env._compute_full_body_obs()
                # The env already has per-motion body policies; get their actions
                body_action_full = env._get_body_action(full_obs)
                target = body_action_full[:, ACTIVE_PROSTHESIS_JOINTS]

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

        # Compute epoch stats
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_ep_len = ep_lens_sum / max(ep_count, 1)

        # TensorBoard
        if writer:
            writer.add_scalar("loss/mse", avg_loss, epoch)
            writer.add_scalar("reward/avg_ep_len", avg_ep_len, epoch)
            writer.add_scalar("reward/noise_std", noise_std, epoch)
            # Log motion distribution
            dist = env.get_motion_distribution()
            for name, count in dist.items():
                writer.add_scalar(f"motion_dist/{name}", count, epoch)

        # Logging
        if epoch % args.log_interval == 0:
            elapsed = time.time() - train_start
            eps = (epoch - start_epoch + 1) / elapsed if elapsed > 0 else 0
            dist_str = " | ".join(f"{k}:{v}" for k, v in env.get_motion_distribution().items())
            print(
                f"  Epoch {epoch:5d}/{args.max_epochs} | "
                f"loss={avg_loss:.6f} | "
                f"ep_len={avg_ep_len:6.1f} | "
                f"noise={noise_std:.3f} | "
                f"{eps:.1f} ep/s | "
                f"dist: [{dist_str}]"
            )

        # Save best
        if avg_ep_len > best_eval_len and ep_count > 0:
            best_eval_len = avg_ep_len
            _save_ckpt(policy, optimizer, epoch, ckpt_dir, "best.pth", motion_in_obs)

        # Periodic save
        if epoch > 0 and epoch % args.save_interval == 0:
            _save_ckpt(policy, optimizer, epoch, ckpt_dir, f"epoch_{epoch}.pth", motion_in_obs)

    # ── Final save ────────────────────────────────────────────────────
    _save_ckpt(policy, optimizer, args.max_epochs - 1, ckpt_dir, "final.pth", motion_in_obs)
    if writer:
        writer.close()

    elapsed = time.time() - train_start
    print("\n" + "=" * 60)
    print("  Multi-Motion Stage 1 Training complete!")
    print(f"  Total time:    {elapsed/3600:.2f} hours")
    print(f"  Best ep_len:   {best_eval_len:.1f}")
    print(f"  Motions:       {motion_names}")
    print(f"  Final ckpt:    {ckpt_dir}/final.pth")
    print(f"  Best ckpt:     {ckpt_dir}/best.pth")
    print("=" * 60)

    # Copy best to standard location
    import shutil
    tag = "stage1_multi_realistic" if not motion_in_obs else "stage1_multi"
    std_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", tag)
    os.makedirs(std_dir, exist_ok=True)
    src_ckpt = os.path.join(ckpt_dir, "best.pth")
    if not os.path.exists(src_ckpt):
        src_ckpt = os.path.join(ckpt_dir, "final.pth")
    shutil.copy2(src_ckpt, os.path.join(std_dir, "best.pth"))
    print(f"  Copied best → {std_dir}/best.pth")


def _save_ckpt(policy, optimizer, epoch, ckpt_dir, filename, motion_in_obs=True):
    path = os.path.join(ckpt_dir, filename)
    torch.save({
        'epoch': epoch,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'motion_in_obs': motion_in_obs,
    }, path)


if __name__ == "__main__":
    main()
