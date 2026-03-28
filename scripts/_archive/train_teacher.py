#!/usr/bin/env python3
"""
Standalone Stage-1 Teacher training script.

Avoids Hydra overhead and ensures Isaac Gym is imported before torch.
Usage:
    conda run -n proknee_tc python scripts/train_teacher.py
    conda run -n proknee_tc python scripts/train_teacher.py --resume outputs/stage1_teacher/checkpoints/epoch_800.pth
"""
import os, sys, argparse, numpy as np

# ── Fix numpy deprecations & import Isaac Gym before torch ──────────
if not hasattr(np, 'float'):
    np.float = float; np.int = int; np.bool = bool  # type: ignore
from isaacgym import gymapi                         # noqa: E402

import torch                                         # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_teacher import ProKneeTeacher
from proknee_hora.algo.ppo import PPO
from proknee_hora.algo.models.actor_critic import ProKneePolicy

# ── Hyperparameters ──────────────────────────────────────────────────
BODY_POLICY = os.path.join(
    ROOT,
    "ProKnee-Simulator/runs/HumanoidAMP_stable_06-18-14-00/"
    "nn/HumanoidAMP_stable_06-18-14-01_5000.pth",
)

NUM_ENVS     = 4096
MAX_EPOCHS   = 3000
SAVE_EVERY   = 100
DEVICE       = "cuda:0"
LR           = 5e-4
HORIZON      = 16
MINIBATCH    = NUM_ENVS * HORIZON // 4   # ~16 384
MINI_EPOCHS  = 5
GAMMA        = 0.99
TAU          = 0.95
E_CLIP       = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF   = 1.0
MAX_GRAD     = 1.0
EPISODE_LEN  = 500

PRIV_INFO_DIM  = 18
PROPRIO_DIM    = 16
HIST_LEN       = 30
LATENT_DIM     = 8
HIDDEN_DIMS    = [256, 128, 64]

OUTPUT_DIR = os.path.join(ROOT, "outputs", "stage1_teacher")
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    torch.manual_seed(42)

    # ── Environment ──────────────────────────────────────────────────
    print(f"[train] Creating teacher env with {NUM_ENVS} envs …", flush=True)
    env = ProKneeTeacher(
        num_envs=NUM_ENVS,
        device=DEVICE,
        headless=True,
        body_policy_checkpoint=BODY_POLICY,
        proprio_hist_len=HIST_LEN,
        episode_length=EPISODE_LEN,
    )
    print(f"[train] Env OK  — obs={env.num_obs}, act={env.num_actions}", flush=True)

    # ── Policy ───────────────────────────────────────────────────────
    policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=env.num_actions,
        priv_info_dim=PRIV_INFO_DIM,
        proprio_dim=PROPRIO_DIM,
        history_len=HIST_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        mode="teacher",
    ).to(DEVICE)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[train] Policy  — {n_params:,} parameters", flush=True)

    # ── PPO trainer ──────────────────────────────────────────────────
    trainer = PPO(
        policy=policy,
        env=env,
        learning_rate=LR,
        gamma=GAMMA,
        tau=TAU,
        e_clip=E_CLIP,
        entropy_coef=ENTROPY_COEF,
        value_loss_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD,
        horizon_length=HORIZON,
        minibatch_size=MINIBATCH,
        mini_epochs=MINI_EPOCHS,
        device=DEVICE,
        use_priv_info=True,
        priv_info_dim=PRIV_INFO_DIM,
    )

    # Resume from checkpoint
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        trainer.load(args.resume)
        start_epoch = trainer.epoch + 1
        print(f"[train] Resumed from {args.resume} (epoch {trainer.epoch})", flush=True)

    # ── Training loop ────────────────────────────────────────────────
    best_reward = float("-inf")
    print(f"[train] Starting Stage-1 Teacher training epochs {start_epoch}..{MAX_EPOCHS} …\n", flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # Collect rollout
        rollout_stats = trainer.collect_rollout()

        # PPO update
        update_stats = trainer.update()

        # Merge stats
        stats = {**rollout_stats, **update_stats}
        reward_mean = stats["reward_mean"]
        trainer.epoch = epoch

        # Log
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:5d} | reward {reward_mean:+8.3f} | "
                f"policy_loss {stats.get('policy_loss', 0):.4f} | "
                f"value_loss {stats.get('value_loss', 0):.4f} | "
                f"episodes {stats.get('episode_count', 0):.0f}",
                flush=True,
            )

        # Save best
        if reward_mean > best_reward:
            best_reward = reward_mean
            trainer.save(os.path.join(OUTPUT_DIR, "checkpoints", "best.pth"))

        # Periodic save
        if epoch % SAVE_EVERY == 0:
            trainer.save(os.path.join(OUTPUT_DIR, "checkpoints", f"epoch_{epoch}.pth"))

    # Final save
    trainer.save(os.path.join(OUTPUT_DIR, "checkpoints", "final.pth"))
    print(f"\n[train] Done! Best reward: {best_reward:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
