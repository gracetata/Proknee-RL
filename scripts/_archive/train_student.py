#!/usr/bin/env python3
"""
Standalone Stage-2 Student training script.

Trains a student policy via PPO + distillation from a frozen teacher.
Usage:
    conda run -n proknee_tc python scripts/train_student.py
    conda run -n proknee_tc python scripts/train_student.py --resume outputs/stage2_student/checkpoints/epoch_500.pth
"""
import os, sys, argparse, numpy as np

# Fix numpy deprecations & import Isaac Gym before torch
if not hasattr(np, 'float'):
    np.float = float; np.int = int; np.bool = bool  # type: ignore
from isaacgym import gymapi                         # noqa: E402

import torch                                         # noqa: E402

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_student import ProKneeStudent
from proknee_hora.algo.teacher_student import TeacherStudentTrainer, load_teacher_from_checkpoint
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.envs.constants import (
    ACTIVE_PROSTHESIS_JOINTS, STUDENT_PROPRIO_DIM, TEACHER_PRIV_INFO_DIM,
)

# ── Hyperparameters ──────────────────────────────────────────────────
BODY_POLICY = os.path.join(
    ROOT,
    "ProKnee-Simulator/runs/HumanoidAMP_stable_06-18-14-00/"
    "nn/HumanoidAMP_stable_06-18-14-01_5000.pth",
)
TEACHER_CKPT = os.path.join(ROOT, "outputs/stage1_teacher/checkpoints/epoch_3000.pth")

NUM_ENVS     = 4096
MAX_EPOCHS   = 3000
SAVE_EVERY   = 100
DEVICE       = "cuda:0"
LR           = 3e-4
HORIZON      = 16
MINIBATCH    = NUM_ENVS * HORIZON // 4
MINI_EPOCHS  = 5
GAMMA        = 0.99
TAU          = 0.95
E_CLIP       = 0.2
ENTROPY_COEF = 0.005
VALUE_COEF   = 1.0
MAX_GRAD     = 1.0
DISTILL_COEF = 1.0
EPISODE_LEN  = 500

PRIV_INFO_DIM = TEACHER_PRIV_INFO_DIM
PROPRIO_DIM   = STUDENT_PROPRIO_DIM
HIST_LEN      = 30
LATENT_DIM    = 8
HIDDEN_DIMS   = [256, 128, 64]
ACTION_DIM    = len(ACTIVE_PROSTHESIS_JOINTS)

OUTPUT_DIR = os.path.join(ROOT, "outputs", "stage2_student")
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--teacher', type=str, default=TEACHER_CKPT, help='Teacher checkpoint')
    args = parser.parse_args()

    torch.manual_seed(42)

    # ── Environment ──────────────────────────────────────────────────
    print(f"[train] Creating student env with {NUM_ENVS} envs …", flush=True)
    env = ProKneeStudent(
        num_envs=NUM_ENVS,
        device=DEVICE,
        headless=True,
        body_policy_checkpoint=BODY_POLICY,
        teacher_checkpoint=None,  # We load teacher separately
        proprio_hist_len=HIST_LEN,
        episode_length=EPISODE_LEN,
        distill_coef=DISTILL_COEF,
    )
    print(f"[train] Env OK — obs={env.num_obs}, act={env.num_actions}", flush=True)

    # ── Load frozen teacher ──────────────────────────────────────────
    print(f"[train] Loading teacher from {args.teacher}", flush=True)
    teacher_policy = load_teacher_from_checkpoint(args.teacher, DEVICE)
    teacher_policy.eval()
    for p in teacher_policy.parameters():
        p.requires_grad = False
    print(f"[train] Teacher loaded — {sum(p.numel() for p in teacher_policy.parameters()):,} params (frozen)", flush=True)

    # ── Student policy ───────────────────────────────────────────────
    student_policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=ACTION_DIM,
        priv_info_dim=PRIV_INFO_DIM,
        proprio_dim=PROPRIO_DIM,
        history_len=HIST_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        mode="student",
    ).to(DEVICE)
    n_params = sum(p.numel() for p in student_policy.parameters() if p.requires_grad)
    print(f"[train] Student — {n_params:,} trainable parameters", flush=True)

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = TeacherStudentTrainer(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        env=env,
        learning_rate=LR,
        gamma=GAMMA,
        tau=TAU,
        e_clip=E_CLIP,
        entropy_coef=ENTROPY_COEF,
        value_loss_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD,
        distill_coef=DISTILL_COEF,
        horizon_length=HORIZON,
        minibatch_size=MINIBATCH,
        mini_epochs=MINI_EPOCHS,
        device=DEVICE,
        proprio_dim=PROPRIO_DIM,
        history_len=HIST_LEN,
    )

    # Resume from checkpoint
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        trainer.load(args.resume)
        start_epoch = trainer.epoch + 1
        print(f"[train] Resumed from {args.resume} (epoch {trainer.epoch})", flush=True)

    # ── Training loop ────────────────────────────────────────────────
    best_reward = float("-inf")
    print(f"[train] Starting Stage-2 Student training epochs {start_epoch}..{MAX_EPOCHS} …\n", flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        rollout_stats = trainer.collect_rollout()
        update_stats = trainer.update()
        stats = {**rollout_stats, **update_stats}
        reward_mean = stats["reward_mean"]
        trainer.epoch = epoch

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:5d} | reward {reward_mean:+8.3f} | "
                f"policy {stats.get('policy_loss', 0):.4f} | "
                f"value {stats.get('value_loss', 0):.4f} | "
                f"distill {stats.get('distill_loss', 0):.4f} | "
                f"eps {stats.get('episode_count', 0):.0f}",
                flush=True,
            )

        if reward_mean > best_reward:
            best_reward = reward_mean
            trainer.save(os.path.join(OUTPUT_DIR, "checkpoints", "best.pth"))

        if epoch % SAVE_EVERY == 0:
            trainer.save(os.path.join(OUTPUT_DIR, "checkpoints", f"epoch_{epoch}.pth"))

    trainer.save(os.path.join(OUTPUT_DIR, "checkpoints", "final.pth"))
    print(f"\n[train] Done! Best reward: {best_reward:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
