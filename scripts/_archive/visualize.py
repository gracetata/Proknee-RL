#!/usr/bin/env python3
"""
Visualize Stage 0 / Stage 1 / Stage 2 policies side by side.

Usage:
    conda run -n proknee_tc python scripts/visualize.py --stage 0   # Body policy only
    conda run -n proknee_tc python scripts/visualize.py --stage 1   # Teacher
    conda run -n proknee_tc python scripts/visualize.py --stage 2   # Student
"""
import os, sys, argparse, numpy as np, time

if not hasattr(np, 'float'):
    np.float = float; np.int = int; np.bool = bool
from isaacgym import gymapi, gymtorch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BODY_POLICY = os.path.join(
    ROOT, "outputs/checkpoints/stage0/stage0_amp_walk_5050.pth"
)
TEACHER_CKPT = os.path.join(ROOT, "outputs/checkpoints/stage1/best.pth")
STUDENT_CKPT = os.path.join(ROOT, "outputs/checkpoints/stage2/best.pth")

DEVICE = "cuda:0"


def visualize_stage0(num_envs=1):
    """Visualize Stage 0: frozen body policy controlling all 28 DOFs."""
    from proknee_hora.envs.proknee_base import ProKneeBase, load_body_policy

    print("[Stage 0] Creating env (full body, no prosthesis)...")
    env = ProKneeBase(
        num_envs=num_envs,
        device=DEVICE,
        headless=False,
        body_policy_checkpoint=BODY_POLICY,
        prosthesis_only=False,
        freeze_body=False,
        episode_length=1000,
    )

    body_policy = load_body_policy(BODY_POLICY, DEVICE)
    print(f"[Stage 0] Body policy loaded — {sum(p.numel() for p in body_policy.parameters()):,} params")

    obs_dict = env.reset()
    episodes = 0
    total_reward = 0.0
    steps = 0

    print("[Stage 0] Running — close viewer window or Ctrl+C to exit...")
    try:
        while True:
            with torch.no_grad():
                obs = env._compute_base_obs()
                action = body_policy(obs)
            obs_dict, reward, done, info = env.step(action)
            total_reward += reward.sum().item()
            steps += 1
            if done.any():
                episodes += 1
                print(f"  Episode {episodes} | step {steps} | avg reward: {total_reward/episodes:.2f}")
    except KeyboardInterrupt:
        pass

    print(f"\n[Stage 0] Done — {episodes} episodes, avg reward: {total_reward/max(episodes,1):.2f}")
    env.close()


def visualize_stage1(num_envs=1, checkpoint=None):
    """Visualize Stage 1: teacher policy (privileged info, prosthesis only)."""
    from proknee_hora.envs import ProKneeTeacher
    from proknee_hora.algo.models.actor_critic import ProKneePolicy
    from proknee_hora.envs.constants import (
        ACTIVE_PROSTHESIS_JOINTS, STUDENT_PROPRIO_DIM, TEACHER_PRIV_INFO_DIM,
    )

    ckpt_path = checkpoint or TEACHER_CKPT
    ACTION_DIM = len(ACTIVE_PROSTHESIS_JOINTS)

    print(f"[Stage 1] Loading teacher from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Create env first to get obs_dim
    print(f"[Stage 1] Creating teacher env with {num_envs} envs...")
    env = ProKneeTeacher(
        num_envs=num_envs, device=DEVICE, headless=False,
        body_policy_checkpoint=BODY_POLICY,
        episode_length=1000,
    )

    policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=ACTION_DIM,
        priv_info_dim=TEACHER_PRIV_INFO_DIM,
        proprio_dim=STUDENT_PROPRIO_DIM,
        history_len=30,
        latent_dim=8,
        hidden_dims=[256, 128, 64],
        mode='teacher',
    ).to(DEVICE)

    if 'policy_state_dict' in ckpt:
        policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()
    print(f"[Stage 1] Teacher loaded — {sum(p.numel() for p in policy.parameters() if p.requires_grad):,} params")

    obs_dict = env.reset()
    episodes = 0
    total_reward = 0.0
    steps = 0

    print("[Stage 1] Running (prosthesis in RED) — close viewer or Ctrl+C to exit...")
    try:
        while True:
            with torch.no_grad():
                output = policy(obs_dict['obs'], priv_info=obs_dict.get('priv_info'))
                action = output['action_mean']
            obs_dict, reward, done, info = env.step(action)
            total_reward += reward.sum().item()
            steps += 1
            if done.any():
                episodes += 1
                print(f"  Episode {episodes} | step {steps} | avg reward: {total_reward/episodes:.2f}")
    except KeyboardInterrupt:
        pass

    print(f"\n[Stage 1] Done — {episodes} episodes, avg reward: {total_reward/max(episodes,1):.2f}")
    env.close()


def visualize_stage2(num_envs=1, checkpoint=None):
    """Visualize Stage 2: student policy (no privileged info, prosthesis only)."""
    from proknee_hora.envs import ProKneeStudent
    from proknee_hora.algo.models.actor_critic import ProKneePolicy
    from proknee_hora.envs.constants import (
        ACTIVE_PROSTHESIS_JOINTS, STUDENT_PROPRIO_DIM, TEACHER_PRIV_INFO_DIM,
    )

    ckpt_path = checkpoint or STUDENT_CKPT
    ACTION_DIM = len(ACTIVE_PROSTHESIS_JOINTS)

    print(f"[Stage 2] Loading student from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    print(f"[Stage 2] Creating student env with {num_envs} envs...")
    env = ProKneeStudent(
        num_envs=num_envs, device=DEVICE, headless=False,
        body_policy_checkpoint=BODY_POLICY,
        episode_length=1000,
    )

    policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=ACTION_DIM,
        priv_info_dim=TEACHER_PRIV_INFO_DIM,
        proprio_dim=STUDENT_PROPRIO_DIM,
        history_len=30,
        latent_dim=8,
        hidden_dims=[256, 128, 64],
        mode='student',
    ).to(DEVICE)

    if 'student_state_dict' in ckpt:
        policy.load_state_dict(ckpt['student_state_dict'])
    elif 'policy_state_dict' in ckpt:
        policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()
    print(f"[Stage 2] Student loaded — {sum(p.numel() for p in policy.parameters()):,} params")

    obs_dict = env.reset()
    episodes = 0
    total_reward = 0.0
    steps = 0

    print("[Stage 2] Running (prosthesis in RED) — close viewer or Ctrl+C to exit...")
    try:
        while True:
            with torch.no_grad():
                output = policy(obs_dict['obs'], proprio_hist=obs_dict.get('proprio_hist'))
                action = output['action_mean']
            obs_dict, reward, done, info = env.step(action)
            total_reward += reward.sum().item()
            steps += 1
            if done.any():
                episodes += 1
                print(f"  Episode {episodes} | step {steps} | avg reward: {total_reward/episodes:.2f}")
    except KeyboardInterrupt:
        pass

    print(f"\n[Stage 2] Done — {episodes} episodes, avg reward: {total_reward/max(episodes,1):.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize ProKnee policies")
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2],
                        help='Stage to visualize: 0=body, 1=teacher, 2=student')
    parser.add_argument('--checkpoint', type=str, default=None, help='Override checkpoint')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of envs')
    args = parser.parse_args()

    if args.stage == 0:
        visualize_stage0(args.num_envs)
    elif args.stage == 1:
        visualize_stage1(args.num_envs, args.checkpoint)
    elif args.stage == 2:
        visualize_stage2(args.num_envs, args.checkpoint)


if __name__ == "__main__":
    main()
