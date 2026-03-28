#!/usr/bin/env python3
"""
Evaluate Multi-Motion Stage 2 student policy.

Tests the multi-motion student across all motion types and generates
per-motion statistics and gait analysis.

Usage:
    # Headless evaluation (all motions)
    python scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 256

    # Single motion evaluation
    python scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 256 --motion walk

    # Visual evaluation
    python scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 1 --visualize
"""

import isaacgym  # noqa: F401

import os
import sys
import argparse
import time
import numpy as np

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_multi_motion import ProKneeMultiMotionEnv
from proknee_hora.envs.constants_multi import (
    OBS_DIM_MULTI, OBS_DIM_REALISTIC,
    TEACHER_PRIV_INFO_DIM, TEACHER_PRIV_INFO_DIM_MULTI,
    STUDENT_PROPRIO_DIM_MULTI, STUDENT_PROPRIO_DIM_REALISTIC,
    LATENT_DIM, PROPRIO_HISTORY_LEN, MOTION_NAMES,
    MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND,
    NUM_MOTION_TYPES,
)
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.algo.models.running_mean_std import RunningMeanStd

MOTION_NAME_TO_ID = {v: k for k, v in MOTION_NAMES.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Multi-Motion Stage 2")
    p.add_argument("--student-ckpt", type=str,
                    default="outputs/checkpoints/stage2_multi/best.pth")
    p.add_argument("--motions", nargs='+', default=['walk', 'run', 'dance', 'stand'])
    p.add_argument("--motion", type=str, default=None,
                    help="Evaluate single motion (overrides --motions)")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-length", type=int, default=300)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--no-motion-in-obs", action="store_true",
                    help="Realistic mode (obs=16D)")
    p.add_argument("--motion-in-obs", action="store_true",
                    help="Legacy mode (obs=20D, auto-detected from checkpoint)")
    p.add_argument("--output-dir", type=str,
                    default=os.path.join(ROOT, "outputs", "eval_stage2_multi"))
    return p.parse_args()


def evaluate_motion(env, model, running_mean_std, sa_mean_std, motion_id,
                    num_episodes, device):
    """Evaluate student policy on a single motion type."""
    env.set_all_motions(motion_id)

    ep_lengths = []
    total_rewards = []

    obs_dict = env.reset()
    ep_reward = torch.zeros(env.num_envs, device=device)
    completed = 0

    while completed < num_episodes * env.num_envs:
        with torch.no_grad():
            obs_norm = running_mean_std(obs_dict['obs']).detach()
            hist_norm = sa_mean_std(obs_dict['proprio_hist']).detach()

            student_latent = model.adaptation.adapt_tconv(hist_norm)
            action_mean, _, _ = model.actor_critic(obs_norm, student_latent)
            action = action_mean.clamp(-1.0, 1.0)

        obs_dict, reward, done, info = env.step(action)
        ep_reward += reward

        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            # Use info['finished_ep_len'] which is captured BEFORE reset
            finished_lens = info.get('finished_ep_len', None)
            for i, idx in enumerate(done_ids):
                if finished_lens is not None:
                    ep_lengths.append(finished_lens[i].item())
                else:
                    ep_lengths.append(info['episode_length'][idx.item()].item())
                total_rewards.append(ep_reward[idx].item())
            ep_reward[done_ids] = 0
            completed += done_ids.shape[0]

            if completed >= num_episodes * env.num_envs:
                break

    return {
        'motion': MOTION_NAMES[motion_id],
        'avg_ep_len': np.mean(ep_lengths) if ep_lengths else 0,
        'std_ep_len': np.std(ep_lengths) if ep_lengths else 0,
        'avg_reward': np.mean(total_rewards) if total_rewards else 0,
        'survival_rate': np.mean([l >= 290 for l in ep_lengths]) if ep_lengths else 0,
        'num_episodes': len(ep_lengths),
    }


def main():
    args = parse_args()

    # Resolve motions
    if args.motion:
        motion_list = [args.motion]
    else:
        motion_list = args.motions

    enabled_motions = []
    for name in motion_list:
        if name.lower() in MOTION_NAME_TO_ID:
            enabled_motions.append(MOTION_NAME_TO_ID[name.lower()])

    print("=" * 60)
    print("  ProKnee Stage 2 Multi-Motion Evaluation")
    print("=" * 60)

    # Determine observation mode
    if args.motion_in_obs:
        motion_in_obs = True
    elif args.no_motion_in_obs:
        motion_in_obs = False
    else:
        ckpt_meta = torch.load(args.student_ckpt, map_location='cpu')
        motion_in_obs = ckpt_meta.get('motion_in_obs', True)
        del ckpt_meta

    if motion_in_obs:
        obs_dim = OBS_DIM_MULTI
        priv_dim = TEACHER_PRIV_INFO_DIM
        proprio_dim = STUDENT_PROPRIO_DIM_MULTI
    else:
        obs_dim = OBS_DIM_REALISTIC
        priv_dim = TEACHER_PRIV_INFO_DIM_MULTI
        proprio_dim = STUDENT_PROPRIO_DIM_REALISTIC

    mode_str = "legacy (obs=20D)" if motion_in_obs else "realistic (obs=16D)"
    print(f"  Device:     {args.device}")
    print(f"  Num envs:   {args.num_envs}")
    print(f"  Motions:    {[MOTION_NAMES[m] for m in enabled_motions]}")
    print(f"  Checkpoint: {args.student_ckpt}")
    print(f"  Obs mode:   {mode_str}")
    print("=" * 60)

    # Create environment
    print("\nCreating environment...")
    env = ProKneeMultiMotionEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=not args.visualize,
        enabled_motions=enabled_motions,
        episode_length=args.episode_length,
        motion_in_obs=motion_in_obs,
    )

    # Create model
    model = ProKneePolicy(
        obs_dim=obs_dim,
        action_dim=env.num_actions,
        priv_info_dim=priv_dim,
        proprio_dim=proprio_dim,
        history_len=PROPRIO_HISTORY_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64],
        mode='student',
    ).to(args.device)

    running_mean_std = RunningMeanStd(obs_dim).to(args.device)
    sa_mean_std = RunningMeanStd((PROPRIO_HISTORY_LEN, proprio_dim)).to(args.device)

    # Load checkpoint
    ckpt = torch.load(args.student_ckpt, map_location=args.device, weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt['policy_state_dict'])
    if 'running_mean_std' in ckpt:
        running_mean_std.load_state_dict(ckpt['running_mean_std'])
    if 'sa_mean_std' in ckpt:
        sa_mean_std.load_state_dict(ckpt['sa_mean_std'])

    model.eval()
    running_mean_std.eval()
    sa_mean_std.eval()
    print("Model loaded.")

    # Evaluate each motion
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for mid in enabled_motions:
        print(f"\nEvaluating: {MOTION_NAMES[mid]}...")
        result = evaluate_motion(
            env, model, running_mean_std, sa_mean_std,
            mid, args.num_episodes, args.device
        )
        results.append(result)
        print(f"  avg_ep_len: {result['avg_ep_len']:.1f} ± {result['std_ep_len']:.1f}")
        print(f"  avg_reward: {result['avg_reward']:.2f}")
        print(f"  survival:   {result['survival_rate']:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"  {'Motion':<10} {'Avg EP Len':>12} {'Reward':>10} {'Survival':>10}")
    print("-" * 60)
    for r in results:
        print(f"  {r['motion']:<10} {r['avg_ep_len']:>10.1f}   {r['avg_reward']:>10.2f} {r['survival_rate']:>9.1%}")
    print("=" * 60)

    # Save results
    import json
    results_path = os.path.join(args.output_dir, "multi_motion_eval.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    env.close()


if __name__ == "__main__":
    main()
