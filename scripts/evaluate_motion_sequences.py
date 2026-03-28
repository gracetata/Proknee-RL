#!/usr/bin/env python3
"""
Evaluate multi-motion Stage 2 student with predefined motion sequences.

Tests:
  1. Per-motion evaluation (walk, run, dance, stand independently)
  2. Walk-Stop-Walk sequence
  3. Walk-Run-Walk-Stop-Run sequence
  4. Intra-episode random switching

Reports per-sequence ep_len, reward, and survival rate.

Usage:
    python scripts/evaluate_motion_sequences.py --device cuda:0
    python scripts/evaluate_motion_sequences.py --device cuda:0 --num-envs 256 --episodes 20
"""

import isaacgym  # noqa: F401

import os
import sys
import json
import argparse
import time
from datetime import datetime

import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_multi_motion import ProKneeMultiMotionEnv
from proknee_hora.envs.constants_multi import (
    OBS_DIM_MULTI, OBS_DIM_REALISTIC,
    TEACHER_PRIV_INFO_DIM, TEACHER_PRIV_INFO_DIM_MULTI,
    STUDENT_PROPRIO_DIM_MULTI, STUDENT_PROPRIO_DIM_REALISTIC,
    LATENT_DIM, PROPRIO_HISTORY_LEN,
    MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND,
    MOTION_NAMES,
)
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.algo.models.running_mean_std import RunningMeanStd


# ── Predefined motion sequences ──────────────────────────────────────
SEQUENCES = {
    'walk-stop-walk': [
        (MOTION_WALK, 100),     # walk for 100 steps
        (MOTION_STAND, 50),     # stop for 50 steps
        (MOTION_WALK, 100),     # walk for 100 steps
    ],
    'walk-run-walk-stop-run': [
        (MOTION_WALK, 60),      # walk for 60 steps
        (MOTION_RUN, 60),       # run for 60 steps
        (MOTION_WALK, 40),      # walk for 40 steps
        (MOTION_STAND, 40),     # stop for 40 steps
        (MOTION_RUN, 100),      # run for 100 steps
    ],
    'dance-stand-dance': [
        (MOTION_DANCE, 80),
        (MOTION_STAND, 40),
        (MOTION_DANCE, 80),
    ],
    'full-cycle': [
        (MOTION_STAND, 30),
        (MOTION_WALK, 60),
        (MOTION_RUN, 60),
        (MOTION_DANCE, 60),
        (MOTION_STAND, 30),
        (MOTION_WALK, 60),
    ],
}


def load_stage2_model(checkpoint_path, device, motion_in_obs=None):
    """Load Stage 2 multi-motion model.
    
    Auto-detects observation mode from checkpoint if motion_in_obs is None.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Auto-detect mode from checkpoint metadata
    if motion_in_obs is None:
        motion_in_obs = ckpt.get('motion_in_obs', True)  # default legacy for old ckpts
    
    if motion_in_obs:
        obs_dim = OBS_DIM_MULTI              # 20
        priv_dim = TEACHER_PRIV_INFO_DIM     # 113
        proprio_dim = STUDENT_PROPRIO_DIM_MULTI  # 20
    else:
        obs_dim = OBS_DIM_REALISTIC          # 16
        priv_dim = TEACHER_PRIV_INFO_DIM_MULTI  # 117
        proprio_dim = STUDENT_PROPRIO_DIM_REALISTIC  # 16
    
    model = ProKneePolicy(
        obs_dim=obs_dim,
        action_dim=4,
        priv_info_dim=priv_dim,
        proprio_dim=proprio_dim,
        history_len=PROPRIO_HISTORY_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64],
        mode='student',
    ).to(device)

    running_mean_std = RunningMeanStd(obs_dim).to(device)
    sa_mean_std = RunningMeanStd((PROPRIO_HISTORY_LEN, proprio_dim)).to(device)

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    if 'running_mean_std' in ckpt:
        running_mean_std.load_state_dict(ckpt['running_mean_std'])
    if 'sa_mean_std' in ckpt:
        sa_mean_std.load_state_dict(ckpt['sa_mean_std'])

    model.eval()
    running_mean_std.eval()
    sa_mean_std.eval()

    return model, running_mean_std, sa_mean_std, motion_in_obs


@torch.no_grad()
def evaluate_single_motion(env, model, running_mean_std, sa_mean_std, motion_id, num_episodes, device):
    """Evaluate a single motion type across multiple episodes."""
    env.set_all_motions(motion_id)
    obs_dict = env.reset()

    ep_lens = []
    ep_rewards = []
    step_count = 0
    cumulative_reward = torch.zeros(env.num_envs, device=device)
    ep_length = torch.zeros(env.num_envs, device=device)

    while len(ep_lens) < num_episodes:
        obs_norm = running_mean_std(obs_dict['obs'])
        hist_norm = sa_mean_std(obs_dict['proprio_hist'])

        student_latent = model.adaptation.adapt_tconv(hist_norm)
        action_mean, _, _ = model.actor_critic(obs_norm, student_latent)
        action = action_mean.clamp(-1.0, 1.0)

        obs_dict, reward, done, info = env.step(action)
        cumulative_reward += reward
        ep_length += 1
        step_count += 1

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            for idx in done_ids:
                ep_lens.append(ep_length[idx].item())
                ep_rewards.append(cumulative_reward[idx].item())
            cumulative_reward[done_ids] = 0
            ep_length[done_ids] = 0

        if step_count > num_episodes * 350:
            break

    return {
        'motion': MOTION_NAMES[motion_id],
        'num_episodes': len(ep_lens),
        'avg_ep_len': float(np.mean(ep_lens)) if ep_lens else 0,
        'std_ep_len': float(np.std(ep_lens)) if ep_lens else 0,
        'avg_reward': float(np.mean(ep_rewards)) if ep_rewards else 0,
        'survival_rate': float(np.mean([1 if l >= 290 else 0 for l in ep_lens])) if ep_lens else 0,
    }


@torch.no_grad()
def evaluate_sequence(env, model, running_mean_std, sa_mean_std, sequence_name, sequence,
                      num_trials, device, visualize=False, soft_reset=False,
                      smooth_transition=True, transition_frames=30):
    """Evaluate a motion sequence across multiple trials.

    Each trial runs the full sequence on all environments simultaneously.
    A trial succeeds if the agent survives the entire sequence.

    Args:
        soft_reset: If True, use soft_reset_to_motion() (hard teleport).
        smooth_transition: If True (default), use smooth N-frame interpolation.
        transition_frames: Duration of smooth transition in frames.
    """
    total_steps = sum(duration for _, duration in sequence)
    results = {'trials': [], 'sequence': sequence_name, 'total_steps': total_steps}
    transitions = []

    for trial in range(num_trials):
        # Start with first motion
        env.set_all_motions(sequence[0][0])
        obs_dict = env.reset()

        alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        cumulative_reward = torch.zeros(env.num_envs, device=device)
        step_in_sequence = 0

        # Track transition smoothness (action diff at switch points)
        prev_action = None

        for seg_idx, (motion_id, duration) in enumerate(sequence):
            if seg_idx == 0:
                # First segment already set by reset above
                pass
            elif smooth_transition:
                obs_dict = env.smooth_transition_to_motion(
                    motion_id, transition_frames=transition_frames)
            elif soft_reset:
                obs_dict = env.soft_reset_to_motion(motion_id)
            else:
                env.set_all_motions(motion_id)
            if visualize:
                vel = env._root_states[0, 7].item() if env.num_envs > 0 else 0
                if smooth_transition and seg_idx > 0:
                    trans_label = f' [smooth {transition_frames}f]'
                elif soft_reset and seg_idx > 0:
                    trans_label = ' [soft-reset]'
                else:
                    trans_label = ''
                print(f"  [Seg {seg_idx}] → {MOTION_NAMES[motion_id]} "
                      f"for {duration} steps (vel_x={vel:+.2f}, "
                      f"alive: {alive.sum().item()}/{env.num_envs})"
                      f"{trans_label}")

            for step in range(duration):
                obs_norm = running_mean_std(obs_dict['obs'])
                hist_norm = sa_mean_std(obs_dict['proprio_hist'])

                student_latent = model.adaptation.adapt_tconv(hist_norm)
                action_mean, _, _ = model.actor_critic(obs_norm, student_latent)
                action = action_mean.clamp(-1.0, 1.0)

                # Track transition smoothness at segment boundaries
                if step == 0 and prev_action is not None:
                    action_diff = (action - prev_action).abs().mean().item()
                    transitions.append({
                        'from': MOTION_NAMES[sequence[seg_idx - 1][0]],
                        'to': MOTION_NAMES[motion_id],
                        'action_diff': action_diff,
                    })

                prev_action = action.clone()

                obs_dict, reward, done, info = env.step(action)
                cumulative_reward += reward * alive.float()
                alive &= ~done.bool()
                step_in_sequence += 1

                if visualize and step % 50 == 0 and step > 0:
                    print(f"    step {step}/{duration}, alive: {alive.sum().item()}/{env.num_envs}")

        survival_rate = alive.float().mean().item()
        avg_reward = cumulative_reward.mean().item()

        results['trials'].append({
            'trial': trial,
            'survival_rate': survival_rate,
            'avg_reward': avg_reward,
        })
        if visualize:
            print(f"  Trial {trial}: survival={survival_rate:.1%}, reward={avg_reward:.1f}")

    # Aggregate results
    survival_rates = [t['survival_rate'] for t in results['trials']]
    avg_rewards = [t['avg_reward'] for t in results['trials']]
    results['avg_survival_rate'] = float(np.mean(survival_rates))
    results['avg_reward'] = float(np.mean(avg_rewards))
    results['transitions'] = transitions

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Multi-Motion Sequences")
    p.add_argument("--checkpoint", type=str,
                    default="outputs/checkpoints/stage2_multi_switch/best.pth")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--episodes", type=int, default=50,
                    help="Episodes per single-motion evaluation")
    p.add_argument("--trials", type=int, default=5,
                    help="Trials per sequence evaluation")
    p.add_argument("--output", type=str, default=None,
                    help="Output JSON file path")
    p.add_argument("--blend-steps", type=int, default=50,
                    help="Transition blending steps (dual-policy live blend, 0 to disable)")
    p.add_argument("--soft-reset", action="store_true", default=False,
                    help="Use hard soft_reset (teleport) on motion switch")
    p.add_argument("--smooth-transition", action="store_true", default=True,
                    help="Use smooth N-frame interpolation (default)")
    p.add_argument("--no-smooth-transition", dest="smooth_transition", action="store_false",
                    help="Disable smooth transition")
    p.add_argument("--transition-frames", type=int, default=30,
                    help="Smooth transition duration in frames (default: 30)")
    p.add_argument("--visualize", action="store_true",
                    help="Show Isaac Gym viewer (headless=False)")
    p.add_argument("--sequence", type=str, default=None,
                    help="Run only a specific sequence (e.g., walk-stop-walk)")
    p.add_argument("--no-motion-in-obs", action="store_true",
                    help="Realistic mode: motion one-hot in priv_info, not obs")
    p.add_argument("--motion-in-obs", action="store_true",
                    help="Legacy mode: motion one-hot in obs (auto-detected from checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ProKnee Multi-Motion Sequence Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print(f"  Num envs:   {args.num_envs}")
    print(f"  Blend steps:{args.blend_steps}")
    print(f"  Soft reset: {args.soft_reset}")
    print(f"  Smooth:     {args.smooth_transition} ({args.transition_frames} frames)")

    # Determine observation mode: explicit flag > auto-detect from checkpoint
    if args.motion_in_obs:
        motion_in_obs = True
    elif args.no_motion_in_obs:
        motion_in_obs = False
    else:
        ckpt_meta = torch.load(args.checkpoint, map_location='cpu')
        motion_in_obs = ckpt_meta.get('motion_in_obs', True)
        del ckpt_meta
    mode_str = "legacy (obs=20D)" if motion_in_obs else "realistic (obs=16D)"
    print(f"  Obs mode:   {mode_str}")

    # Create environment
    print("\n[1/3] Creating environment...")
    num_envs = args.num_envs
    if args.visualize and num_envs > 4:
        num_envs = 1
        print(f"  (--visualize active, using {num_envs} env for single-agent view)")
    env = ProKneeMultiMotionEnv(
        num_envs=num_envs,
        device=args.device,
        headless=not args.visualize,
        enabled_motions=[MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND],
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=1000,
        transition_blend_steps=args.blend_steps,
        motion_in_obs=motion_in_obs,
    )
    env.manual_motion_control = True

    # Load model
    print("[2/3] Loading Stage 2 model...")
    model, rms, sa_rms, _ = load_stage2_model(args.checkpoint, args.device, motion_in_obs)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'per_motion': {},
        'sequences': {},
    }

    # ── Part 1: Per-motion evaluation (skip when visualizing) ──
    print("\n[3/3] Evaluating...")
    if args.visualize:
        print("\n── Skipping per-motion evaluation in visualize mode ──")
    else:
        print("\n── Per-Motion Evaluation ──")
        for mid in [MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND]:
            name = MOTION_NAMES[mid]
            print(f"  Evaluating {name}...", end=' ', flush=True)
            result = evaluate_single_motion(env, model, rms, sa_rms, mid, args.episodes, args.device)
            print(f"ep_len={result['avg_ep_len']:.1f}±{result['std_ep_len']:.1f} "
                  f"survival={result['survival_rate']:.1%}")
            all_results['per_motion'][name] = result

    # ── Part 2: Sequence evaluation ──
    print("\n── Sequence Evaluation ──")
    seqs_to_eval = SEQUENCES
    if args.sequence:
        if args.sequence in SEQUENCES:
            seqs_to_eval = {args.sequence: SEQUENCES[args.sequence]}
        else:
            print(f"  WARNING: unknown sequence '{args.sequence}', available: {list(SEQUENCES.keys())}")
    for seq_name, sequence in seqs_to_eval.items():
        seq_desc = ' → '.join(f"{MOTION_NAMES[m]}({d})" for m, d in sequence)
        print(f"  {seq_name}: {seq_desc}")
        result = evaluate_sequence(env, model, rms, sa_rms, seq_name, sequence, args.trials,
                                   args.device, visualize=args.visualize,
                                   soft_reset=args.soft_reset,
                                   smooth_transition=args.smooth_transition,
                                   transition_frames=args.transition_frames)
        print(f"    survival={result['avg_survival_rate']:.1%} "
              f"reward={result['avg_reward']:.1f}")
        if result['transitions']:
            for t in result['transitions'][:3]:
                print(f"    transition {t['from']}→{t['to']}: "
                      f"action_diff={t['action_diff']:.4f}")
        all_results['sequences'][seq_name] = result

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Motion':<12} {'Ep Len':>8} {'Survival':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10}")
    for name, r in all_results['per_motion'].items():
        print(f"  {name:<12} {r['avg_ep_len']:>8.1f} {r['survival_rate']:>9.1%}")
    print()
    print(f"  {'Sequence':<30} {'Survival':>10}")
    print(f"  {'-'*30} {'-'*10}")
    for name, r in all_results['sequences'].items():
        print(f"  {name:<30} {r['avg_survival_rate']:>9.1%}")

    # Save results
    if args.output is None:
        args.output = os.path.join(ROOT, "outputs", "eval_multi_motion_results.json")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
