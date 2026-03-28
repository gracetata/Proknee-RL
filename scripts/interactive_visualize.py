#!/usr/bin/env python3
"""
Interactive multi-motion visualization with keyboard control.

Keyboard controls:
    W = Walk    S = Stand    R = Run    D = Dance    Q = Quit

Uses smooth_transition_to_motion() for seamless motion switching:
  - N-frame interpolation of DOF targets (no body policy during transition)
  - Prosthesis adapts to body movement naturally
  - No teleporting or history reset

Falls back to soft_reset_to_motion() only for fall recovery.

The prosthesis does NOT receive motion commands
(in realistic mode, motion one-hot is in priv_info, not obs).

Usage:
    # With existing legacy model (obs=20D, auto-detected)
    $PYTHON scripts/interactive_visualize.py --device cuda:0

    # With realistic model (obs=16D, auto-detected)
    $PYTHON scripts/interactive_visualize.py --device cuda:0 \
        --checkpoint outputs/checkpoints/stage2_multi_realistic/best.pth

    # Force specific mode
    $PYTHON scripts/interactive_visualize.py --device cuda:0 --no-motion-in-obs

    # Adjust transition speed (default 30 frames ≈ 1 sec)
    $PYTHON scripts/interactive_visualize.py --device cuda:0 --transition-frames 20
"""

import isaacgym  # noqa: F401
from isaacgym import gymapi

import os
import sys
import argparse
import time

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

# Keyboard mapping: key → (motion_id, display_name)
KEY_MOTION_MAP = {
    'walk':  (MOTION_WALK,  'Walk'),
    'run':   (MOTION_RUN,   'Run'),
    'dance': (MOTION_DANCE, 'Dance'),
    'stand': (MOTION_STAND, 'Stand'),
}


def load_model(checkpoint_path, device, motion_in_obs):
    """Load Stage 2 model with correct dimensions."""
    if motion_in_obs:
        obs_dim = OBS_DIM_MULTI
        priv_dim = TEACHER_PRIV_INFO_DIM
        proprio_dim = STUDENT_PROPRIO_DIM_MULTI
    else:
        obs_dim = OBS_DIM_REALISTIC
        priv_dim = TEACHER_PRIV_INFO_DIM_MULTI
        proprio_dim = STUDENT_PROPRIO_DIM_REALISTIC

    model = ProKneePolicy(
        obs_dim=obs_dim, action_dim=4,
        priv_info_dim=priv_dim, proprio_dim=proprio_dim,
        history_len=PROPRIO_HISTORY_LEN, latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64], mode='student',
    ).to(device)

    rms = RunningMeanStd(obs_dim).to(device)
    sa_rms = RunningMeanStd((PROPRIO_HISTORY_LEN, proprio_dim)).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    if 'running_mean_std' in ckpt:
        rms.load_state_dict(ckpt['running_mean_std'])
    if 'sa_mean_std' in ckpt:
        sa_rms.load_state_dict(ckpt['sa_mean_std'])

    model.eval()
    rms.eval()
    sa_rms.eval()
    return model, rms, sa_rms


def parse_args():
    p = argparse.ArgumentParser(description="Interactive Multi-Motion Visualization")
    p.add_argument("--checkpoint", type=str,
                    default="outputs/checkpoints/stage2_multi_realistic/best.pth",
                    help="Stage 2 checkpoint (default: realistic model)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-motion-in-obs", action="store_true",
                    help="Realistic mode (obs=16D)")
    p.add_argument("--motion-in-obs", action="store_true",
                    help="Legacy mode (obs=20D)")
    p.add_argument("--initial-motion", type=str, default="walk",
                    choices=["walk", "run", "dance", "stand"])
    p.add_argument("--transition-frames", type=int, default=30,
                    help="Smooth transition duration in frames (default: 30 ≈ 1s)")
    p.add_argument("--hard-reset", action="store_true",
                    help="Use hard soft_reset (teleport) instead of smooth transition")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # Determine observation mode
    if args.motion_in_obs:
        motion_in_obs = True
    elif args.no_motion_in_obs:
        motion_in_obs = False
    else:
        ckpt_meta = torch.load(args.checkpoint, map_location='cpu')
        motion_in_obs = ckpt_meta.get('motion_in_obs', True)
        del ckpt_meta

    mode_str = "legacy (obs=20D)" if motion_in_obs else "realistic (obs=16D)"
    transition_mode = "hard (teleport)" if args.hard_reset else f"smooth ({args.transition_frames} frames)"

    print("=" * 60)
    print("  ProKnee Interactive Multi-Motion Visualization")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Obs mode:    {mode_str}")
    print(f"  Transition:  {transition_mode}")
    print()
    print("  Keyboard Controls:")
    print("    W = Walk    R = Run    D = Dance    S = Stand    Q = Quit")
    print("=" * 60)

    # Create environment with very long episode
    env = ProKneeMultiMotionEnv(
        num_envs=1,
        device=args.device,
        headless=False,
        enabled_motions=[MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND],
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=100000,  # effectively infinite
        motion_in_obs=motion_in_obs,
    )
    env.manual_motion_control = True

    # Subscribe keyboard events
    gym = env.gym
    viewer = env.viewer
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "walk")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "run")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "dance")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "stand")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")

    # Load model
    model, rms, sa_rms = load_model(args.checkpoint, args.device, motion_in_obs)

    # Set initial motion
    initial_id = KEY_MOTION_MAP[args.initial_motion][0]
    env.set_all_motions(initial_id)
    obs_dict = env.reset()
    current_motion = args.initial_motion
    step_count = 0

    print(f"\n  Starting with: {current_motion.upper()}")
    print("  Press W/R/D/S to switch motions, Q to quit")
    print("  Status updates shown below:\n")

    # Statistics tracking
    total_reward = 0.0
    fall_count = 0

    try:
        while not gym.query_viewer_has_closed(viewer):
            # Process keyboard events
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value > 0:  # key press (not release)
                    if evt.action == "quit":
                        print("\n  [Q] Quitting...")
                        env.close()
                        return
                    elif evt.action in KEY_MOTION_MAP:
                        mid, name = KEY_MOTION_MAP[evt.action]
                        if evt.action != current_motion:
                            if args.hard_reset:
                                print(f"  [{evt.action.upper()}] Hard-reset → {name} "
                                      f"(step {step_count})")
                                obs_dict = env.soft_reset_to_motion(mid)
                            else:
                                print(f"  [{evt.action.upper()}] Smooth → {name} "
                                      f"({args.transition_frames}f, step {step_count})")
                                obs_dict = env.smooth_transition_to_motion(
                                    mid, transition_frames=args.transition_frames)
                            current_motion = evt.action

            # Forward pass
            obs_norm = rms(obs_dict['obs'])
            hist_norm = sa_rms(obs_dict['proprio_hist'])
            student_latent = model.adaptation.adapt_tconv(hist_norm)
            action_mean, _, _ = model.actor_critic(obs_norm, student_latent)
            action = action_mean.clamp(-1.0, 1.0)

            obs_dict, reward, done, info = env.step(action)
            step_count += 1
            total_reward += reward.item()

            # Periodic status update (every 100 steps)
            if step_count % 100 == 0:
                print(f"\r  [Status] Motion: {current_motion.upper():5s} | "
                      f"Step: {step_count:6d} | Reward: {total_reward:.1f} | "
                      f"Falls: {fall_count}", end='', flush=True)

            # Auto-recover from falls (always use hard reset for recovery)
            if done.any():
                fall_count += 1
                print(f"\n  [!] Agent fell at step {step_count}, hard-resetting to {current_motion}...")
                mid = KEY_MOTION_MAP[current_motion][0]
                obs_dict = env.soft_reset_to_motion(mid)

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    env.close()
    print("  Done.")


if __name__ == "__main__":
    main()
