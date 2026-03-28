#!/usr/bin/env python3
"""
Interactive unified velocity-controlled visualization with keyboard control.

Keyboard controls:
    ↑/↓ = Increase/Decrease target velocity by 0.25
    W = Set velocity to 1.0 (Walk)
    R = Set velocity to 2.5 (Run)
    S = Set velocity to 0.0 (Stand)
    0-9 = Direct velocity setting (0=0.0, 5=1.25, 9=2.25)
    Q = Quit

The prosthesis Student policy infers velocity from proprio_hist
(no explicit velocity command input to the prosthesis).

Usage:
    # With unified Stage 2 model
    $PYTHON scripts/interactive_unified.py --device cuda:0

    # Specify checkpoint
    $PYTHON scripts/interactive_unified.py --device cuda:0 \
        --checkpoint outputs/checkpoints/stage2_unified/best.pth

    # Start with specific velocity
    $PYTHON scripts/interactive_unified.py --device cuda:0 --initial-velocity 1.0
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

from proknee_hora.envs.proknee_unified import ProKneeUnifiedEnv
from proknee_hora.envs.constants_unified import (
    OBS_DIM_UNIFIED,
    TEACHER_PRIV_INFO_DIM_UNIFIED,
    STUDENT_PROPRIO_DIM_UNIFIED,
    VELOCITY_MIN,
    VELOCITY_MAX,
    VELOCITY_STAND,
    VELOCITY_WALK,
    VELOCITY_RUN,
)
from proknee_hora.envs.constants import (
    LATENT_DIM, PROPRIO_HISTORY_LEN,
)
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.algo.models.running_mean_std import RunningMeanStd


VELOCITY_STEP = 0.25  # Step size for ↑/↓ keys


def load_model(checkpoint_path, device):
    """Load Stage 2 unified model."""
    obs_dim = OBS_DIM_UNIFIED  # 16
    priv_dim = TEACHER_PRIV_INFO_DIM_UNIFIED  # 114
    proprio_dim = STUDENT_PROPRIO_DIM_UNIFIED  # 16

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
    p = argparse.ArgumentParser(description="Interactive Unified Velocity-Controlled Visualization")
    p.add_argument("--checkpoint", type=str,
                    default="outputs/checkpoints/stage2_unified/best.pth",
                    help="Stage 2 unified checkpoint")
    p.add_argument("--body-policy", type=str,
                    default="outputs/checkpoints/stage0/stage0_unified_1800.pth",
                    help="Unified body policy checkpoint")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--initial-velocity", type=float, default=VELOCITY_WALK,
                    help=f"Initial velocity command (default: {VELOCITY_WALK})")
    return p.parse_args()


def velocity_to_label(v):
    """Convert velocity to human-readable label."""
    if v == 0.0:
        return "STAND"
    elif abs(v - 1.0) < 0.1:
        return "WALK"
    elif v >= 2.5:
        return "RUN"
    else:
        return f"{v:.2f} m/s"


@torch.no_grad()
def main():
    args = parse_args()

    # Check files exist
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Run Stage 2 Unified training first, or specify --checkpoint path.")
        sys.exit(1)
    if not os.path.exists(args.body_policy):
        print(f"ERROR: Body policy not found: {args.body_policy}")
        sys.exit(1)

    print("=" * 60)
    print("  ProKnee Interactive Unified Velocity Control")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Body policy: {args.body_policy}")
    print(f"  Initial vel: {args.initial_velocity} m/s")
    print()
    print("  Keyboard Controls:")
    print("    ↑ = Increase velocity (+0.25)")
    print("    ↓ = Decrease velocity (-0.25)")
    print("    W = Walk (1.0 m/s)")
    print("    R = Run (2.5 m/s)")
    print("    S = Stand (0.0 m/s)")
    print("    0-9 = Direct set (0=0.0, 5=1.25, 9=2.25)")
    print("    Q = Quit")
    print("=" * 60)

    # Create environment
    env = ProKneeUnifiedEnv(
        num_envs=1,
        device=args.device,
        headless=False,
        body_policy_checkpoint=args.body_policy,
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=100000,  # Effectively infinite
        initial_velocity=args.initial_velocity,
    )
    env.manual_velocity_control = True  # Disable auto velocity switching

    # Subscribe keyboard events
    gym = env.gym
    viewer = env.viewer
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "vel_up")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "vel_down")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "walk")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "run")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "stand")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")
    # Number keys 0-9
    num_keys = [
        gymapi.KEY_0, gymapi.KEY_1, gymapi.KEY_2, gymapi.KEY_3, gymapi.KEY_4,
        gymapi.KEY_5, gymapi.KEY_6, gymapi.KEY_7, gymapi.KEY_8, gymapi.KEY_9
    ]
    for i, key in enumerate(num_keys):
        gym.subscribe_viewer_keyboard_event(viewer, key, f"num_{i}")

    # Load model
    model, rms, sa_rms = load_model(args.checkpoint, args.device)

    # Initialize
    current_vel = args.initial_velocity
    env.set_velocity(current_vel)
    obs_dict = env.reset()
    step_count = 0
    total_reward = 0.0
    fall_count = 0

    print(f"\n  Starting with velocity: {current_vel:.2f} m/s ({velocity_to_label(current_vel)})")
    print("  Press ↑/↓ to adjust, W/R/S for presets, Q to quit\n")

    try:
        while not gym.query_viewer_has_closed(viewer):
            # Process keyboard events
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value > 0:  # Key press
                    new_vel = None
                    
                    if evt.action == "quit":
                        print("\n  [Q] Quitting...")
                        env.close()
                        return
                    elif evt.action == "vel_up":
                        new_vel = min(VELOCITY_MAX, current_vel + VELOCITY_STEP)
                        print(f"  [↑] Velocity: {current_vel:.2f} → {new_vel:.2f} m/s")
                    elif evt.action == "vel_down":
                        new_vel = max(VELOCITY_MIN, current_vel - VELOCITY_STEP)
                        print(f"  [↓] Velocity: {current_vel:.2f} → {new_vel:.2f} m/s")
                    elif evt.action == "walk":
                        new_vel = VELOCITY_WALK
                        print(f"  [W] Walk: {current_vel:.2f} → {new_vel:.2f} m/s")
                    elif evt.action == "run":
                        new_vel = VELOCITY_RUN
                        print(f"  [R] Run: {current_vel:.2f} → {new_vel:.2f} m/s")
                    elif evt.action == "stand":
                        new_vel = VELOCITY_STAND
                        print(f"  [S] Stand: {current_vel:.2f} → {new_vel:.2f} m/s")
                    elif evt.action.startswith("num_"):
                        num = int(evt.action.split("_")[1])
                        # Map 0-9 to velocity range 0.0-2.25
                        new_vel = num * 0.25
                        print(f"  [{num}] Set: {current_vel:.2f} → {new_vel:.2f} m/s")

                    if new_vel is not None and new_vel != current_vel:
                        current_vel = new_vel
                        env.set_velocity(current_vel)

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
                label = velocity_to_label(current_vel)
                print(f"\r  [Status] Vel: {current_vel:.2f} ({label:5s}) | "
                      f"Step: {step_count:6d} | Reward: {total_reward:.1f} | "
                      f"Falls: {fall_count}", end='', flush=True)

            # Auto-recover from falls
            if done.any():
                fall_count += 1
                print(f"\n  [!] Agent fell at step {step_count}, resetting...")
                obs_dict = env.reset()
                # Keep current velocity
                env.set_velocity(current_vel)

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    env.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
