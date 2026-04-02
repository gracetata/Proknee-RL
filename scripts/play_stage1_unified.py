#!/usr/bin/env python3
"""
Play / visualize Unified Stage 1 Teacher (privileged obs + velocity in priv_info).

Unlike interactive_unified.py (Stage 2 Student), this uses mode='teacher':
  policy(obs, priv_info=priv_info) -> action_mean

Training used raw obs/priv without RunningMeanStd — same here.

Usage:
  conda activate proknee_tc
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  python scripts/play_stage1_unified.py --device cuda:0 \\
      --checkpoint outputs/checkpoints/stage1_unified/best.pth \\
      --body-policy outputs/checkpoints/stage0/stage0_unified_1800.pth
"""

import isaacgym  # noqa: F401
from isaacgym import gymapi

import os
import sys
import argparse

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_unified import ProKneeUnifiedEnv
from proknee_hora.envs.constants import LATENT_DIM, PROPRIO_HISTORY_LEN, PROSTHESIS_ACTION_DIM
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
from proknee_hora.algo.models.actor_critic import ProKneePolicy

VELOCITY_STEP = 0.25


def velocity_to_label(v):
    if v == 0.0:
        return "STAND"
    if abs(v - 1.0) < 0.1:
        return "WALK"
    if v >= 2.5:
        return "RUN"
    return f"{v:.2f} m/s"


def load_teacher(checkpoint_path: str, device: str) -> ProKneePolicy:
    policy = ProKneePolicy(
        obs_dim=OBS_DIM_UNIFIED,
        action_dim=PROSTHESIS_ACTION_DIM,
        priv_info_dim=TEACHER_PRIV_INFO_DIM_UNIFIED,
        proprio_dim=STUDENT_PROPRIO_DIM_UNIFIED,
        history_len=PROPRIO_HISTORY_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64],
        mode="teacher",
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("policy_state_dict", ckpt.get("model", ckpt))
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def parse_args():
    p = argparse.ArgumentParser(description="Play Unified Stage 1 Teacher")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/stage1_unified/best.pth",
        help="Stage 1 unified teacher checkpoint",
    )
    p.add_argument(
        "--body-policy",
        type=str,
        default="outputs/checkpoints/stage0/stage0_unified_1800.pth",
        help="Unified Stage 0 body policy",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--initial-velocity", type=float, default=VELOCITY_WALK)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Missing checkpoint: {args.checkpoint}")
        sys.exit(1)
    if not os.path.isfile(args.body_policy):
        print(f"ERROR: Missing body policy: {args.body_policy}")
        sys.exit(1)

    print("=" * 60)
    print("  ProKnee Unified Stage 1 Teacher (play)")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Body policy: {args.body_policy}")
    print(f"  Initial vel: {args.initial_velocity} m/s")
    print()
    print("  Keyboard: ↑↓ / W R S / 0-9 / Q quit")
    print("=" * 60)

    env = ProKneeUnifiedEnv(
        num_envs=1,
        device=args.device,
        headless=False,
        body_policy_checkpoint=args.body_policy,
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=100000,
        initial_velocity=args.initial_velocity,
    )
    env.manual_velocity_control = True

    gym = env.gym
    viewer = env.viewer
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "vel_up")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "vel_down")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "walk")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "run")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "stand")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")
    for i, key in enumerate(
        [
            gymapi.KEY_0, gymapi.KEY_1, gymapi.KEY_2, gymapi.KEY_3, gymapi.KEY_4,
            gymapi.KEY_5, gymapi.KEY_6, gymapi.KEY_7, gymapi.KEY_8, gymapi.KEY_9,
        ]
    ):
        gym.subscribe_viewer_keyboard_event(viewer, key, f"num_{i}")

    policy = load_teacher(args.checkpoint, args.device)

    current_vel = args.initial_velocity
    env.set_velocity(current_vel)
    obs_dict = env.reset()
    step_count = 0

    print(f"\n  Start vel: {current_vel:.2f} m/s ({velocity_to_label(current_vel)})\n")

    try:
        while not gym.query_viewer_has_closed(viewer):
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value <= 0:
                    continue
                new_vel = None
                if evt.action == "quit":
                    print("\n  [Q] Quit")
                    env.close()
                    return
                if evt.action == "vel_up":
                    new_vel = min(VELOCITY_MAX, current_vel + VELOCITY_STEP)
                elif evt.action == "vel_down":
                    new_vel = max(VELOCITY_MIN, current_vel - VELOCITY_STEP)
                elif evt.action == "walk":
                    new_vel = VELOCITY_WALK
                elif evt.action == "run":
                    new_vel = VELOCITY_RUN
                elif evt.action == "stand":
                    new_vel = VELOCITY_STAND
                elif evt.action.startswith("num_"):
                    num = int(evt.action.split("_")[1])
                    new_vel = num * 0.25
                if new_vel is not None and new_vel != current_vel:
                    current_vel = new_vel
                    env.set_velocity(current_vel)
                    print(f"  Velocity → {current_vel:.2f} m/s ({velocity_to_label(current_vel)})")

            obs = obs_dict["obs"]
            priv_info = obs_dict["priv_info"]
            out = policy(obs, priv_info=priv_info)
            action = out["action_mean"].clamp(-1.0, 1.0)

            obs_dict, reward, done, info = env.step(action)
            step_count += 1

            if done.any():
                print(f"\n  [!] Fall at step {step_count}, reset (vel unchanged)")
                obs_dict = env.reset()
                env.set_velocity(current_vel)

    except KeyboardInterrupt:
        print("\n  Interrupted")

    env.close()
    print("  Done.")


if __name__ == "__main__":
    main()
