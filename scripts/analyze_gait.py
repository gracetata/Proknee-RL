#!/usr/bin/env python3
"""
Gait Analysis: Compare RL prosthesis knee/ankle curves to healthy reference.

Records per-step joint angles and torques from Stage 2 multi-motion model,
comparing:
  - LEFT_KNEE (DOF 24)  = prosthesis knee (RL policy output)
  - RIGHT_KNEE (DOF 17) = healthy knee (body policy / frozen Stage 0)
  - LEFT_ANKLE_X (25)   = prosthesis ankle
  - RIGHT_ANKLE_X (18)  = healthy ankle

Generates comparison plots saved to outputs/gait_analysis/.

Usage:
    python scripts/analyze_gait.py --device cuda:0
    python scripts/analyze_gait.py --device cuda:0 --motions walk run stand
"""

import isaacgym  # noqa: F401

import os
import sys
import argparse
from collections import defaultdict

import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_multi_motion import ProKneeMultiMotionEnv
from proknee_hora.envs.constants_multi import (
    OBS_DIM_MULTI, TEACHER_PRIV_INFO_DIM, STUDENT_PROPRIO_DIM_MULTI,
    LATENT_DIM, PROPRIO_HISTORY_LEN,
    MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND,
    MOTION_NAMES,
)
from proknee_hora.envs.constants import (
    LEFT_KNEE, LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z,
    RIGHT_KNEE, RIGHT_ANKLE_X, RIGHT_ANKLE_Y, RIGHT_ANKLE_Z,
    LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y,
    RIGHT_HIP_X, RIGHT_HIP_Z, RIGHT_HIP_Y,
)
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.algo.models.running_mean_std import RunningMeanStd

MOTION_NAME_TO_ID = {'walk': MOTION_WALK, 'run': MOTION_RUN, 'dance': MOTION_DANCE, 'stand': MOTION_STAND}

# Joint groups for analysis
JOINT_GROUPS = {
    'knee': {
        'prosthesis': [LEFT_KNEE],
        'healthy': [RIGHT_KNEE],
        'labels': ['Knee Flexion'],
    },
    'ankle': {
        'prosthesis': [LEFT_ANKLE_X],
        'healthy': [RIGHT_ANKLE_X],
        'labels': ['Ankle Dorsi/Plantar'],
    },
    'hip': {
        'prosthesis': [LEFT_HIP_Y],
        'healthy': [RIGHT_HIP_Y],
        'labels': ['Hip Flexion'],
    },
}


def load_model(checkpoint_path, device):
    """Load Stage 2 multi-motion model."""
    model = ProKneePolicy(
        obs_dim=OBS_DIM_MULTI,
        action_dim=4,
        priv_info_dim=TEACHER_PRIV_INFO_DIM,
        proprio_dim=STUDENT_PROPRIO_DIM_MULTI,
        history_len=PROPRIO_HISTORY_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64],
        mode='student',
    ).to(device)

    running_mean_std = RunningMeanStd(OBS_DIM_MULTI).to(device)
    sa_mean_std = RunningMeanStd((PROPRIO_HISTORY_LEN, STUDENT_PROPRIO_DIM_MULTI)).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
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
    return model, running_mean_std, sa_mean_std


@torch.no_grad()
def record_gait_data(env, model, rms, sa_rms, motion_id, num_steps, device):
    """Record joint angles and torques for a single motion type.

    Returns dict with arrays of shape (num_steps,) for each joint.
    """
    env.set_all_motions(motion_id)
    obs_dict = env.reset()

    data = defaultdict(list)

    for step in range(num_steps):
        obs_norm = rms(obs_dict['obs'])
        hist_norm = sa_rms(obs_dict['proprio_hist'])
        student_latent = model.adaptation.adapt_tconv(hist_norm)
        action_mean, _, _ = model.actor_critic(obs_norm, student_latent)
        action = action_mean.clamp(-1.0, 1.0)

        obs_dict, reward, done, info = env.step(action)

        # Record joint positions (angles in radians)
        dof_pos = env._dof_pos[0].cpu().numpy()  # env 0
        dof_force = env.dof_force_tensor[0].cpu().numpy()

        # Knee
        data['prosthesis_knee_angle'].append(np.degrees(dof_pos[LEFT_KNEE]))
        data['healthy_knee_angle'].append(np.degrees(dof_pos[RIGHT_KNEE]))
        data['prosthesis_knee_torque'].append(dof_force[LEFT_KNEE])
        data['healthy_knee_torque'].append(dof_force[RIGHT_KNEE])

        # Ankle
        data['prosthesis_ankle_angle'].append(np.degrees(dof_pos[LEFT_ANKLE_X]))
        data['healthy_ankle_angle'].append(np.degrees(dof_pos[RIGHT_ANKLE_X]))
        data['prosthesis_ankle_torque'].append(dof_force[LEFT_ANKLE_X])
        data['healthy_ankle_torque'].append(dof_force[RIGHT_ANKLE_X])

        # Hip
        data['prosthesis_hip_angle'].append(np.degrees(dof_pos[LEFT_HIP_Y]))
        data['healthy_hip_angle'].append(np.degrees(dof_pos[RIGHT_HIP_Y]))
        data['prosthesis_hip_torque'].append(dof_force[LEFT_HIP_Y])
        data['healthy_hip_torque'].append(dof_force[RIGHT_HIP_Y])

        # Root height (for context)
        data['root_height'].append(env._root_states[0, 2].cpu().item())

        # Check if agent fell
        if done[0].item():
            print(f"    Agent fell at step {step}/{num_steps}")
            break

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return dict(data)


def plot_gait_comparison(all_data, output_dir):
    """Generate comparison plots for all motions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    motions = list(all_data.keys())
    joints = ['knee', 'ankle', 'hip']
    measures = ['angle', 'torque']

    # ── Plot 1: Per-motion joint angle comparison (prosthesis vs healthy) ──
    for joint in joints:
        fig, axes = plt.subplots(len(motions), 2, figsize=(14, 4 * len(motions)), squeeze=False)
        fig.suptitle(f'{joint.capitalize()} Joint — Prosthesis vs Healthy', fontsize=14, fontweight='bold')

        for row, motion_name in enumerate(motions):
            data = all_data[motion_name]
            steps = np.arange(len(data[f'prosthesis_{joint}_angle']))
            time_s = steps * (2.0 / 180.0)  # dt=1/180, control_freq_inv=2

            # Angle subplot
            ax = axes[row, 0]
            ax.plot(time_s, data[f'prosthesis_{joint}_angle'], 'r-', linewidth=1.5, label='Prosthesis (Left)')
            ax.plot(time_s, data[f'healthy_{joint}_angle'], 'b--', linewidth=1.5, label='Healthy (Right)')
            ax.set_ylabel(f'{motion_name}\nAngle (deg)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title('Joint Angle')

            # Torque subplot
            ax = axes[row, 1]
            ax.plot(time_s, data[f'prosthesis_{joint}_torque'], 'r-', linewidth=1.5, label='Prosthesis (Left)')
            ax.plot(time_s, data[f'healthy_{joint}_torque'], 'b--', linewidth=1.5, label='Healthy (Right)')
            ax.set_ylabel('Torque (Nm)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title('Joint Torque')

        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')

        plt.tight_layout()
        path = os.path.join(output_dir, f'{joint}_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ── Plot 2: All-in-one knee angle comparison ──
    fig, axes = plt.subplots(1, len(motions), figsize=(5 * len(motions), 4), squeeze=False)
    fig.suptitle('Knee Angle: Prosthesis vs Healthy (all motions)', fontsize=13, fontweight='bold')
    for col, motion_name in enumerate(motions):
        data = all_data[motion_name]
        steps = np.arange(len(data['prosthesis_knee_angle']))
        time_s = steps * (2.0 / 180.0)
        ax = axes[0, col]
        ax.plot(time_s, data['prosthesis_knee_angle'], 'r-', linewidth=1.5, label='Prosthesis')
        ax.plot(time_s, data['healthy_knee_angle'], 'b--', linewidth=1.5, label='Healthy')
        ax.set_title(motion_name.capitalize())
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'knee_angle_all_motions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Root height overlay ──
    fig, ax = plt.subplots(figsize=(10, 4))
    for motion_name in motions:
        data = all_data[motion_name]
        steps = np.arange(len(data['root_height']))
        time_s = steps * (2.0 / 180.0)
        ax.plot(time_s, data['root_height'], linewidth=1.5, label=motion_name.capitalize())
    ax.set_title('Root Height Over Time (all motions)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'root_height.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── Summary statistics ──
    print("\n  ── Gait Statistics Summary ──")
    print(f"  {'Motion':<10} {'Joint':<8} {'Prosth Mean':>12} {'Health Mean':>12} {'RMSE':>8} {'Correlation':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*12}")
    for motion_name in motions:
        data = all_data[motion_name]
        for joint in joints:
            p_angle = data[f'prosthesis_{joint}_angle']
            h_angle = data[f'healthy_{joint}_angle']
            min_len = min(len(p_angle), len(h_angle))
            p_angle = p_angle[:min_len]
            h_angle = h_angle[:min_len]
            rmse = np.sqrt(np.mean((p_angle - h_angle) ** 2))
            corr = np.corrcoef(p_angle, h_angle)[0, 1] if min_len > 1 else 0.0
            print(f"  {motion_name:<10} {joint:<8} {np.mean(p_angle):>10.1f}° {np.mean(h_angle):>10.1f}° "
                  f"{rmse:>6.1f}° {corr:>10.3f}")

    return True


def parse_args():
    p = argparse.ArgumentParser(description="Gait Analysis: RL Prosthesis vs Healthy Reference")
    p.add_argument("--checkpoint", type=str,
                    default="outputs/checkpoints/stage2_multi_switch/best.pth")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-steps", type=int, default=300,
                    help="Steps to record per motion")
    p.add_argument("--motions", nargs='+', default=['walk', 'run', 'dance', 'stand'],
                    help="Motions to analyze")
    p.add_argument("--output-dir", type=str, default="outputs/gait_analysis")
    p.add_argument("--blend-steps", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ProKnee Gait Analysis")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Motions:    {args.motions}")
    print(f"  Steps:      {args.num_steps}")

    # Validate motions
    motion_ids = []
    for name in args.motions:
        if name not in MOTION_NAME_TO_ID:
            print(f"  WARNING: unknown motion '{name}', skipping")
            continue
        motion_ids.append((name, MOTION_NAME_TO_ID[name]))

    # Create environment (single env for clean recording)
    print("\n[1/3] Creating environment...")
    env = ProKneeMultiMotionEnv(
        num_envs=1,
        device=args.device,
        headless=True,
        enabled_motions=[MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND],
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=args.num_steps + 10,
        transition_blend_steps=args.blend_steps,
    )
    env.manual_motion_control = True

    # Load model
    print("[2/3] Loading model...")
    model, rms, sa_rms = load_model(args.checkpoint, args.device)

    # Record gait data
    print("[3/3] Recording gait data...")
    all_data = {}
    for motion_name, motion_id in motion_ids:
        print(f"  Recording {motion_name} ({args.num_steps} steps)...", end=' ', flush=True)
        data = record_gait_data(env, model, rms, sa_rms, motion_id, args.num_steps, args.device)
        all_data[motion_name] = data
        print(f"done ({len(data['prosthesis_knee_angle'])} steps recorded)")

    # Save raw data
    output_dir = os.path.join(ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'gait_data.npz'), **{
        f"{motion}_{key}": arr for motion, data in all_data.items() for key, arr in data.items()
    })
    print(f"\n  Raw data saved to: {output_dir}/gait_data.npz")

    # Generate plots
    print("\n  Generating plots...")
    plot_gait_comparison(all_data, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete! Results in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
