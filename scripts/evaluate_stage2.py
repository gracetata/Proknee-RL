#!/usr/bin/env python3
"""
Evaluate Stage 2 student policy and generate gait analysis.

Outputs:
  1. Episode statistics (avg ep_len, survival rate)
  2. Average gait cycle knee/ankle angle curves (saved as PNG)
  3. Raw joint angle data (saved as CSV)

Usage:
    # Quick evaluation (headless, uses default checkpoint paths)
    python scripts/evaluate_stage2.py --device cuda:0 --num-envs 256

    # With visualization (Isaac Gym viewer + matplotlib angle curves)
    python scripts/evaluate_stage2.py --device cuda:0 --num-envs 1 --visualize
"""

import isaacgym  # noqa: F401

import os
import sys
import argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_base import ProKneeBase
from proknee_hora.algo.proprio_adapt import ProprioAdapt
from proknee_hora.algo.models.running_mean_std import RunningMeanStd
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.envs.constants import (
    OBS_DIM, TEACHER_PRIV_INFO_DIM, STUDENT_PROPRIO_DIM,
    LATENT_DIM, PROPRIO_HISTORY_LEN,
    LEFT_KNEE, LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Stage 2 student policy")
    p.add_argument("--body-policy", type=str,
                    default="outputs/checkpoints/stage0/stage0_amp_walk_5050.pth",
                    help="Stage 0 body policy (controls 24 frozen body DOFs)")
    p.add_argument("--student-ckpt", type=str,
                    default="outputs/checkpoints/stage2/best.pth",
                    help="Stage 2 student checkpoint")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-episodes", type=int, default=300,
                    help="Max simulation steps")
    p.add_argument("--visualize", action="store_true",
                    help="Open Isaac Gym viewer + matplotlib realtime curves")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def load_student_policy(ckpt_path, device):
    """Load Stage 2 student model and normalization stats."""
    model = ProKneePolicy(
        obs_dim=OBS_DIM,
        action_dim=4,
        priv_info_dim=TEACHER_PRIV_INFO_DIM,
        proprio_dim=STUDENT_PROPRIO_DIM,
        history_len=PROPRIO_HISTORY_LEN,
        latent_dim=LATENT_DIM,
        hidden_dims=[256, 128, 64],
        mode='student',
    ).to(device)

    running_mean_std = RunningMeanStd(OBS_DIM).to(device)
    sa_mean_std = RunningMeanStd((PROPRIO_HISTORY_LEN, STUDENT_PROPRIO_DIM)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'policy_state_dict' in ckpt:
        model.load_state_dict(ckpt['policy_state_dict'])

    if 'running_mean_std' in ckpt:
        running_mean_std.load_state_dict(ckpt['running_mean_std'])
    if 'sa_mean_std' in ckpt:
        sa_mean_std.load_state_dict(ckpt['sa_mean_std'])

    model.eval()
    running_mean_std.eval()
    sa_mean_std.eval()
    return model, running_mean_std, sa_mean_std


def evaluate_headless(env, model, running_mean_std, sa_mean_std, max_steps=300):
    """Run evaluation and collect joint angle data."""
    device = next(model.parameters()).device
    N = env.num_envs

    # Storage for joint angles
    knee_angles_all = []
    ankle_x_all = []
    ankle_y_all = []
    ankle_z_all = []
    foot_contact_all = []

    alive = torch.ones(N, dtype=torch.bool, device=device)
    steps = torch.zeros(N, device=device)
    ep_lens = []

    obs_dict = env.reset()

    for t in range(max_steps):
        with torch.no_grad():
            obs_norm = running_mean_std(obs_dict['obs']).detach()
            hist_norm = sa_mean_std(obs_dict['proprio_hist'].detach())

            student_latent = model.adaptation.adapt_tconv(hist_norm)
            action_mean, _, _ = model.actor_critic(obs_norm, student_latent)
            action = torch.clamp(action_mean, -1.0, 1.0)

        # Record joint angles (radians)
        knee_angles_all.append(env._dof_pos[:, LEFT_KNEE].cpu().numpy().copy())
        ankle_x_all.append(env._dof_pos[:, LEFT_ANKLE_X].cpu().numpy().copy())
        ankle_y_all.append(env._dof_pos[:, LEFT_ANKLE_Y].cpu().numpy().copy())
        ankle_z_all.append(env._dof_pos[:, LEFT_ANKLE_Z].cpu().numpy().copy())

        # Foot contact
        left_fz = env.vec_sensor_tensor[:, 5].abs().cpu().numpy()
        foot_contact_all.append((left_fz > 1.0).copy())

        obs_dict, reward, done, info = env.step(action)

        steps += alive.float()
        newly_done = done & alive
        for idx in torch.where(newly_done)[0]:
            ep_lens.append(steps[idx].item())
        alive = alive & ~done
        if not alive.any():
            break

    # Add remaining alive envs
    for idx in torch.where(alive)[0]:
        ep_lens.append(steps[idx].item())

    return {
        'ep_lens': np.array(ep_lens),
        'knee': np.array(knee_angles_all),    # (T, N)
        'ankle_x': np.array(ankle_x_all),
        'ankle_y': np.array(ankle_y_all),
        'ankle_z': np.array(ankle_z_all),
        'foot_contact': np.array(foot_contact_all),
    }


def detect_gait_cycles(contact, min_cycle_len=20, max_cycle_len=120):
    """Detect gait cycles from foot contact signal.
    
    A gait cycle = heel-strike to next heel-strike (contact 0→1 transitions).
    Returns list of (start, end) frame indices.
    """
    cycles = []
    in_contact = False
    cycle_start = None

    for t in range(len(contact)):
        if contact[t] and not in_contact:
            # Heel-strike detected
            if cycle_start is not None:
                cycle_len = t - cycle_start
                if min_cycle_len <= cycle_len <= max_cycle_len:
                    cycles.append((cycle_start, t))
            cycle_start = t
        in_contact = bool(contact[t])

    return cycles


def compute_average_gait_cycle(data, cycles, num_points=100):
    """Resample each gait cycle to fixed length and average."""
    if not cycles:
        return None

    resampled = []
    for start, end in cycles:
        segment = data[start:end]
        # Resample to num_points
        x_old = np.linspace(0, 1, len(segment))
        x_new = np.linspace(0, 1, num_points)
        resampled.append(np.interp(x_new, x_old, segment))

    return np.mean(resampled, axis=0), np.std(resampled, axis=0), len(cycles)


def plot_gait_analysis(results, output_path):
    """Plot average gait cycle joint angles."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ProKnee Stage 2 — Average Gait Cycle Joint Angles', fontsize=14)

    joint_data = {
        'Knee': results['knee'],
        'Ankle X (Sagittal)': results['ankle_x'],
        'Ankle Y': results['ankle_y'],
        'Ankle Z': results['ankle_z'],
    }

    contact = results['foot_contact']
    N = results['knee'].shape[1]

    for ax_idx, (name, data) in enumerate(joint_data.items()):
        ax = axes[ax_idx // 2, ax_idx % 2]

        all_means = []
        all_stds = []
        total_cycles = 0

        for env_i in range(min(N, 64)):  # Use up to 64 envs for averaging
            # Get contact for this env
            env_contact = contact[:, env_i]
            env_data = data[:, env_i]

            cycles = detect_gait_cycles(env_contact)
            if not cycles:
                continue

            result = compute_average_gait_cycle(env_data, cycles)
            if result is not None:
                mean, std, n = result
                all_means.append(mean)
                total_cycles += n

        if all_means:
            grand_mean = np.mean(all_means, axis=0)
            grand_std = np.std(all_means, axis=0)

            # Convert to degrees
            grand_mean_deg = np.degrees(grand_mean)
            grand_std_deg = np.degrees(grand_std)

            x = np.linspace(0, 100, len(grand_mean_deg))
            ax.plot(x, grand_mean_deg, 'b-', linewidth=2)
            ax.fill_between(x,
                            grand_mean_deg - grand_std_deg,
                            grand_mean_deg + grand_std_deg,
                            alpha=0.3, color='blue')
            ax.set_title(f'{name} ({total_cycles} cycles)')
        else:
            ax.set_title(f'{name} (no cycles detected)')

        ax.set_xlabel('Gait Cycle (%)')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Gait analysis plot saved: {output_path}")
    return output_path


def visualize_with_curves(env, model, running_mean_std, sa_mean_std, max_steps=3000):
    """Run visualization with Isaac Gym viewer + matplotlib realtime angle curves."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device

    # Setup matplotlib figure for realtime angle curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('ProKnee Stage 2 — Realtime Joint Angles', fontsize=12)

    window_len = 200  # Show last 200 frames
    joint_names = ['Knee', 'Ankle X', 'Ankle Y', 'Ankle Z']
    joint_indices = [LEFT_KNEE, LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z]
    histories = {name: [] for name in joint_names}

    lines = {}
    for i, name in enumerate(joint_names):
        ax = axes[i // 2, i % 2]
        ax.set_xlim(0, window_len)
        ax.set_ylim(-60, 60)
        ax.set_ylabel('Angle (deg)')
        ax.set_xlabel('Frame')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        line, = ax.plot([], [], 'b-', linewidth=1.5)
        lines[name] = line

    plt.tight_layout()
    plt.ion()
    plt.show()

    obs_dict = env.reset()

    for t in range(max_steps):
        with torch.no_grad():
            obs_norm = running_mean_std(obs_dict['obs']).detach()
            hist_norm = sa_mean_std(obs_dict['proprio_hist'].detach())

            student_latent = model.adaptation.adapt_tconv(hist_norm)
            action_mean, _, _ = model.actor_critic(obs_norm, student_latent)
            action = torch.clamp(action_mean, -1.0, 1.0)

        # Record angles from env 0
        for name, idx in zip(joint_names, joint_indices):
            angle_deg = np.degrees(env._dof_pos[0, idx].cpu().item())
            histories[name].append(angle_deg)

        obs_dict, reward, done, info = env.step(action)

        # Update plot every 5 frames
        if t % 5 == 0:
            for i, name in enumerate(joint_names):
                data = histories[name][-window_len:]
                x = np.arange(len(data))
                lines[name].set_data(x, data)
                ax = axes[i // 2, i % 2]
                if data:
                    y_min = min(data) - 5
                    y_max = max(data) + 5
                    ax.set_ylim(y_min, y_max)
                ax.set_xlim(0, max(len(data), window_len))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    plt.ioff()
    plt.close()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ProKnee Stage 2 Evaluation & Gait Analysis")
    print("=" * 60)

    if args.output_dir is None:
        args.output_dir = os.path.join(ROOT, "outputs", "eval_stage2")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create environment
    print("\n[1/3] Creating environment...")
    env = ProKneeBase(
        num_envs=args.num_envs,
        device=args.device,
        headless=not args.visualize,
        body_policy_checkpoint=args.body_policy,
        prosthesis_only=True,
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=args.num_episodes,
    )

    # Load student policy
    print("[2/3] Loading student policy...")
    model, running_mean_std, sa_mean_std = load_student_policy(
        args.student_ckpt, args.device
    )

    if args.visualize:
        print("[3/3] Running visualization with realtime curves...")
        visualize_with_curves(env, model, running_mean_std, sa_mean_std)
    else:
        print("[3/3] Running headless evaluation...")
        results = evaluate_headless(env, model, running_mean_std, sa_mean_std,
                                     max_steps=args.num_episodes)

        # Episode stats
        ep = results['ep_lens']
        print(f"\n  Avg episode length: {ep.mean():.1f}")
        print(f"  Std episode length: {ep.std():.1f}")
        print(f"  >= 100 steps: {(ep >= 100).sum()}/{len(ep)}")
        print(f"  >= 200 steps: {(ep >= 200).sum()}/{len(ep)}")
        print(f"  >= 300 steps: {(ep >= 300).sum()}/{len(ep)}")

        # Gait analysis plot
        plot_path = os.path.join(args.output_dir, "gait_cycle_analysis.png")
        plot_gait_analysis(results, plot_path)

        # Save raw CSV
        csv_path = os.path.join(args.output_dir, "joint_angles.csv")
        T = results['knee'].shape[0]
        N = results['knee'].shape[1]
        with open(csv_path, 'w') as f:
            f.write("timestep,env_id,knee_rad,ankle_x_rad,ankle_y_rad,ankle_z_rad,foot_contact\n")
            for t in range(T):
                for n in range(min(N, 16)):  # Save first 16 envs
                    f.write(f"{t},{n},{results['knee'][t,n]:.6f},"
                            f"{results['ankle_x'][t,n]:.6f},"
                            f"{results['ankle_y'][t,n]:.6f},"
                            f"{results['ankle_z'][t,n]:.6f},"
                            f"{int(results['foot_contact'][t,n])}\n")
        print(f"  Joint angle CSV saved: {csv_path}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
