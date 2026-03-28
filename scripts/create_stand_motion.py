#!/usr/bin/env python3
"""
Generate a standing reference motion file (amp_humanoid_stand.npy) for AMP training.

Creates a SYMMETRIC DOUBLE-LEG STANDING pose:
  - Symmetric left/right legs with ~15° knee bend for physical stability
  - Arms at near-identity (natural rest pose)
  - Root Z ≈ 0.84 (appropriate for slightly bent knees)
  - All velocities = 0 (standing still)

Approach: extract from walk motion's most symmetric double-support frame,
then force exact left/right symmetry on the leg joints.

IMPORTANT: Must run with numpy 1.x (conda env proknee_tc).

Usage:
    python scripts/create_stand_motion.py
"""

import os
import sys
import copy
import numpy as np
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTIONS_DIR = os.path.join(PROJECT_ROOT, 'IsaacGymEnvs', 'assets', 'amp', 'motions')

# Skeleton joint indices (15 joints)
PELVIS, TORSO, HEAD = 0, 1, 2
R_UPPER_ARM, R_LOWER_ARM, R_HAND = 3, 4, 5
L_UPPER_ARM, L_LOWER_ARM, L_HAND = 6, 7, 8
R_THIGH, R_SHIN, R_FOOT = 9, 10, 11
L_THIGH, L_SHIN, L_FOOT = 12, 13, 14

NUM_JOINTS = 15
NUM_FRAMES = 30
IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw format


def quat_slerp(q0, q1, t=0.5):
    """Spherical linear interpolation between two quaternions (xyzw)."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1


def create_stand_motion():
    """Create a symmetric double-leg standing motion from walk data."""
    walk_path = os.path.join(MOTIONS_DIR, 'amp_humanoid_walk.npy')
    if not os.path.exists(walk_path):
        print(f"Error: Walk motion not found at {walk_path}")
        sys.exit(1)

    walk_data = np.load(walk_path, allow_pickle=True).item()
    rotation = walk_data['rotation']['arr']       # (F, 15, 4) xyzw
    root_trans = walk_data['root_translation']['arr']  # (F, 3)

    num_frames = rotation.shape[0]
    print(f"Walk motion: {num_frames} frames")

    # --- Strategy: find the most symmetric double-support frame ---
    # Measure leg asymmetry as ||right_leg_quats - left_leg_quats||
    # Right leg: joints 9,10,11 ; Left leg: joints 12,13,14
    best_frame = 0
    best_asym = float('inf')

    for i in range(num_frames):
        asym = 0.0
        for r_j, l_j in [(R_THIGH, L_THIGH), (R_SHIN, L_SHIN), (R_FOOT, L_FOOT)]:
            r_q = rotation[i, r_j]
            l_q = rotation[i, l_j]
            # Compare accounting for sign ambiguity
            d1 = np.sum((r_q - l_q) ** 2)
            d2 = np.sum((r_q + l_q) ** 2)
            asym += min(d1, d2)
        if asym < best_asym:
            best_asym = asym
            best_frame = i

    print(f"Most symmetric frame: {best_frame} (asymmetry={best_asym:.4f})")
    print(f"  root_z = {root_trans[best_frame, 2]:.4f}")

    # --- Build symmetric standing pose from this frame ---
    frame_rot = rotation[best_frame].copy()  # (15, 4)

    # Average left/right leg quaternions for perfect symmetry
    for r_j, l_j in [(R_THIGH, L_THIGH), (R_SHIN, L_SHIN), (R_FOOT, L_FOOT)]:
        avg_q = quat_slerp(frame_rot[r_j], frame_rot[l_j], 0.5)
        avg_q = avg_q / np.linalg.norm(avg_q)
        frame_rot[r_j] = avg_q.astype(np.float32)
        frame_rot[l_j] = avg_q.astype(np.float32)

    # Zero out pelvis yaw/roll (keep only slight pitch if any)
    frame_rot[PELVIS] = IDENTITY_Q.copy()

    # Arms at identity (natural rest pose hanging down)
    for j in [R_UPPER_ARM, R_LOWER_ARM, R_HAND, L_UPPER_ARM, L_LOWER_ARM, L_HAND]:
        frame_rot[j] = IDENTITY_Q.copy()

    # Torso and head at identity
    frame_rot[TORSO] = IDENTITY_Q.copy()
    frame_rot[HEAD] = IDENTITY_Q.copy()

    # Use the frame's root_z (from a natural walking double-support phase)
    root_z = root_trans[best_frame, 2]
    print(f"  Using root_z = {root_z:.4f}")
    print(f"  Right shin quat: {frame_rot[R_SHIN]}")
    print(f"  Left shin quat:  {frame_rot[L_SHIN]}")
    print(f"  Right thigh quat: {frame_rot[R_THIGH]}")

    # Tile to NUM_FRAMES
    stand_rot = np.tile(frame_rot.reshape(1, NUM_JOINTS, 4), (NUM_FRAMES, 1, 1))
    stand_root = np.zeros((NUM_FRAMES, 3), dtype=np.float32)
    stand_root[:, 2] = root_z
    stand_vel = np.zeros((NUM_FRAMES, NUM_JOINTS, 3), dtype=np.float32)
    stand_ang_vel = np.zeros((NUM_FRAMES, NUM_JOINTS, 3), dtype=np.float32)

    # Build SkeletonMotion dict
    stand_data = OrderedDict()
    stand_data['rotation'] = {
        'arr': stand_rot.astype(np.float32),
        'context': {'dtype': 'float32'},
    }
    stand_data['root_translation'] = {
        'arr': stand_root,
        'context': {'dtype': 'float32'},
    }
    stand_data['global_velocity'] = {
        'arr': stand_vel,
        'context': {'dtype': 'float32'},
    }
    stand_data['global_angular_velocity'] = {
        'arr': stand_ang_vel,
        'context': {'dtype': 'float32'},
    }
    stand_data['skeleton_tree'] = copy.deepcopy(walk_data['skeleton_tree'])
    stand_data['is_local'] = walk_data['is_local']
    stand_data['fps'] = walk_data['fps']
    stand_data['__name__'] = 'SkeletonMotion'

    output_path = os.path.join(MOTIONS_DIR, 'amp_humanoid_stand.npy')
    np.save(output_path, stand_data)

    print(f"\n✅ Generated symmetric standing motion: {output_path}")
    print(f"   Frames: {NUM_FRAMES} ({NUM_FRAMES / walk_data['fps']:.2f}s)")
    print(f"   Source: walk frame {best_frame} (symmetrized)")
    print(f"   Root Z: {root_z:.4f}")
    print(f"   All velocities: 0 (standing still)")


if __name__ == '__main__':
    create_stand_motion()
