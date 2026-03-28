#!/bin/bash
# Evaluate trained policy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/proknee_hora"

CHECKPOINT="${1:-$PROJECT_DIR/outputs/stage2_student/checkpoints/best.pth}"
MODE="${2:-student}"  # teacher or student
BODY_POLICY="$PROJECT_DIR/ProKnee-Simulator/runs/HumanoidAMP_stable_06-18-14-00/nn/HumanoidAMP_stable_06-18-14-01_5000.pth"
NUM_ENVS="${3:-1}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "Evaluating Policy"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Mode: $MODE"
echo "Num envs: $NUM_ENVS"

cd "$PROJECT_DIR"

python -c "
import sys, os, numpy as np
if not hasattr(np, 'float'):
    np.float = float; np.int = int; np.bool = bool
from isaacgym import gymapi
import torch

sys.path.insert(0, '$PROJECT_DIR')
from proknee_hora.envs import ProKneeStudent, ProKneeTeacher
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.envs.constants import (
    ACTIVE_PROSTHESIS_JOINTS, STUDENT_PROPRIO_DIM, TEACHER_PRIV_INFO_DIM,
)

DEVICE = 'cuda:0'
ACTION_DIM = len(ACTIVE_PROSTHESIS_JOINTS)  # 4

# Load checkpoint
ckpt = torch.load('$CHECKPOINT', map_location=DEVICE, weights_only=False)

# Create policy
policy = ProKneePolicy(
    obs_dim=105,
    action_dim=ACTION_DIM,
    priv_info_dim=TEACHER_PRIV_INFO_DIM,
    proprio_dim=STUDENT_PROPRIO_DIM,
    history_len=30,
    latent_dim=8,
    hidden_dims=[256, 128, 64],
    mode='$MODE',
).to(DEVICE)

# Load weights
if 'student_state_dict' in ckpt:
    policy.load_state_dict(ckpt['student_state_dict'])
elif 'policy_state_dict' in ckpt:
    policy.load_state_dict(ckpt['policy_state_dict'])
policy.eval()
print(f'Loaded policy ({sum(p.numel() for p in policy.parameters()):,} params)')

# Create environment with visualization
if '$MODE' == 'student':
    env = ProKneeStudent(
        num_envs=int($NUM_ENVS), device=DEVICE, headless=False,
        body_policy_checkpoint='$BODY_POLICY',
        episode_length=1000,
    )
else:
    env = ProKneeTeacher(
        num_envs=int($NUM_ENVS), device=DEVICE, headless=False,
        body_policy_checkpoint='$BODY_POLICY',
        episode_length=1000,
    )

# Evaluate loop
obs = env.reset()
total_reward = 0.0
steps = 0
episodes = 0

print('Starting evaluation (close viewer window or Ctrl+C to exit)...')
try:
    while True:
        with torch.no_grad():
            if '$MODE' == 'student':
                output = policy(obs['obs'], proprio_hist=obs.get('proprio_hist'))
            else:
                output = policy(obs['obs'], priv_info=obs.get('priv_info'))
            action = output['action_mean']

        obs, reward, done, info = env.step(action)
        total_reward += reward.sum().item()
        steps += 1

        if done.any():
            episodes += 1
            avg_r = total_reward / max(episodes, 1)
            print(f'  Episode {episodes} done at step {steps}, avg reward: {avg_r:.2f}')

except KeyboardInterrupt:
    pass

print(f'Total episodes: {episodes}, steps: {steps}, avg reward: {total_reward / max(episodes,1):.2f}')
env.close()
"

echo "Evaluation complete!"
