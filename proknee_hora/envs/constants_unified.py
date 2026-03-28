"""Unified velocity-conditioned constants for ProKnee-HoraStyle.

Extends the base constants for the unified velocity-controlled policy (Phase 11+).
The original constants.py is NOT modified — all single-motion pipelines remain intact.

Unified mode: Single body policy with continuous velocity command (0~2.5 m/s)
  - v=0.0 → Stand
  - v=1.0 → Walk
  - v=2.5 → Run
  - Intermediate values → Natural transition

Observation modes:
  - obs = 16D proprio (same as single-motion)
  - priv_info = 113D + 1D velocity_cmd = 114D
  - Student infers velocity from proprio_hist (30×16D)
"""

from .constants import (
    # Re-export everything from single-motion constants
    NUM_DOFS,
    ACTIVE_PROSTHESIS_JOINTS,
    PASSIVE_PROSTHESIS_JOINTS,
    LEFT_HIP_JOINTS,
    FROZEN_BODY_JOINTS,
    LEFT_KNEE,
    LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y,
    LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z,
    STUDENT_PROPRIO_INDICES,
    TEACHER_PRIV_INFO_DIM as BASE_PRIV_INFO_DIM,  # 113
    LATENT_DIM,
    PROPRIO_HISTORY_LEN,
    PROSTHESIS_ACTION_DIM,
    OBS_DIM,
)

# ── Unified velocity command configuration ────────────────────────────
VELOCITY_CMD_DIM = 1  # Single scalar velocity command

# Velocity range
VELOCITY_MIN = 0.0   # Stand
VELOCITY_MAX = 2.5   # Run

# Discrete velocity levels for keyboard control
VELOCITY_STAND = 0.0
VELOCITY_WALK = 1.0
VELOCITY_RUN = 2.5

VELOCITY_LEVELS = [VELOCITY_STAND, VELOCITY_WALK, VELOCITY_RUN]

# ── Observation dimensions ────────────────────────────────────────────
# obs = 16D proprio (same as single-motion, no velocity command for student)
OBS_DIM_UNIFIED = OBS_DIM  # 16

# priv_info = 113D base + 1D velocity_cmd = 114D
TEACHER_PRIV_INFO_DIM_UNIFIED = BASE_PRIV_INFO_DIM + VELOCITY_CMD_DIM  # 114

# proprio_dim for student history buffer
STUDENT_PROPRIO_DIM_UNIFIED = OBS_DIM  # 16

# ── Unified Stage 0 body policy configuration ─────────────────────────
# The unified body policy takes 106D input: 105D humanoid_obs + 1D velocity_cmd
UNIFIED_BODY_OBS_DIM = 105 + VELOCITY_CMD_DIM  # 106

# Default checkpoint path (relative to outputs/checkpoints/stage0/)
UNIFIED_BODY_CHECKPOINT = 'stage0_unified_1800.pth'

# ── Velocity command dynamics ─────────────────────────────────────────
# How often to randomly switch velocity during training
VELOCITY_SWITCH_PROB = 0.005  # Per step probability
VELOCITY_SWITCH_INTERVAL = 100  # Minimum steps between switches

# Velocity sampling distribution during training
VELOCITY_SAMPLE_WEIGHTS = {
    0.0: 0.2,   # 20% Stand
    1.0: 0.5,   # 50% Walk
    2.5: 0.3,   # 30% Run
}

# ── Reward configuration ──────────────────────────────────────────────
# Velocity tracking reward: exp(-w * (v_actual - v_cmd)^2)
VELOCITY_REWARD_WEIGHT = 2.0

# Height/uprightness rewards (same as base)
UP_REWARD_WEIGHT = 1.0
HEIGHT_REWARD_WEIGHT = 0.5
LATERAL_PENALTY_WEIGHT = 0.3
