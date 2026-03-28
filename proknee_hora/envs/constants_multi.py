"""Multi-motion constants for ProKnee-HoraStyle.

Extends the single-motion constants (constants.py) with multi-motion support.
The original constants.py is NOT modified — all walk-only pipelines remain intact.

Motion types: WALK, RUN, DANCE, STAND

Two observation modes:
  motion_in_obs=True  (legacy): obs=20D (16D proprio + 4D onehot), priv=113D
  motion_in_obs=False (realistic): obs=16D (pure proprio), priv=117D (113D + 4D onehot)

Realistic mode: prosthesis has NO task command — it infers motion type
from proprioceptive history (as a real prosthesis would).
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
    TEACHER_PRIV_INFO_DIM,      # 113 (base, no motion onehot)
    LATENT_DIM,
    PROPRIO_HISTORY_LEN,
    PROSTHESIS_ACTION_DIM,
    OBS_DIM as SINGLE_OBS_DIM,  # 16 (single-motion obs)
)

# ── Motion type definitions ──────────────────────────────────────────
MOTION_WALK  = 0
MOTION_RUN   = 1
MOTION_DANCE = 2
MOTION_STAND = 3

NUM_MOTION_TYPES = 4

MOTION_NAMES = {
    MOTION_WALK:  'walk',
    MOTION_RUN:   'run',
    MOTION_DANCE: 'dance',
    MOTION_STAND: 'stand',
}

# Default motion file paths (relative to IsaacGymEnvs/assets/amp/motions/)
MOTION_FILES = {
    MOTION_WALK:  'amp_humanoid_walk.npy',
    MOTION_RUN:   'amp_humanoid_run.npy',
    MOTION_DANCE: 'amp_humanoid_dance.npy',
    MOTION_STAND: 'amp_humanoid_stand.npy',
}

# Default Stage 0 checkpoint names (relative to outputs/checkpoints/stage0/)
MOTION_STAGE0_CHECKPOINTS = {
    MOTION_WALK:  'stage0_amp_walk_5050.pth',
    MOTION_RUN:   'stage0_amp_run_1700.pth',
    MOTION_DANCE: 'stage0_amp_dance_8000.pth',
    MOTION_STAND: 'stage0_amp_stand_900.pth',
}

# ── Multi-motion observation dimensions ──────────────────────────────
SINGLE_PROPRIO_DIM = 16            # Original 16D proprio (unchanged)
MOTION_ENCODING_DIM = NUM_MOTION_TYPES  # 4D one-hot

# Legacy mode (motion_in_obs=True): prosthesis receives motion command
STUDENT_PROPRIO_DIM_MULTI = SINGLE_PROPRIO_DIM + MOTION_ENCODING_DIM  # 20
OBS_DIM_MULTI = STUDENT_PROPRIO_DIM_MULTI  # 20

# Realistic mode (motion_in_obs=False): prosthesis has NO motion command
# obs = 16D pure proprio (same as single-motion)
# priv_info = 113D + 4D motion_onehot = 117D
STUDENT_PROPRIO_DIM_REALISTIC = SINGLE_PROPRIO_DIM  # 16
OBS_DIM_REALISTIC = SINGLE_PROPRIO_DIM               # 16
TEACHER_PRIV_INFO_DIM_MULTI = TEACHER_PRIV_INFO_DIM + MOTION_ENCODING_DIM  # 117

# ── Target velocities per motion (for reward computation) ────────────
MOTION_TARGET_VELOCITIES = {
    MOTION_WALK:  1.0,   # m/s
    MOTION_RUN:   2.5,   # m/s
    MOTION_DANCE: 0.0,   # stationary dance
    MOTION_STAND: 0.0,   # standing still
}

# ── Reward weights per motion (override base reward structure) ────────
MOTION_REWARD_WEIGHTS = {
    MOTION_WALK: {
        'vel_reward': 3.0,
        'up_reward': 1.0,
        'height_reward': 0.5,
        'lateral_penalty': 0.3,
    },
    MOTION_RUN: {
        'vel_reward': 3.0,
        'up_reward': 1.5,
        'height_reward': 0.5,
        'lateral_penalty': 0.2,
    },
    MOTION_DANCE: {
        'vel_reward': 0.5,   # Less emphasis on velocity for dance
        'up_reward': 2.0,    # Stay upright is critical
        'height_reward': 1.0,
        'lateral_penalty': 0.1,
    },
    MOTION_STAND: {
        'vel_reward': 0.5,   # Penalize movement
        'up_reward': 2.0,    # Stay upright
        'height_reward': 1.5,
        'lateral_penalty': 0.5,
    },
}

# ── Motion switching configuration ───────────────────────────────────
# For episode-level motion assignment (Phase 2-3)
MOTION_SWITCH_INTERVAL_RANGE = (50, 150)  # steps between switches (Phase 4)
