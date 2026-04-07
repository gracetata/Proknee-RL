"""Unified velocity-conditioned constants for ProKnee-HoraStyle.

Extends the base constants for the unified velocity-controlled policy (Phase 11+).
The original constants.py is NOT modified — all single-motion pipelines remain intact.

Unified mode: Single body policy with continuous velocity command
  - 与 HumanMimic Stage0（velocityMax=3.0）对齐时，命令上界为 VELOCITY_MAX（默认 3.0 m/s）
  - v=0.0 → Stand, v=1.0 → Walk, v=2.5 → Run；亦可采样至 3.0（快跑上限）

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

# Velocity range（与 IsaacGym HumanoidAMPUnifiedHumanMimic_phase1 的 velocityMax 一致）
VELOCITY_MIN = 0.0   # Stand
VELOCITY_MAX = 3.0   # 上限 m/s（原 Phase11 为 2.5；HumanMimic Stage0 为 0~3）

# Discrete velocity levels for env randomization / TB（Stage1/2 训练时在各档间均匀采样）
VELOCITY_STAND = 0.0
VELOCITY_WALK = 1.0
VELOCITY_RUN = 2.5

VELOCITY_LEVELS = [VELOCITY_STAND, VELOCITY_WALK, VELOCITY_RUN, VELOCITY_MAX]

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

# 与 Stage1/2 脚本默认一致：仓库根下 outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth
UNIFIED_BODY_CHECKPOINT = 'HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth'

# ── Velocity command dynamics ─────────────────────────────────────────
# How often to randomly switch velocity during training
VELOCITY_SWITCH_PROB = 0.005  # Per step probability
VELOCITY_SWITCH_INTERVAL = 100  # Minimum steps between switches

# Velocity sampling distribution during training
VELOCITY_SAMPLE_WEIGHTS = {
    0.0: 0.2,
    1.0: 0.4,
    2.5: 0.25,
    3.0: 0.15,
}

# ── Reward configuration ──────────────────────────────────────────────
# Velocity tracking reward: exp(-w * (v_actual - v_cmd)^2)
VELOCITY_REWARD_WEIGHT = 2.0

# Height/uprightness rewards (same as base)
UP_REWARD_WEIGHT = 1.0
HEIGHT_REWARD_WEIGHT = 0.5
LATERAL_PENALTY_WEIGHT = 0.3
