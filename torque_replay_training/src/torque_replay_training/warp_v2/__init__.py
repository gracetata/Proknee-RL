"""MuJoCo Warp flat-walk prosthesis training.

The package is deliberately separate from the legacy CPU residual environment.
Importing it does not initialize CUDA; MuJoCo Warp is imported lazily by
``WarpReplayEnv``.
"""

from .config import (
    PPOConfig,
    ReplayAnnealingConfig,
    RewardConfig,
    WarpEnvironmentConfig,
    WarpTrainingConfig,
    load_training_config,
)
from .curriculum import replay_beta
from .rewards import RewardInputs, compute_reward

__all__ = [
    "PPOConfig",
    "ReplayAnnealingConfig",
    "RewardConfig",
    "RewardInputs",
    "WarpEnvironmentConfig",
    "WarpTrainingConfig",
    "compute_reward",
    "load_training_config",
    "replay_beta",
]
