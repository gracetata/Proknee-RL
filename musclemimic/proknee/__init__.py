"""MuscleMimic-ProKnee integration utilities.

This package contains a minimally invasive Teacher-Student scaffold that uses a
frozen MuscleMimic full-body policy as the Stage-0 oracle and trains a left-leg
prosthesis controller on top of MyoFullBody.
"""

from .constants import (
    LEFT_ANKLE_JOINT_NAMES,
    LEFT_KNEE_ANKLE_CONFLICT_MUSCLE_NAMES,
    LEFT_KNEE_JOINT_NAMES,
    LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES,
    LEFT_PROSTHESIS_JOINT_NAMES,
    LEFT_PROSTHESIS_SITE_NAMES,
    audit_myofullbody_left_leg,
)
from .hybrid_env import MuscleProKneeHybridEnv
from .action_env import MuscleProKneeCoupledActionEnv
from .oracle import FrozenMMOracle

__all__ = [
    "FrozenMMOracle",
    "MuscleProKneeHybridEnv",
    "MuscleProKneeCoupledActionEnv",
    "LEFT_KNEE_JOINT_NAMES",
    "LEFT_ANKLE_JOINT_NAMES",
    "LEFT_PROSTHESIS_JOINT_NAMES",
    "LEFT_PROSTHESIS_SITE_NAMES",
    "LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES",
    "LEFT_KNEE_ANKLE_CONFLICT_MUSCLE_NAMES",
    "audit_myofullbody_left_leg",
]
