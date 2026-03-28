"""Environment modules for ProKnee-Hora.

Joint indices follow the MJCF actuator order in amp_humanoid.xml.
Left leg is the prosthetic side.
"""

# Constants first (no Isaac Gym / torch dependency)
from .constants import (                     # noqa: F401
    ABDOMEN_X, ABDOMEN_Y, ABDOMEN_Z,
    NECK_X, NECK_Y, NECK_Z,
    RIGHT_SHOULDER_X, RIGHT_SHOULDER_Y, RIGHT_SHOULDER_Z, RIGHT_ELBOW,
    LEFT_SHOULDER_X, LEFT_SHOULDER_Y, LEFT_SHOULDER_Z, LEFT_ELBOW,
    RIGHT_HIP_X, RIGHT_HIP_Z, RIGHT_HIP_Y, RIGHT_KNEE,
    RIGHT_ANKLE_X, RIGHT_ANKLE_Y, RIGHT_ANKLE_Z,
    LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y, LEFT_KNEE,
    LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z,
    NUM_DOFS,
    ACTIVE_PROSTHESIS_JOINTS, PASSIVE_PROSTHESIS_JOINTS,
    LEFT_HIP_JOINTS, FROZEN_BODY_JOINTS,
    STUDENT_PROPRIO_INDICES, STUDENT_PROPRIO_DIM,
    TEACHER_PRIV_INFO_DIM, LATENT_DIM,
)


def __getattr__(name):
    """Lazy import of Isaac-Gym-dependent classes."""
    if name == "ProKneeBase":
        from .proknee_base import ProKneeBase
        return ProKneeBase
    if name == "ProKneeTeacher":
        from .proknee_teacher import ProKneeTeacher
        return ProKneeTeacher
    if name == "ProKneeStudent":
        from .proknee_student import ProKneeStudent
        return ProKneeStudent
    if name == "ProKneeMultiMotionEnv":
        from .proknee_multi_motion import ProKneeMultiMotionEnv
        return ProKneeMultiMotionEnv
    if name == "ProKneeUnifiedEnv":
        from .proknee_unified import ProKneeUnifiedEnv
        return ProKneeUnifiedEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
