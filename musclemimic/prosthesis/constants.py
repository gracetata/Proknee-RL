"""Shared constants for the MyoFullBody prosthesis environment."""

from __future__ import annotations

PROSTHESIS_JOINT_NAMES = (
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
)

PROSTHESIS_MOTOR_NAMES = (
    "prosthesis_motor_knee_l",
    "prosthesis_motor_ankle_l",
    "prosthesis_motor_subtalar_l",
    "prosthesis_motor_mtp_l",
)

DEFAULT_PROSTHESIS_TORQUE_LIMITS = (120.0, 120.0, 30.0, 20.0)

# Conservative first-pass control-level amputation mask. This is the previously
# audited left knee/ankle/foot boundary set; mapping.py can extend/override it.
DEFAULT_DISABLED_MUSCLE_NAMES = (
    "bflh_l",
    "bfsh_l",
    "semimem_l",
    "semiten_l",
    "recfem_l",
    "vasint_l",
    "vaslat_l",
    "vasmed_l",
    "gaslat_l",
    "gasmed_l",
    "soleus_l",
    "tibant_l",
    "tibpost_l",
    "perbrev_l",
    "perlong_l",
    "edl_l",
    "ehl_l",
    "fdl_l",
    "fhl_l",
)

FOOT1_DISABLED_MUSCLE_NAMES = (
    "fdl_l",
)

# Toe + lateral foot extrinsics only (below ankle); thigh/calf muscles remain active.
FOOT5_DISABLED_MUSCLE_NAMES = (
    "edl_l",
    "ehl_l",
    "fdl_l",
    "fhl_l",
    "perbrev_l",
)

DISTAL11_DISABLED_MUSCLE_NAMES = (
    "gaslat_l",
    "gasmed_l",
    "soleus_l",
    "tibant_l",
    "tibpost_l",
    "perbrev_l",
    "perlong_l",
    "edl_l",
    "ehl_l",
    "fdl_l",
    "fhl_l",
)

KNEE15_DISABLED_MUSCLE_NAMES = (
    "bfsh_l",
    "vasint_l",
    "vaslat_l",
    "vasmed_l",
    *DISTAL11_DISABLED_MUSCLE_NAMES,
)

FULL_REPLAY_DISABLED_MUSCLE_NAMES: tuple[str, ...] = ()

MUSCLE_MASK_PRESETS = {
    "full": FULL_REPLAY_DISABLED_MUSCLE_NAMES,
    "foot1": FOOT1_DISABLED_MUSCLE_NAMES,
    "foot5": FOOT5_DISABLED_MUSCLE_NAMES,
    "distal11": DISTAL11_DISABLED_MUSCLE_NAMES,
    "knee15": KNEE15_DISABLED_MUSCLE_NAMES,
    "strict19": DEFAULT_DISABLED_MUSCLE_NAMES,
}

# Default prosthesis distillation still uses strict19 unless a preset overrides it.
DEFAULT_PROSTHESIS_DISTILL_MASK_PRESET = "strict19"


def disabled_muscle_names_for_preset(
    preset: str,
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Resolve disabled muscle names from a named preset plus optional include/exclude."""
    key = str(preset).strip().lower()
    if key not in MUSCLE_MASK_PRESETS:
        raise ValueError(f"Unknown muscle mask preset {preset!r}; choices={sorted(MUSCLE_MASK_PRESETS)}")
    base = set(MUSCLE_MASK_PRESETS[key])
    base |= set(include)
    base -= set(exclude)
    return tuple(sorted(base))

# Default for OSL FSM deployment after KIT/425 joint-replay calibration.
DEFAULT_OSL_MASK_PRESET = "knee15"

LEFT_LEG_NAME_KEYWORDS = (
    "_l",
    "left",
    "lf",
)

LEFT_DISTAL_PATH_KEYWORDS = (
    "knee",
    "ankle",
    "foot",
    "toe",
    "tibia",
    "fibula",
    "calcaneus",
    "calc",
    "talus",
    "metatarsal",
    "mtp",
)
