"""Constants and model-audit helpers for the MuscleMimic-ProKnee bridge."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


# The first implementation follows the ProKnee 4-DOF control semantics:
# knee flexion + ankle/subtalar/mtp. MyoFullBody has additional knee coupling
# DOFs; those remain under the frozen full-body oracle in the initial prototype.
LEFT_KNEE_JOINT_NAMES = ("knee_angle_l",)
LEFT_ANKLE_JOINT_NAMES = ("ankle_angle_l", "subtalar_angle_l", "mtp_angle_l")
LEFT_PROSTHESIS_JOINT_NAMES = LEFT_KNEE_JOINT_NAMES + LEFT_ANKLE_JOINT_NAMES

LEFT_PROSTHESIS_SITE_NAMES = (
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
)

# Full left lower-limb muscle set. This is not the first action target, but we
# keep it explicit for the later high-fidelity muscle/torque variants.
LEFT_LOWER_LIMB_MUSCLE_NAMES = (
    "addbrev_l",
    "addlong_l",
    "addmagDist_l",
    "addmagIsch_l",
    "addmagMid_l",
    "addmagProx_l",
    "bflh_l",
    "bfsh_l",
    "edl_l",
    "ehl_l",
    "fdl_l",
    "fhl_l",
    "gaslat_l",
    "gasmed_l",
    "glmax1_l",
    "glmax2_l",
    "glmax3_l",
    "glmed1_l",
    "glmed2_l",
    "glmed3_l",
    "glmin1_l",
    "glmin2_l",
    "glmin3_l",
    "grac_l",
    "iliacus_l",
    "perbrev_l",
    "perlong_l",
    "piri_l",
    "psoas_l",
    "recfem_l",
    "sart_l",
    "semimem_l",
    "semiten_l",
    "soleus_l",
    "tfl_l",
    "tibant_l",
    "tibpost_l",
    "vasint_l",
    "vaslat_l",
    "vasmed_l",
)

# Muscles that directly cross the current prosthesis boundary (left knee/ankle/toes).
# Used for curriculum "prosthesis_muscle_scale": scale these actuators toward 0 while
# Stage1 PD motors take over the 4 prosthesis DOFs.
# Preferred alias (same tuple): prosthesis-boundary muscles for motor-replacement schedule.
LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES = (
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

# Backward-compatible name used in earlier ProKnee bridge code / checkpoints docs.
LEFT_KNEE_ANKLE_CONFLICT_MUSCLE_NAMES = LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES


@dataclass(frozen=True)
class JointAudit:
    name: str
    joint_id: int
    qposadr: int
    dofadr: int
    joint_range: tuple[float, float]


@dataclass(frozen=True)
class ActuatorAudit:
    name: str
    actuator_id: int
    ctrlrange: tuple[float, float]
    dyntype: int


@dataclass(frozen=True)
class LeftLegAudit:
    joints: tuple[JointAudit, ...]
    sites: tuple[tuple[str, int], ...]
    actuators: tuple[ActuatorAudit, ...]

    @property
    def qpos_indices(self) -> np.ndarray:
        return np.asarray([j.qposadr for j in self.joints], dtype=np.int32)

    @property
    def qvel_indices(self) -> np.ndarray:
        return np.asarray([j.dofadr for j in self.joints], dtype=np.int32)

    @property
    def actuator_indices(self) -> np.ndarray:
        return np.asarray([a.actuator_id for a in self.actuators], dtype=np.int32)


def _id(model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return int(idx)


def audit_myofullbody_left_leg(model) -> LeftLegAudit:
    """Return the MyoFullBody left prosthesis boundary used by Stage 1/2."""

    joints: list[JointAudit] = []
    for name in LEFT_PROSTHESIS_JOINT_NAMES:
        jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joints.append(
            JointAudit(
                name=name,
                joint_id=jid,
                qposadr=int(model.jnt_qposadr[jid]),
                dofadr=int(model.jnt_dofadr[jid]),
                joint_range=tuple(float(x) for x in model.jnt_range[jid]),
            )
        )

    sites: list[tuple[str, int]] = []
    for name in LEFT_PROSTHESIS_SITE_NAMES:
        sid = _id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        sites.append((name, sid))

    actuators: list[ActuatorAudit] = []
    for name in LEFT_LOWER_LIMB_MUSCLE_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            actuators.append(
                ActuatorAudit(
                    name=name,
                    actuator_id=int(aid),
                    ctrlrange=tuple(float(x) for x in model.actuator_ctrlrange[aid]),
                    dyntype=int(model.actuator_dyntype[aid]),
                )
            )

    return LeftLegAudit(joints=tuple(joints), sites=tuple(sites), actuators=tuple(actuators))
