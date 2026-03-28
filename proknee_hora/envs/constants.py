"""Joint-index constants matching the amp_humanoid.xml actuator order.

Left leg is the prosthetic side.
"""

# ── DOF mapping from amp_humanoid.xml actuator order ──────────────────
# Torso
ABDOMEN_X = 0
ABDOMEN_Y = 1
ABDOMEN_Z = 2

# Neck
NECK_X = 3
NECK_Y = 4
NECK_Z = 5

# Right arm
RIGHT_SHOULDER_X = 6
RIGHT_SHOULDER_Y = 7
RIGHT_SHOULDER_Z = 8
RIGHT_ELBOW = 9

# Left arm
LEFT_SHOULDER_X = 10
LEFT_SHOULDER_Y = 11
LEFT_SHOULDER_Z = 12
LEFT_ELBOW = 13

# Right leg (healthy side)
RIGHT_HIP_X = 14
RIGHT_HIP_Z = 15
RIGHT_HIP_Y = 16
RIGHT_KNEE = 17
RIGHT_ANKLE_X = 18
RIGHT_ANKLE_Y = 19
RIGHT_ANKLE_Z = 20

# Left leg (PROSTHETIC side)
LEFT_HIP_X = 21
LEFT_HIP_Z = 22
LEFT_HIP_Y = 23
LEFT_KNEE = 24           # ACTIVE PROSTHESIS – to be trained
LEFT_ANKLE_X = 25        # PASSIVE – fixed
LEFT_ANKLE_Y = 26        # PASSIVE – fixed
LEFT_ANKLE_Z = 27        # PASSIVE – fixed

NUM_DOFS = 28

# ── Joint classification for Teacher-Student training ─────────────────
ACTIVE_PROSTHESIS_JOINTS = [LEFT_KNEE, LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z]  # 4 DOFs — knee + ankle
PASSIVE_PROSTHESIS_JOINTS = []  # Nothing locked — ankle is actively controlled
LEFT_HIP_JOINTS = [LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y]
PROSTHESIS_ACTION_DIM = len(ACTIVE_PROSTHESIS_JOINTS)  # 4

# Body joints controlled by frozen Stage-0 policy (all except active prosthesis)
FROZEN_BODY_JOINTS = sorted(
    set(range(NUM_DOFS)) - set(ACTIVE_PROSTHESIS_JOINTS) - set(PASSIVE_PROSTHESIS_JOINTS)
)  # 24 DOFs

# ── Student proprioception ────────────────────────────────────────────
STUDENT_PROPRIO_INDICES = {
    'knee_pos': LEFT_KNEE,                                  # 1D
    'knee_vel': LEFT_KNEE,                                  # 1D
    'ankle_pos': [LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z],  # 3D
    'ankle_vel': [LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z],  # 3D
    'hip_pos': [LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y],      # 3D
    'hip_vel': [LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y],      # 3D
    # foot_force_z: from contact sensor (1D)
    # command: target velocity (1D)
}

STUDENT_PROPRIO_DIM = 16   # knee(2) + ankle(6) + hip(6) + foot_fz(1) + command(1)
# Hora-faithful: obs = proprio (deployment-available), priv_info = sim-only
OBS_DIM = STUDENT_PROPRIO_DIM  # 16D — obs fed to backbone (same as proprio_current)
TEACHER_PRIV_INFO_DIM = 113   # full_body_obs(105) + GRF(6) + contacts(2)
LATENT_DIM = 32
PROPRIO_HISTORY_LEN = 30   # number of history frames for student TConv
