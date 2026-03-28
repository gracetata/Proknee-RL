"""
ProKnee Base Environment for Isaac Gym.

Standalone Isaac Gym environment for prosthetic knee training.
Creates the physics simulation directly and implements:
  - Frozen body policy (Stage 0 AMP checkpoint)
  - 1-DOF active prosthesis control (left knee)
  - Passive ankle locking (left ankle)
  - Observation / reward / termination logic
"""

import os
import sys
import math
import numpy as np

# Fix numpy deprecations before isaacgym imports (isaacgym uses np.float)
if not hasattr(np, 'float'):
    np.float = float      # type: ignore[attr-defined]
    np.int = int          # type: ignore[attr-defined]
    np.bool = bool        # type: ignore[attr-defined]

# Isaac Gym MUST be imported before torch
try:
    from isaacgym import gymapi, gymtorch, gymutil
    from isaacgym.torch_utils import to_torch, quat_rotate
    ISAACGYM_AVAILABLE = True
except (ImportError, AttributeError) as _e:
    ISAACGYM_AVAILABLE = False
    _ISAACGYM_ERR = str(_e)

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# ── Import observation/quaternion utilities from IsaacGymEnvs ──────────
ISAAC_GYM_ENVS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'IsaacGymEnvs',
)
sys.path.insert(0, ISAAC_GYM_ENVS_PATH)
sys.path.insert(0, os.path.join(ISAAC_GYM_ENVS_PATH, 'isaacgymenvs'))

# Import the exact JIT functions used during Stage-0 training
from isaacgymenvs.utils.torch_jit_utils import (
    exp_map_to_quat,
    quat_to_tan_norm,
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_from_angle_axis,
    normalize,
    quat_unit,
    normalize_angle,
)
# Import the full observation builder
from isaacgymenvs.tasks.amp.humanoid_amp_base import (
    compute_humanoid_observations,
    dof_to_obs,
)
# Import motion library for reference state initialisation
from tasks.amp.utils_amp.motion_lib import MotionLib

from .constants import (
    NUM_DOFS,
    ACTIVE_PROSTHESIS_JOINTS,
    PASSIVE_PROSTHESIS_JOINTS,
    LEFT_HIP_JOINTS,
    FROZEN_BODY_JOINTS,
    LEFT_KNEE,
    LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y,
    LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z,
    STUDENT_PROPRIO_DIM,
    TEACHER_PRIV_INFO_DIM,
    OBS_DIM,
)

# ── Observation constants (same as HumanoidAMPBase) ───────────────────
DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 105  # 13 + 52 + 28 + 12
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]


# ── Frozen body policy (matches Stage-0 AMP network) ─────────────────
class FrozenBodyPolicy(nn.Module):
    """Reconstructs the Stage-0 AMP actor network to load checkpoint."""

    def __init__(self, obs_dim: int = 105, action_dim: int = 28):
        super().__init__()
        # Architecture must match: actor_mlp [1024, 512] + mu head
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.mu = nn.Linear(512, action_dim)

        # Running mean/std for observation normalisation
        self.running_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(obs_dim), requires_grad=False)

    def normalise_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.running_mean) / torch.sqrt(self.running_var + 1e-5)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (28-D)."""
        x = self.normalise_obs(obs)
        x = self.actor_mlp(x)
        return self.mu(x)


def load_body_policy(checkpoint_path: str, device: str) -> FrozenBodyPolicy:
    """Load Stage-0 AMP checkpoint into FrozenBodyPolicy."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_sd = ckpt['model']

    policy = FrozenBodyPolicy().to(device)

    # Map rl_games keys → our keys
    mapping = {
        'a2c_network.actor_mlp.0.weight': 'actor_mlp.0.weight',
        'a2c_network.actor_mlp.0.bias':   'actor_mlp.0.bias',
        'a2c_network.actor_mlp.2.weight': 'actor_mlp.2.weight',
        'a2c_network.actor_mlp.2.bias':   'actor_mlp.2.bias',
        'a2c_network.mu.weight':          'mu.weight',
        'a2c_network.mu.bias':            'mu.bias',
        'running_mean_std.running_mean':  'running_mean',
        'running_mean_std.running_var':   'running_var',
    }
    new_sd = {}
    for src, dst in mapping.items():
        if src in model_sd:
            new_sd[dst] = model_sd[src]
    policy.load_state_dict(new_sd, strict=False)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy


# (JIT helpers are now imported from ProKnee-Simulator, see top of file)


# ── Main environment ──────────────────────────────────────────────────
class ProKneeBase:
    """Isaac Gym environment for prosthetic knee Teacher-Student training.

    Creates physics simulation, loads frozen body policy, and provides
    observation / reward / termination logic.
    """

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda:0',
        headless: bool = True,
        # Prosthesis configuration
        prosthesis_only: bool = False,
        freeze_body: bool = True,
        body_policy_checkpoint: Optional[str] = None,
        # History buffer
        proprio_hist_len: int = 30,
        # Simulation
        control_freq_inv: int = 2,
        dt: float = 1.0 / 180.0,
        episode_length: int = 300,
        target_velocity: float = 1.0,
        motion_file: Optional[str] = None,
    ):
        if not ISAACGYM_AVAILABLE:
            raise RuntimeError("Isaac Gym is not available in this environment")

        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.prosthesis_only = prosthesis_only
        self.freeze_body = freeze_body
        self.proprio_hist_len = proprio_hist_len
        self.control_freq_inv = control_freq_inv
        self.sim_dt = dt
        self.dt = control_freq_inv * dt
        self.max_episode_length = episode_length
        self.target_velocity = target_velocity

        # Dimensions
        self.num_dofs = NUM_DOFS
        self.base_obs_dim = OBS_DIM  # 16D proprio (deployment-available)
        self._full_body_obs_dim = NUM_OBS  # 105D full body obs (for priv_info)
        self.priv_info_dim = TEACHER_PRIV_INFO_DIM  # 113D expanded
        self.proprio_dim = STUDENT_PROPRIO_DIM

        if prosthesis_only:
            self.num_actions = len(ACTIVE_PROSTHESIS_JOINTS)
        else:
            self.num_actions = NUM_DOFS
        self.num_obs = self.base_obs_dim

        # Create Isaac Gym simulation
        self._local_root_obs = False  # must match Stage-0 training (localRootObs: false)
        self._create_sim()
        self._create_envs()
        self._acquire_tensors()

        # ── Load walking motion for reference state init ──────────────
        if motion_file is None:
            motion_file = os.path.join(ISAAC_GYM_ENVS_PATH,
                                       'assets', 'amp', 'motions',
                                       'amp_humanoid_walk.npy')
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            num_dofs=self.num_dof,
            key_body_ids=torch.tensor(self._key_body_ids).cpu().long(),
            device=self.device,
        )

        # Proprioceptive history buffer
        self.proprio_hist = torch.zeros(
            self.num_envs, self.proprio_hist_len, self.proprio_dim, device=self.device
        )

        # Episode tracking
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Reward tracking
        self._potentials = torch.zeros(self.num_envs, device=self.device)
        self._prev_potentials = torch.zeros_like(self._potentials)

        # Current actions
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # Load frozen body policy
        self.body_policy = None
        if freeze_body and body_policy_checkpoint:
            self._load_body_policy(body_policy_checkpoint)

    # ── Simulation setup ──────────────────────────────────────────────

    def _create_sim(self):
        """Create Isaac Gym simulator."""
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        is_gpu = 'cuda' in self.device

        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1  # TGS
        sim_params.physx.use_gpu = is_gpu
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 5.0
        sim_params.physx.max_gpu_contact_pairs = 8388608

        sim_params.use_gpu_pipeline = is_gpu

        compute_device = int(self.device.split(':')[-1]) if 'cuda' in self.device else 0
        graphics_device = -1 if self.headless else compute_device

        self.sim = self.gym.create_sim(compute_device, graphics_device,
                                       gymapi.SIM_PHYSX, sim_params)

        # Ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Create parallel environments with humanoid actors."""
        spacing = 5.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(self.num_envs))

        asset_root = os.path.join(ISAAC_GYM_ENVS_PATH, 'assets')
        asset_file = 'mjcf/amp_humanoid.xml'

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(self._humanoid_asset)
        motor_efforts = [p.motor_effort for p in actuator_props]
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.num_bodies = self.gym.get_asset_rigid_body_count(self._humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(self._humanoid_asset)

        # Force sensors on feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(self._humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(self._humanoid_asset, left_foot_idx, sensor_pose)

        # Key body indices
        self._key_body_ids = []
        for name in KEY_BODY_NAMES:
            idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, name)
            self._key_body_ids.append(idx)
        self._key_body_ids = to_torch(self._key_body_ids, dtype=torch.long, device=self.device)

        # Contact body indices (feet – allowed to contact ground)
        self._contact_body_ids = []
        for name in ["right_foot", "left_foot"]:
            idx = self.gym.find_asset_rigid_body_index(self._humanoid_asset, name)
            self._contact_body_ids.append(idx)
        self._contact_body_ids = to_torch(self._contact_body_ids, dtype=torch.long, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.89)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.envs = []
        self.humanoid_handles = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            handle = self.gym.create_actor(env_ptr, self._humanoid_asset, start_pose,
                                           "humanoid", i, 0, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            # PD control
            dof_prop = self.gym.get_asset_dof_properties(self._humanoid_asset)
            dof_prop["driveMode"][:] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

            # Color prosthetic limb (left shin + left foot) in red
            if not self.headless:
                for bname in ["left_shin", "left_foot"]:
                    bidx = self.gym.find_actor_rigid_body_index(env_ptr, handle, bname, gymapi.DOMAIN_ACTOR)
                    if bidx >= 0:
                        self.gym.set_rigid_body_color(env_ptr, handle, bidx,
                                                      gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.1, 0.1))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

        # DOF limits
        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        lower_lim = []
        upper_lim = []
        for j in range(self.num_dof):
            lo, hi = dof_prop['lower'][j], dof_prop['upper'][j]
            if lo > hi:
                lo, hi = hi, lo
            lower_lim.append(lo)
            upper_lim.append(hi)
        self.dof_limits_lower = to_torch(lower_lim, device=self.device)
        self.dof_limits_upper = to_torch(upper_lim, device=self.device)

        # PD action offset / scale (must match HumanoidAMPBase._build_pd_action_offset_scale)
        lim_low = np.array(lower_lim, dtype=np.float32)
        lim_high = np.array(upper_lim, dtype=np.float32)
        dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        for j in range(len(dof_offsets) - 1):
            dof_off = dof_offsets[j]
            dof_size = dof_offsets[j + 1] - dof_offsets[j]
            if dof_size == 3:
                lim_low[dof_off:dof_off + dof_size] = -np.pi
                lim_high[dof_off:dof_off + dof_size] = np.pi
            elif dof_size == 1:
                mid = 0.5 * (lim_high[dof_off] + lim_low[dof_off])
                scale = 0.7 * (lim_high[dof_off] - lim_low[dof_off])
                lim_low[dof_off] = mid - scale
                lim_high[dof_off] = mid + scale
        self._pd_action_offset = to_torch(0.5 * (lim_high + lim_low), device=self.device)
        self._pd_action_scale = to_torch(0.5 * (lim_high - lim_low), device=self.device)

        # Initial DOF pose
        self._initial_dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        # Shoulders outward
        right_sh = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_shoulder_x")
        left_sh = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_shoulder_x")
        self._initial_dof_pos[:, right_sh] = 0.5 * np.pi
        self._initial_dof_pos[:, left_sh] = -0.5 * np.pi
        self._initial_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        # Viewer (for non-headless mode)
        self.viewer = None
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Prepare simulation
        self.gym.prepare_sim(self.sim)

    def _acquire_tensors(self):
        """Acquire GPU tensor handles from Isaac Gym."""
        # Do one simulation step to populate state tensors
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        actor_root = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        sensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact = self.gym.acquire_net_contact_force_tensor(self.sim)
        dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        # Refresh to populate wrapped tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        self._dof_state = gymtorch.wrap_tensor(dof_state)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor).view(self.num_envs, 2 * 6)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force).view(self.num_envs, self.num_dof)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]

        self._contact_forces = gymtorch.wrap_tensor(contact).view(self.num_envs, self.num_bodies, 3)

        # Observation buffer
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)

    def _refresh_sim_tensors(self):
        """Refresh all GPU tensors from simulation."""
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

    # ── Body policy ───────────────────────────────────────────────────

    def _load_body_policy(self, checkpoint_path: str):
        """Load pre-trained body policy."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Body policy checkpoint not found: {checkpoint_path}")
        self.body_policy = load_body_policy(checkpoint_path, self.device)
        print(f"[ProKneeBase] Loaded frozen body policy from {checkpoint_path}")

    def _get_body_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from frozen body policy for all 28 DOFs."""
        if self.body_policy is None:
            return torch.zeros(self.num_envs, NUM_DOFS, device=self.device)
        with torch.no_grad():
            return self.body_policy(obs)

    # ── Observations ──────────────────────────────────────────────────

    def _compute_base_obs(self) -> torch.Tensor:
        """Compute 16-D proprioceptive observation (deployment-available).

        Hora-faithful: obs = only what prosthetic sensors can measure.
        Same content as _compute_student_proprio().
        """
        return self._compute_student_proprio()

    def _compute_full_body_obs(self) -> torch.Tensor:
        """Compute 105-D full body observation (sim-only, used in priv_info)."""
        root_states = self._root_states[:self.num_envs]
        dof_pos = self._dof_pos
        dof_vel = self._dof_vel
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]

        obs = compute_humanoid_observations(
            root_states, dof_pos, dof_vel, key_body_pos, self._local_root_obs,
        )
        return obs

    def _compute_priv_info(self) -> torch.Tensor:
        """Compute 113-D privileged information (sim-only).

        Includes full body observation (105D) + GRF (6D) + contacts (2D).
        This is the expanded priv_info for Hora-faithful architecture.
        """
        # Full body observation (105-D) — NOT available at deployment
        full_body_obs = self._compute_full_body_obs()

        # GRF 3-D for each foot (6-D total) from force sensors
        grf = self.vec_sensor_tensor[:, :6]

        # Foot contact (2-D)
        contact_threshold = 1.0
        right_fz = self.vec_sensor_tensor[:, 2:3].abs()
        left_fz = self.vec_sensor_tensor[:, 5:6].abs()
        contacts = torch.cat([
            (right_fz > contact_threshold).float(),
            (left_fz > contact_threshold).float(),
        ], dim=-1)

        return torch.cat([full_body_obs, grf, contacts], dim=-1)  # 105 + 6 + 2 = 113

    def _compute_student_proprio(self) -> torch.Tensor:
        """Compute 16-D student proprioception per timestep.
        
        Includes knee, ankle, hip pos/vel + foot force + command.
        """
        knee_pos = self._dof_pos[:, LEFT_KNEE:LEFT_KNEE+1]
        knee_vel = self._dof_vel[:, LEFT_KNEE:LEFT_KNEE+1]

        ankle_idx = [LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z]
        ankle_pos = self._dof_pos[:, ankle_idx]
        ankle_vel = self._dof_vel[:, ankle_idx]

        hip_idx = [LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y]
        hip_pos = self._dof_pos[:, hip_idx]
        hip_vel = self._dof_vel[:, hip_idx]

        # Left foot Fz (from force sensor, sensor 1 = left foot)
        left_foot_fz = self.vec_sensor_tensor[:, 5:6]  # Fz component

        # Command velocity
        command = torch.full((self.num_envs, 1), self.target_velocity, device=self.device)

        return torch.cat([
            knee_pos,       # 1
            knee_vel,       # 1
            ankle_pos,      # 3
            ankle_vel,      # 3
            hip_pos,        # 3
            hip_vel,        # 3
            left_foot_fz,   # 1
            command,        # 1
        ], dim=-1)

    def _update_proprio_history(self, proprio: torch.Tensor):
        """Shift history buffer and append new proprio."""
        self.proprio_hist = torch.roll(self.proprio_hist, shifts=-1, dims=1)
        self.proprio_hist[:, -1, :] = proprio

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get all observations (obs, priv_info, proprio_hist)."""
        obs = self._compute_base_obs()
        self.obs_buf = obs
        priv_info = self._compute_priv_info()
        proprio = self._compute_student_proprio()
        self._update_proprio_history(proprio)

        return {
            'obs': obs,
            'priv_info': priv_info,
            'proprio_hist': self.proprio_hist.clone(),
        }

    # ── Action application ────────────────────────────────────────────

    def _apply_prosthesis_action(self, prosthesis_action: torch.Tensor) -> torch.Tensor:
        """Combine prosthesis action with frozen body policy."""
        full_body_obs = self._compute_full_body_obs()  # 105D for body policy
        full_action = self._get_body_action(full_body_obs)

        # Override active prosthesis joints (knee + ankle)
        for i, joint_idx in enumerate(ACTIVE_PROSTHESIS_JOINTS):
            full_action[:, joint_idx] = prosthesis_action[:, i]

        return full_action

    def _action_to_pd_targets(self, action: torch.Tensor) -> torch.Tensor:
        """Convert normalised [-1,1] action to DOF position targets.

        Must match _build_pd_action_offset_scale from HumanoidAMPBase:
        - 3-DOF spherical joints use [-π, π] regardless of MJCF limits
        - 1-DOF hinge joints extend the range by 0.7× instead of 0.5×
        """
        return self._pd_action_offset + self._pd_action_scale * action

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_rewards(self) -> torch.Tensor:
        """Compute reward for prosthetic knee training.
        
        Uses direct walking quality metrics rather than body policy tracking,
        to avoid feedback loop where the body policy adapts to degraded states.
        """
        root_states = self._root_states[:self.num_envs]
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]
        root_vel = root_states[:, 7:10]

        # 1) Forward velocity tracking (DOMINANT reward)
        forward_vel = root_vel[:, 0]
        vel_err = (forward_vel - self.target_velocity) ** 2
        vel_reward = torch.exp(-2.0 * vel_err)

        # 2) Upright reward
        qx, qy = root_rot[:, 0], root_rot[:, 1]
        up_proj = 1 - 2 * (qx * qx + qy * qy)
        up_reward = torch.clamp(up_proj, 0.0, 1.0)

        # 3) Height reward — stay near standing height
        height_err = root_pos[:, 2] - 0.9
        height_reward = torch.exp(-10.0 * height_err ** 2)

        # 4) Lateral stability (penalize sideways drift)
        lateral_vel = root_vel[:, 1]
        lateral_penalty = lateral_vel ** 2

        # 5) Action smoothness (penalize saturation at ±1)
        action_cost = 0.01 * torch.sum(self.actions ** 2, dim=-1)
        # Extra penalty for actions near joint limits
        limit_cost = torch.sum(
            ((self.actions.abs() - 0.9).clamp(min=0) ** 2), dim=-1
        )

        reward = (
            3.0 * vel_reward
            + 1.0 * up_reward
            + 0.5 * height_reward
            - 0.3 * lateral_penalty
            - action_cost
            - 0.5 * limit_cost
        )

        # Death penalty
        fallen = root_pos[:, 2] < 0.3
        reward = torch.where(fallen, torch.ones_like(reward) * -10.0, reward)

        return reward

    # ── Termination ───────────────────────────────────────────────────

    def _check_termination(self) -> torch.Tensor:
        """Check episode termination."""
        root_h = self._root_states[:self.num_envs, 2]
        fallen = root_h < 0.3  # lower threshold to allow crouching gait
        timeout = self.progress_buf >= self.max_episode_length
        return (fallen | timeout).long()

    # ── Step / Reset ──────────────────────────────────────────────────

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Take a step in the environment."""
        self.actions = action.clamp(-1.0, 1.0).clone()

        # Resolve full 28-DOF action
        if self.prosthesis_only:
            full_action = self._apply_prosthesis_action(action)
        else:
            full_action = action

        # Convert to PD targets and apply
        pd_targets = self._action_to_pd_targets(full_action)
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(pd_targets),
        )

        # Simulate
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        # Refresh tensors
        self._refresh_sim_tensors()

        # Update progress
        self.progress_buf += 1

        # Compute observations / rewards / done
        observations = self.get_observations()
        rewards = self._compute_rewards()
        dones = self._check_termination()

        # Track completed episode lengths
        info = {'episode_length': self.progress_buf.clone()}
        reset_ids = torch.where(dones > 0)[0]
        if len(reset_ids) > 0:
            finished_lengths = self.progress_buf[reset_ids].float()
            info['finished_ep_len'] = finished_lengths
            self._reset_envs(reset_ids)

        return observations, rewards, dones.bool(), info

    def _reset_envs(self, env_ids: torch.Tensor):
        """Reset specific environments using AMP reference state initialisation."""
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # ── Sample random walking states from motion clip ──────────
        motion_ids = self._motion_lib.sample_motions(num_reset)
        motion_times = self._motion_lib.sample_time(motion_ids)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _key_pos = \
            self._motion_lib.get_motion_state(motion_ids, motion_times)

        # Set root state (pos/rot/vel/angvel)
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel

        # Set DOF state
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # Reset buffers
        self.progress_buf[env_ids] = 0
        self.proprio_hist[env_ids] = 0.0

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(all_ids)

        # Need one sim step to populate tensors after reset
        self.gym.simulate(self.sim)
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        self._refresh_sim_tensors()

        return self.get_observations()

    def close(self):
        """Destroy simulation."""
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
