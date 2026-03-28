"""
ProKnee Multi-Motion Environment for Isaac Gym.

Extends ProKneeBase to support multiple motion types (walk, run, dance, stand).
Each environment is assigned a motion type at reset, with the corresponding
body policy and reference motion loaded accordingly.

Key differences from ProKneeBase:
  - Loads multiple frozen body policies (one per motion type)
  - Adds motion type one-hot encoding to observation (16D → 20D)
  - Uses per-motion reference motion libraries for RSI
  - Supports episode-level and intra-episode motion switching
  - Does NOT modify ProKneeBase — inherits and extends

Usage:
    env = ProKneeMultiMotionEnv(
        num_envs=4096,
        device='cuda:0',
        motion_checkpoints={0: 'walk.pth', 1: 'run.pth', ...},
    )
"""

import os
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .proknee_base import ProKneeBase, load_body_policy, ISAAC_GYM_ENVS_PATH
from .constants_multi import (
    NUM_MOTION_TYPES,
    MOTION_WALK, MOTION_RUN, MOTION_DANCE, MOTION_STAND,
    MOTION_NAMES, MOTION_FILES,
    MOTION_TARGET_VELOCITIES, MOTION_REWARD_WEIGHTS,
    SINGLE_PROPRIO_DIM, MOTION_ENCODING_DIM,
    OBS_DIM_MULTI, STUDENT_PROPRIO_DIM_MULTI,
    OBS_DIM_REALISTIC, STUDENT_PROPRIO_DIM_REALISTIC,
    TEACHER_PRIV_INFO_DIM_MULTI,
    MOTION_SWITCH_INTERVAL_RANGE,
    ACTIVE_PROSTHESIS_JOINTS,
)

# Import motion library
from tasks.amp.utils_amp.motion_lib import MotionLib


class ProKneeMultiMotionEnv(ProKneeBase):
    """Multi-motion extension of ProKneeBase.

    Adds:
    - Multiple body policies (one per motion type)
    - Multiple motion libraries (for reference state init)
    - Motion type one-hot encoding in observation
    - Per-env motion assignment with optional intra-episode switching
    """

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda:0',
        headless: bool = True,
        # Multi-motion configuration
        motion_checkpoints: Optional[Dict[int, str]] = None,
        enabled_motions: Optional[List[int]] = None,
        motion_in_obs: bool = False,
        # Motion switching
        switch_motion_in_episode: bool = False,
        switch_interval_range: Tuple[int, int] = MOTION_SWITCH_INTERVAL_RANGE,
        # Transition blending
        transition_blend_steps: int = 0,
        # Base configuration (passed to ProKneeBase)
        proprio_hist_len: int = 30,
        control_freq_inv: int = 2,
        dt: float = 1.0 / 180.0,
        episode_length: int = 300,
    ):
        """Initialize multi-motion environment.

        Args:
            num_envs: Number of parallel environments.
            device: 'cuda:0' or 'cpu'.
            headless: If True, no rendering.
            motion_checkpoints: Dict mapping motion_id → Stage 0 checkpoint path.
                If None, uses default paths from constants_multi.
            enabled_motions: List of motion IDs to use. If None, uses all.
            motion_in_obs: If True (legacy), motion one-hot is in obs (20D).
                If False (realistic/default), motion one-hot is in priv_info only.
                The prosthesis must infer motion type from proprioceptive history.
            switch_motion_in_episode: If True, switch motions within an episode.
            switch_interval_range: (min, max) steps between switches.
            transition_blend_steps: Number of steps to linearly blend body actions
                during motion transitions (0 to disable).
            proprio_hist_len: History length for student.
            control_freq_inv: Control frequency inverse.
            dt: Simulation timestep.
            episode_length: Max episode length.
        """
        # Determine enabled motions
        self.enabled_motions = enabled_motions or list(range(NUM_MOTION_TYPES))
        self.num_motion_types = len(self.enabled_motions)
        self.switch_motion_in_episode = switch_motion_in_episode
        self.switch_interval_range = switch_interval_range
        self.motion_in_obs = motion_in_obs
        # When True, _reset_envs() preserves current env_motion_types (no random reassignment)
        # and step() skips intra-episode random switching. Used by evaluation scripts.
        self.manual_motion_control = False

        # Resolve checkpoint paths
        if motion_checkpoints is None:
            from .constants_multi import MOTION_STAGE0_CHECKPOINTS
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            stage0_dir = os.path.join(project_root, 'outputs', 'checkpoints', 'stage0')
            motion_checkpoints = {}
            for mid in self.enabled_motions:
                ckpt_name = MOTION_STAGE0_CHECKPOINTS[mid]
                motion_checkpoints[mid] = os.path.join(stage0_dir, ckpt_name)

        self._motion_checkpoints = motion_checkpoints

        # Use walk motion for base class initialization (will load all motions after)
        walk_ckpt = motion_checkpoints.get(MOTION_WALK)

        # Initialize base class with walk body policy
        # We set freeze_body=True but will manage body policies ourselves
        super().__init__(
            num_envs=num_envs,
            device=device,
            headless=headless,
            prosthesis_only=True,
            freeze_body=True,
            body_policy_checkpoint=walk_ckpt,
            proprio_hist_len=proprio_hist_len,
            control_freq_inv=control_freq_inv,
            dt=dt,
            episode_length=episode_length,
            motion_file=None,  # We'll handle motion loading ourselves
        )

        # Override observation dimensions for multi-motion
        if self.motion_in_obs:
            # Legacy mode: obs = 16D + 4D onehot = 20D, priv = 113D
            self.base_obs_dim = OBS_DIM_MULTI  # 20
            self.num_obs = OBS_DIM_MULTI
            self.proprio_dim = STUDENT_PROPRIO_DIM_MULTI  # 20
        else:
            # Realistic mode: obs = 16D (pure proprio), priv = 117D
            self.base_obs_dim = OBS_DIM_REALISTIC  # 16
            self.num_obs = OBS_DIM_REALISTIC
            self.proprio_dim = STUDENT_PROPRIO_DIM_REALISTIC  # 16
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)

        # Re-create proprio history buffer with new dimension
        self.proprio_hist = torch.zeros(
            self.num_envs, self.proprio_hist_len, self.proprio_dim,
            device=self.device
        )

        # ── Load per-motion body policies ──
        self._body_policies = {}
        for mid in self.enabled_motions:
            ckpt_path = motion_checkpoints[mid]
            if os.path.exists(ckpt_path):
                self._body_policies[mid] = load_body_policy(ckpt_path, self.device)
                print(f"[MultiMotion] Loaded body policy for {MOTION_NAMES[mid]}: {ckpt_path}")
            else:
                print(f"[MultiMotion] WARNING: Body policy not found for {MOTION_NAMES[mid]}: {ckpt_path}")
                # Fall back to walk policy if available
                if MOTION_WALK in self._body_policies:
                    self._body_policies[mid] = self._body_policies[MOTION_WALK]
                    print(f"  → Falling back to walk body policy")

        # ── Load per-motion reference motion libraries ──
        self._motion_libs = {}
        motions_dir = os.path.join(ISAAC_GYM_ENVS_PATH, 'assets', 'amp', 'motions')
        for mid in self.enabled_motions:
            motion_file = os.path.join(motions_dir, MOTION_FILES[mid])
            if os.path.exists(motion_file):
                self._motion_libs[mid] = MotionLib(
                    motion_file=motion_file,
                    num_dofs=self.num_dof,
                    key_body_ids=torch.tensor(self._key_body_ids.cpu().numpy()
                                              if hasattr(self._key_body_ids, 'cpu')
                                              else self._key_body_ids).long(),
                    device=self.device,
                )
                print(f"[MultiMotion] Loaded motion lib for {MOTION_NAMES[mid]}: {motion_file}")
            else:
                print(f"[MultiMotion] WARNING: Motion file not found for {MOTION_NAMES[mid]}: {motion_file}")
                if MOTION_WALK in self._motion_libs:
                    self._motion_libs[mid] = self._motion_libs[MOTION_WALK]

        # ── Per-env motion type assignment ──
        self.env_motion_types = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._assign_random_motions(torch.arange(self.num_envs, device=self.device))

        # ── Intra-episode motion switching timer ──
        if self.switch_motion_in_episode:
            self.switch_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._reset_switch_timers(torch.arange(self.num_envs, device=self.device))

        # ── Transition blending for smooth body policy switches ──
        self.transition_blend_steps = transition_blend_steps
        self._blend_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # For dual-policy blending: store the previous motion type per env
        self._prev_motion_types = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._has_stepped = False  # guard: no blending before first step

        print(f"[MultiMotion] Enabled motions: {[MOTION_NAMES[m] for m in self.enabled_motions]}")
        mode_str = "legacy (obs=20D)" if self.motion_in_obs else "realistic (obs=16D, priv=117D)"
        print(f"[MultiMotion] Observation mode: {mode_str}")
        if self.transition_blend_steps > 0:
            print(f"[MultiMotion] Transition blending: {self.transition_blend_steps} steps")
        print(f"[MultiMotion] obs_dim={self.num_obs}, proprio_dim={self.proprio_dim}")

    # ── Motion assignment ─────────────────────────────────────────────

    def _assign_random_motions(self, env_ids: torch.Tensor):
        """Randomly assign motion types to environments."""
        num = len(env_ids)
        motion_indices = torch.randint(
            0, len(self.enabled_motions), (num,), device=self.device
        )
        for i, env_id in enumerate(env_ids):
            self.env_motion_types[env_id] = self.enabled_motions[motion_indices[i]]

    def _assign_specific_motion(self, env_ids: torch.Tensor, motion_id: int):
        """Assign a specific motion type to environments."""
        self.env_motion_types[env_ids] = motion_id

    def _reset_switch_timers(self, env_ids: torch.Tensor):
        """Reset intra-episode switch timers to random values."""
        lo, hi = self.switch_interval_range
        self.switch_timer[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)

    # ── Motion one-hot encoding ───────────────────────────────────────

    def _get_motion_onehot(self) -> torch.Tensor:
        """Get one-hot encoding of current motion types for all envs.

        Returns:
            (num_envs, NUM_MOTION_TYPES) one-hot tensor.
        """
        return F.one_hot(self.env_motion_types, num_classes=NUM_MOTION_TYPES).float()

    # ── Observation override ──────────────────────────────────────────

    def _compute_base_obs(self) -> torch.Tensor:
        """Compute observation.

        motion_in_obs=True  (legacy):    20D = 16D proprio + 4D onehot
        motion_in_obs=False (realistic): 16D = pure proprio
        """
        proprio = self._compute_student_proprio_single()
        if self.motion_in_obs:
            motion_onehot = self._get_motion_onehot()
            return torch.cat([proprio, motion_onehot], dim=-1)
        return proprio

    def _compute_student_proprio_single(self) -> torch.Tensor:
        """Compute 16D proprio (same as base class, with per-motion velocity target)."""
        knee_pos = self._dof_pos[:, self._left_knee_idx:self._left_knee_idx + 1]
        knee_vel = self._dof_vel[:, self._left_knee_idx:self._left_knee_idx + 1]

        from .constants import LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z
        from .constants import LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y, LEFT_KNEE

        ankle_idx = [LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z]
        ankle_pos = self._dof_pos[:, ankle_idx]
        ankle_vel = self._dof_vel[:, ankle_idx]

        hip_idx = [LEFT_HIP_X, LEFT_HIP_Z, LEFT_HIP_Y]
        hip_pos = self._dof_pos[:, hip_idx]
        hip_vel = self._dof_vel[:, hip_idx]

        left_foot_fz = self.vec_sensor_tensor[:, 5:6]

        # Per-motion target velocity
        command = torch.zeros(self.num_envs, 1, device=self.device)
        for mid in self.enabled_motions:
            mask = (self.env_motion_types == mid)
            command[mask] = MOTION_TARGET_VELOCITIES[mid]

        return torch.cat([
            knee_pos, knee_vel,
            ankle_pos, ankle_vel,
            hip_pos, hip_vel,
            left_foot_fz,
            command,
        ], dim=-1)

    def _compute_student_proprio(self) -> torch.Tensor:
        """Compute student proprio for history buffer.

        motion_in_obs=True  (legacy):    20D = 16D proprio + 4D onehot
        motion_in_obs=False (realistic): 16D = pure proprio
        """
        proprio_16d = self._compute_student_proprio_single()
        if self.motion_in_obs:
            motion_onehot = self._get_motion_onehot()
            return torch.cat([proprio_16d, motion_onehot], dim=-1)
        return proprio_16d

    def _compute_priv_info(self) -> torch.Tensor:
        """Compute privileged information (sim-only).

        motion_in_obs=True  (legacy):    113D (same as base)
        motion_in_obs=False (realistic): 117D = 113D + 4D motion onehot
        """
        base_priv = super()._compute_priv_info()  # 113D
        if not self.motion_in_obs:
            motion_onehot = self._get_motion_onehot()
            return torch.cat([base_priv, motion_onehot], dim=-1)  # 117D
        return base_priv

    # ── Body policy override ──────────────────────────────────────────

    @property
    def _left_knee_idx(self):
        from .constants import LEFT_KNEE
        return LEFT_KNEE

    def _get_body_action_raw(self, obs: torch.Tensor) -> torch.Tensor:
        """Get raw body action from per-motion body policies (no blending)."""
        full_action = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        for mid in self.enabled_motions:
            if mid not in self._body_policies:
                continue
            mask = (self.env_motion_types == mid)
            if not mask.any():
                continue
            with torch.no_grad():
                action = self._body_policies[mid](obs[mask])
            full_action[mask] = action

        return full_action

    def _get_body_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get body action with deceleration-first transition blending.

        During transitions, the blend is split into two phases:
          Phase 1 (first half): Gradually reduce OLD policy output → agent decelerates
          Phase 2 (second half): Gradually introduce NEW policy from near-static state

        This avoids the out-of-distribution problem where the new policy
        receives a physical state it was never trained on (e.g., stand policy
        receiving walking-state observations).
        """
        # Get new (target) policy action
        new_action = self._get_body_action_raw(obs)

        if self.transition_blend_steps > 0:
            blending = (self._blend_counter > 0)
            if blending.any():
                blending_ids = torch.where(blending)[0]
                progress = 1.0 - (self._blend_counter[blending].float()
                                  / self.transition_blend_steps)  # 0.0 → 1.0

                # Get live OLD body policy action
                old_action = torch.zeros_like(new_action)
                for mid in self.enabled_motions:
                    if mid not in self._body_policies:
                        continue
                    mid_mask = (self._prev_motion_types[blending_ids] == mid)
                    if not mid_mask.any():
                        continue
                    mid_env_ids = blending_ids[mid_mask]
                    with torch.no_grad():
                        old_action[mid_env_ids] = self._body_policies[mid](obs[mid_env_ids])

                half = 0.5
                old_scale = torch.clamp(1.0 - progress / half, 0.0, 1.0)
                new_scale = torch.clamp((progress - half) / (1.0 - half), 0.0, 1.0)

                blended = (old_scale.unsqueeze(1) * old_action[blending]
                           + new_scale.unsqueeze(1) * new_action[blending])
                new_action[blending] = blended
                self._blend_counter[blending] -= 1

        return new_action

    # ── Reward override ───────────────────────────────────────────────

    def _compute_rewards(self) -> torch.Tensor:
        """Compute per-motion rewards with motion-specific weights."""
        root_states = self._root_states[:self.num_envs]
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]
        root_vel = root_states[:, 7:10]

        reward = torch.zeros(self.num_envs, device=self.device)

        for mid in self.enabled_motions:
            mask = (self.env_motion_types == mid)
            if not mask.any():
                continue

            weights = MOTION_REWARD_WEIGHTS[mid]
            target_vel = MOTION_TARGET_VELOCITIES[mid]

            # Forward velocity tracking
            forward_vel = root_vel[mask, 0]
            vel_err = (forward_vel - target_vel) ** 2
            vel_reward = torch.exp(-2.0 * vel_err)

            # Upright reward
            qx, qy = root_rot[mask, 0], root_rot[mask, 1]
            up_proj = 1 - 2 * (qx * qx + qy * qy)
            up_reward = torch.clamp(up_proj, 0.0, 1.0)

            # Height reward
            height_err = root_pos[mask, 2] - 0.9
            height_reward = torch.exp(-10.0 * height_err ** 2)

            # Lateral stability
            lateral_vel = root_vel[mask, 1]
            lateral_penalty = lateral_vel ** 2

            # Action cost
            action_cost = 0.01 * torch.sum(self.actions[mask] ** 2, dim=-1)
            limit_cost = torch.sum(
                ((self.actions[mask].abs() - 0.9).clamp(min=0) ** 2), dim=-1
            )

            r = (
                weights['vel_reward'] * vel_reward
                + weights['up_reward'] * up_reward
                + weights['height_reward'] * height_reward
                - weights['lateral_penalty'] * lateral_penalty
                - action_cost
                - 0.5 * limit_cost
            )

            # Death penalty
            fallen = root_pos[mask, 2] < 0.3
            r = torch.where(fallen, torch.ones_like(r) * -10.0, r)

            reward[mask] = r

        return reward

    # ── Reset override ────────────────────────────────────────────────

    def _reset_envs(self, env_ids: torch.Tensor):
        """Reset environments with per-motion reference state initialization."""
        num_reset = len(env_ids)
        if num_reset == 0:
            return

        # Re-assign random motions (skip when manual_motion_control is active)
        if not self.manual_motion_control:
            self._assign_random_motions(env_ids)

        # Reset switch timers if applicable
        if self.switch_motion_in_episode:
            self._reset_switch_timers(env_ids)

        # Per-motion RSI (Reference State Initialization)
        for mid in self.enabled_motions:
            mask = (self.env_motion_types[env_ids] == mid)
            mid_env_ids = env_ids[mask]
            if len(mid_env_ids) == 0:
                continue

            if mid not in self._motion_libs:
                # Fallback: use zero pose
                self._reset_to_default(mid_env_ids)
                continue

            motion_lib = self._motion_libs[mid]
            num_mid = len(mid_env_ids)

            motion_ids = motion_lib.sample_motions(num_mid)
            motion_times = motion_lib.sample_time(motion_ids)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _key_pos = \
                motion_lib.get_motion_state(motion_ids, motion_times)

            self._root_states[mid_env_ids, 0:3] = root_pos
            self._root_states[mid_env_ids, 3:7] = root_rot
            self._root_states[mid_env_ids, 7:10] = root_vel
            self._root_states[mid_env_ids, 10:13] = root_ang_vel
            self._dof_pos[mid_env_ids] = dof_pos
            self._dof_vel[mid_env_ids] = dof_vel

        # Apply resets to simulation
        from isaacgym import gymtorch
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
        # Clear blend counter for reset envs
        self._blend_counter[env_ids] = 0

    def _reset_to_default(self, env_ids: torch.Tensor):
        """Reset environments to default standing pose."""
        self._root_states[env_ids] = self._initial_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = 0.0

    # ── Step override (for intra-episode switching) ───────────────────

    def step(self, action: torch.Tensor):
        """Step with optional intra-episode motion switching."""
        self._has_stepped = True
        # Check if we need to switch motions mid-episode (skip in manual mode)
        if self.switch_motion_in_episode and not self.manual_motion_control:
            self.switch_timer -= 1
            switch_mask = (self.switch_timer <= 0)
            if switch_mask.any():
                switch_ids = torch.where(switch_mask)[0]
                self._assign_random_motions(switch_ids)
                self._reset_switch_timers(switch_ids)

        return super().step(action)

    # ── Utility ───────────────────────────────────────────────────────

    def get_motion_distribution(self) -> Dict[str, int]:
        """Get current distribution of motion types across environments."""
        dist = {}
        for mid in self.enabled_motions:
            count = (self.env_motion_types == mid).sum().item()
            dist[MOTION_NAMES[mid]] = count
        return dist

    def set_all_motions(self, motion_id: int):
        """Set all environments to a specific motion (for evaluation).

        Uses velocity-aware blend duration:
        - Transitioning to a slower motion (walk→stand): full deceleration blend
        - Transitioning to a faster motion (stand→walk): short blend (quick snap)
        """
        if self.transition_blend_steps > 0 and self._has_stepped:
            changed = (self.env_motion_types != motion_id)
            if changed.any():
                changed_ids = torch.where(changed)[0]
                self._prev_motion_types[changed_ids] = self.env_motion_types[changed_ids]

                # Velocity-aware blend: longer for deceleration, shorter for acceleration
                old_vel = abs(MOTION_TARGET_VELOCITIES.get(
                    self.env_motion_types[changed_ids[0]].item(), 0.0))
                new_vel = abs(MOTION_TARGET_VELOCITIES.get(motion_id, 0.0))
                if new_vel < old_vel:
                    # Decelerating: full blend for smooth stopping
                    effective_steps = self.transition_blend_steps
                else:
                    # Accelerating: short blend, let new policy take over fast
                    effective_steps = min(10, self.transition_blend_steps)

                self._blend_counter[changed_ids] = effective_steps
        self.env_motion_types[:] = motion_id

    def soft_reset_to_motion(self, motion_id: int):
        """Switch all envs to new motion with soft pose reset.

        Preserves root XY position and heading direction, but resets:
        - Joint positions/velocities → from new motion's RSI
        - Root height/velocity → from new motion's RSI
        - Observation history buffer

        This avoids the OOD problem entirely: each body policy receives
        in-distribution observations from its own motion's state space.

        Returns fresh observation dict (after one sim step).
        """
        from isaacgym import gymtorch

        all_ids = torch.arange(self.num_envs, device=self.device)

        # Save current root XY position
        saved_xy = self._root_states[:, 0:2].clone()

        # Update motion type and clear blend
        self.env_motion_types[:] = motion_id
        self._blend_counter[:] = 0

        # Sample RSI from the new motion's reference library
        if motion_id in self._motion_libs:
            mlib = self._motion_libs[motion_id]
            motion_ids = mlib.sample_motions(self.num_envs)
            motion_times = mlib.sample_time(motion_ids)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _ = \
                mlib.get_motion_state(motion_ids, motion_times)

            self._root_states[:, 0:2] = saved_xy  # preserve XY
            self._root_states[:, 2] = root_pos[:, 2]  # height from RSI
            self._root_states[:, 3:7] = root_rot
            self._root_states[:, 7:10] = root_vel
            self._root_states[:, 10:13] = root_ang_vel
            self._dof_pos[:] = dof_pos
            self._dof_vel[:] = dof_vel
        else:
            self._reset_to_default(all_ids)

        # Apply to simulation
        env_ids_int32 = all_ids.to(dtype=torch.int32)
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

        # Reset observation history (important for student model)
        self.proprio_hist[:] = 0.0
        self.progress_buf[:] = 0

        # Run one sim step to populate tensors, then return fresh observations
        self.gym.simulate(self.sim)
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        self._refresh_sim_tensors()

        return self.get_observations()

    def smooth_transition_to_motion(self, motion_id: int, transition_frames: int = 10):
        """Switch all envs to new motion with smooth N-frame state interpolation.

        Unlike soft_reset_to_motion() which teleports in one frame, this
        gradually interpolates the full physics state (root + DOF) over N
        frames using Hermite smooth-step. At each frame:
          - Root state (height, rotation, velocity) is interpolated
          - DOF positions and velocities are interpolated
          - PD targets are set to match interpolated DOF positions
          - Physics simulation runs one step (for viewer update)
          - proprio_hist is updated naturally (no zeroing)

        The prosthesis passively follows the interpolation during transition,
        then resumes student-policy control immediately after.

        Args:
            motion_id: Target motion type (MOTION_WALK/RUN/DANCE/STAND).
            transition_frames: Number of interpolation frames (default 10 ≈ 0.33s).

        Returns:
            Observation dict after transition completes.
        """
        from isaacgym import gymtorch

        all_ids = torch.arange(self.num_envs, device=self.device)
        env_ids_int32 = all_ids.to(dtype=torch.int32)

        # ── Capture current state as transition start ──
        saved_xy = self._root_states[:, 0:2].clone()
        start_root_h = self._root_states[:, 2].clone()
        start_root_rot = self._root_states[:, 3:7].clone()
        start_root_vel = self._root_states[:, 7:10].clone()
        start_root_ang = self._root_states[:, 10:13].clone()
        start_dof_pos = self._dof_pos.clone()
        start_dof_vel = self._dof_vel.clone()

        # ── Sample target RSI from new motion ──
        if motion_id not in self._motion_libs:
            # No motion library — fall back to hard reset
            return self.soft_reset_to_motion(motion_id)

        mlib = self._motion_libs[motion_id]
        motion_ids = mlib.sample_motions(self.num_envs)
        motion_times = mlib.sample_time(motion_ids)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _ = \
            mlib.get_motion_state(motion_ids, motion_times)

        target_root_h = root_pos[:, 2]
        target_root_rot = root_rot
        target_root_vel = root_vel
        target_root_ang = root_ang_vel
        target_dof_pos = dof_pos
        target_dof_vel = dof_vel

        # ── Switch motion type immediately (affects obs, reward, command) ──
        self.env_motion_types[:] = motion_id
        self._blend_counter[:] = 0

        # ── Interpolate over N frames ──
        for i in range(transition_frames):
            t = (i + 1.0) / transition_frames
            alpha = t * t * (3.0 - 2.0 * t)  # Hermite smooth step

            # Root state interpolation
            self._root_states[:, 0:2] = saved_xy  # preserve XY always
            self._root_states[:, 2] = (1.0 - alpha) * start_root_h \
                                      + alpha * target_root_h
            # Quaternion interpolation (lerp + renormalize; OK for small angles)
            interp_rot = (1.0 - alpha) * start_root_rot + alpha * target_root_rot
            interp_rot = interp_rot / interp_rot.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self._root_states[:, 3:7] = interp_rot
            self._root_states[:, 7:10] = (1.0 - alpha) * start_root_vel \
                                         + alpha * target_root_vel
            self._root_states[:, 10:13] = (1.0 - alpha) * start_root_ang \
                                          + alpha * target_root_ang

            # DOF state interpolation
            self._dof_pos[:] = (1.0 - alpha) * start_dof_pos + alpha * target_dof_pos
            self._dof_vel[:] = (1.0 - alpha) * start_dof_vel + alpha * target_dof_vel

            # Force-set physics state
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

            # Set PD targets to match interpolated positions
            interp_action = (self._dof_pos - self._pd_action_offset) \
                            / self._pd_action_scale
            pd_targets = self._action_to_pd_targets(interp_action.clamp(-1.0, 1.0))
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(pd_targets))

            # Simulate one step (updates viewer display)
            for _ in range(self.control_freq_inv):
                self.gym.simulate(self.sim)
                if self.viewer is not None:
                    self.gym.fetch_results(self.sim, True)
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, True)

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()

            # Update proprio history naturally (student sees smooth transition)
            obs = self.get_observations()

        # proprio_hist is NOT zeroed — it contains N frames of smooth transition
        # progress_buf is NOT reset — episode continues uninterrupted
        return obs
