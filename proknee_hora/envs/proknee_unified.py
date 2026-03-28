"""
ProKnee Unified Environment for Velocity-Conditioned Policy Training.

Extends ProKneeBase to support the Unified velocity-conditioned body policy (Phase 11+).
Instead of multiple per-motion body policies, uses a single unified policy that
takes a velocity command (0~2.5 m/s) as input.

Does NOT modify the original proknee_base.py.

Architecture:
  - Body policy: Unified (106D input = 105D humanoid_obs + 1D velocity_cmd)
  - Prosthesis policy: 16D obs + 114D priv_info (113D base + 1D velocity_cmd)
  - Student: Infers velocity from 30×16D proprio_hist (no explicit velocity input)

Usage:
    from proknee_hora.envs.proknee_unified import ProKneeUnifiedEnv

    env = ProKneeUnifiedEnv(
        num_envs=4096,
        device='cuda:0',
        body_policy_checkpoint='outputs/checkpoints/stage0/stage0_unified_1800.pth',
    )
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# Fix numpy deprecations before isaacgym imports
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

from isaacgym import gymapi

from .proknee_base import ProKneeBase, KEY_BODY_NAMES, NUM_OBS
from .constants import (
    NUM_DOFS,
    ACTIVE_PROSTHESIS_JOINTS,
    LATENT_DIM,
    PROPRIO_HISTORY_LEN,
)
from .constants_unified import (
    OBS_DIM_UNIFIED,
    TEACHER_PRIV_INFO_DIM_UNIFIED,
    STUDENT_PROPRIO_DIM_UNIFIED,
    UNIFIED_BODY_OBS_DIM,
    VELOCITY_CMD_DIM,
    VELOCITY_MIN,
    VELOCITY_MAX,
    VELOCITY_LEVELS,
    VELOCITY_STAND,
    VELOCITY_WALK,
    VELOCITY_RUN,
    VELOCITY_SWITCH_PROB,
    VELOCITY_SWITCH_INTERVAL,
    VELOCITY_REWARD_WEIGHT,
)


# ── Unified Body Policy (106D input instead of 105D) ─────────────────────
class UnifiedBodyPolicy(nn.Module):
    """Unified velocity-conditioned body policy (106D input → 28D action).
    
    Input: 106D = [105D humanoid_obs, 1D velocity_cmd]
    Output: 28D joint target positions
    """

    def __init__(self, obs_dim: int = UNIFIED_BODY_OBS_DIM, action_dim: int = 28):
        super().__init__()
        # Architecture must match HumanoidAMPUnified training
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.mu = nn.Linear(512, action_dim)

        # Running mean/std for observation normalization
        self.running_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(obs_dim), requires_grad=False)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.running_mean) / torch.sqrt(self.running_var + 1e-5)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (28-D)."""
        x = self.normalize_obs(obs)
        x = self.actor_mlp(x)
        return self.mu(x)


def load_unified_body_policy(checkpoint_path: str, device: str) -> UnifiedBodyPolicy:
    """Load Stage-0 Unified AMP checkpoint into UnifiedBodyPolicy."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_sd = ckpt['model']

    policy = UnifiedBodyPolicy().to(device)

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


class ProKneeUnifiedEnv(ProKneeBase):
    """ProKnee environment with unified velocity-conditioned body policy.
    
    Key differences from ProKneeBase:
      1. Uses UnifiedBodyPolicy (106D input) instead of FrozenBodyPolicy (105D)
      2. Velocity command stored per-env and passed to body policy
      3. priv_info includes velocity command (113D + 1D = 114D)
      4. Student obs remains 16D (infers velocity from history)
    """

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda:0',
        headless: bool = True,
        body_policy_checkpoint: Optional[str] = None,
        proprio_hist_len: int = PROPRIO_HISTORY_LEN,
        episode_length: int = 300,
        # Velocity command configuration
        velocity_switch_prob: float = VELOCITY_SWITCH_PROB,
        velocity_switch_interval: int = VELOCITY_SWITCH_INTERVAL,
        initial_velocity: Optional[float] = None,
        # For backward compatibility
        motion_file: Optional[str] = None,
    ):
        # Store unified-specific config before super().__init__
        self._vel_switch_prob = velocity_switch_prob
        self._vel_switch_interval = velocity_switch_interval
        self._initial_velocity = initial_velocity

        # Override dimensions for unified mode
        self.unified_obs_dim = OBS_DIM_UNIFIED  # 16
        self.unified_priv_dim = TEACHER_PRIV_INFO_DIM_UNIFIED  # 114
        self.unified_proprio_dim = STUDENT_PROPRIO_DIM_UNIFIED  # 16

        # Default motion file (walk) for reference state init
        if motion_file is None:
            isaac_gym_envs_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'IsaacGymEnvs',
            )
            motion_file = os.path.join(isaac_gym_envs_path,
                                       'assets', 'amp', 'motions',
                                       'amp_humanoid_walk.npy')

        # Initialize base environment (without loading body policy yet)
        super().__init__(
            num_envs=num_envs,
            device=device,
            headless=headless,
            prosthesis_only=True,  # Always prosthesis control
            freeze_body=False,  # We'll handle body policy ourselves
            body_policy_checkpoint=None,  # Load unified policy below
            proprio_hist_len=proprio_hist_len,
            episode_length=episode_length,
            motion_file=motion_file,
        )

        # Override observation/priv dimensions
        self.base_obs_dim = self.unified_obs_dim  # 16
        self.priv_info_dim = self.unified_priv_dim  # 114
        self.proprio_dim = self.unified_proprio_dim  # 16
        self.num_obs = self.base_obs_dim  # 16

        # Reinitialize proprio_hist with correct dimensions
        self.proprio_hist = torch.zeros(
            self.num_envs, self.proprio_hist_len, self.proprio_dim, device=self.device
        )

        # Velocity command buffer (one scalar per env)
        self._velocity_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._cmd_hold_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Load unified body policy
        self.unified_body_policy = None
        if body_policy_checkpoint:
            self._load_unified_body_policy(body_policy_checkpoint)

        # Initialize velocity commands
        self._randomize_velocity_commands(torch.arange(self.num_envs, device=self.device))

        # Override with initial velocity if specified
        if self._initial_velocity is not None:
            self._velocity_cmd[:] = self._initial_velocity

        # Manual control flag (for interactive visualization)
        self.manual_velocity_control = False

    def _load_unified_body_policy(self, checkpoint_path: str):
        """Load unified body policy."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Unified body policy checkpoint not found: {checkpoint_path}")
        self.unified_body_policy = load_unified_body_policy(checkpoint_path, self.device)
        print(f"[ProKneeUnifiedEnv] Loaded unified body policy from {checkpoint_path}")

    # ── Velocity Command Management ───────────────────────────────────────

    def _randomize_velocity_commands(self, env_ids: torch.Tensor):
        """Sample random velocity commands for given envs."""
        n = len(env_ids)
        levels = torch.tensor(VELOCITY_LEVELS, device=self.device, dtype=torch.float)
        indices = torch.randint(0, len(VELOCITY_LEVELS), (n,), device=self.device)
        self._velocity_cmd[env_ids] = levels[indices]
        self._cmd_hold_steps[env_ids] = 0

    def _maybe_switch_commands(self):
        """Stochastically switch velocity commands mid-episode."""
        if self.manual_velocity_control:
            return  # Skip auto-switching in manual mode

        self._cmd_hold_steps += 1

        # Only consider switching for envs that have held command long enough
        eligible = self._cmd_hold_steps >= self._vel_switch_interval
        if not eligible.any():
            return

        # Stochastic switch
        switch_mask = torch.bernoulli(
            torch.full((self.num_envs,), self._vel_switch_prob, device=self.device)
        ).bool() & eligible

        if switch_mask.any():
            switch_ids = torch.where(switch_mask)[0]
            self._randomize_velocity_commands(switch_ids)

    def set_velocity(self, velocity: float, env_ids: Optional[torch.Tensor] = None):
        """Manually set velocity command for specified envs (or all if None)."""
        velocity = max(VELOCITY_MIN, min(VELOCITY_MAX, velocity))
        if env_ids is None:
            self._velocity_cmd[:] = velocity
            self._cmd_hold_steps[:] = 0
        else:
            self._velocity_cmd[env_ids] = velocity
            self._cmd_hold_steps[env_ids] = 0

    def get_velocity(self, env_id: int = 0) -> float:
        """Get current velocity command for an env."""
        return self._velocity_cmd[env_id].item()

    # ── Body Policy Override ──────────────────────────────────────────────

    def _get_body_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from unified body policy (106D input).
        
        obs: 105D humanoid observation (from _compute_full_body_obs)
        Returns: 28D joint target positions
        """
        if self.unified_body_policy is None:
            return torch.zeros(self.num_envs, NUM_DOFS, device=self.device)

        # Append velocity command to observation
        vel_cmd = self._velocity_cmd.unsqueeze(-1)  # (N, 1)
        unified_obs = torch.cat([obs, vel_cmd], dim=-1)  # (N, 106)

        with torch.no_grad():
            return self.unified_body_policy(unified_obs)

    # ── Observations ──────────────────────────────────────────────────────

    def _compute_priv_info(self) -> torch.Tensor:
        """Compute 114-D privileged information (113D base + 1D velocity_cmd).
        
        This overrides the base class to include velocity command in priv_info.
        """
        # Get base privileged info (113D)
        base_priv = super()._compute_priv_info()

        # Append velocity command
        vel_cmd = self._velocity_cmd.unsqueeze(-1)  # (N, 1)
        return torch.cat([base_priv, vel_cmd], dim=-1)  # (N, 114)

    def _compute_reward(self) -> torch.Tensor:
        """Compute reward with velocity tracking component.
        
        Overrides base reward to include velocity tracking.
        """
        # Get base reward
        base_reward = super()._compute_reward()

        # Velocity tracking reward
        root_vel = self._root_states[:, 7:10]  # World-frame velocity
        forward_vel = root_vel[:, 0]  # X-axis (forward)

        vel_error = forward_vel - self._velocity_cmd
        vel_reward = torch.exp(-VELOCITY_REWARD_WEIGHT * vel_error * vel_error)

        # Combine rewards (weighted average)
        return 0.7 * base_reward + 0.3 * vel_reward

    # ── Reset Override ────────────────────────────────────────────────────

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset all environments."""
        obs_dict = super().reset()
        # Randomize velocity commands on reset
        self._randomize_velocity_commands(torch.arange(self.num_envs, device=self.device))
        return obs_dict

    def _reset_envs(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        super()._reset_envs(env_ids)
        self._randomize_velocity_commands(env_ids)

    # ── Step Override ─────────────────────────────────────────────────────

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Step the environment with velocity command switching."""
        result = super().step(actions)
        self._maybe_switch_commands()
        return result

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def velocity_cmd(self) -> torch.Tensor:
        """Current velocity commands for all envs."""
        return self._velocity_cmd

    @property
    def velocity_levels(self) -> list:
        """Available discrete velocity levels."""
        return VELOCITY_LEVELS.copy()
