# Copyright (c) 2021-2023, NVIDIA Corporation
# Unified velocity-conditioned Humanoid AMP: policy obs = 105D humanoid + 1D velocity command.

import torch

from isaacgymenvs.tasks.humanoid_amp import HumanoidAMP
from isaacgymenvs.tasks.amp.humanoid_amp_base import NUM_OBS


class HumanoidAMPUnified(HumanoidAMP):
    """Humanoid AMP with scalar velocity command appended to policy observations (106D)."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self._cmd_switch_prob = cfg["env"].get("cmdSwitchProb", 0.005)
        self._cmd_switch_interval = cfg["env"].get("cmdSwitchInterval", 100)
        self._vel_reward_w = cfg["env"].get("velRewardWeight", 2.0)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self._velocity_cmd = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._cmd_hold_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._randomize_velocity_commands(torch.arange(self.num_envs, device=self.device))

    def get_obs_size(self):
        return NUM_OBS + 1

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            base = self._compute_humanoid_obs(None)
            vel = self._velocity_cmd.unsqueeze(-1)
            self.obs_buf[:] = torch.cat([base, vel], dim=-1)
        else:
            base = self._compute_humanoid_obs(env_ids)
            vel = self._velocity_cmd[env_ids].unsqueeze(-1)
            self.obs_buf[env_ids] = torch.cat([base, vel], dim=-1)
        return

    def reset_idx(self, env_ids):
        self._randomize_velocity_commands(env_ids)
        super().reset_idx(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        self._maybe_switch_velocity_commands()

    def _randomize_velocity_commands(self, env_ids):
        levels = torch.tensor([0.0, 1.0, 2.5], device=self.device, dtype=torch.float)
        idx = torch.randint(0, 3, (env_ids.shape[0],), device=self.device)
        self._velocity_cmd[env_ids] = levels[idx]
        self._cmd_hold_steps[env_ids] = 0

    def _maybe_switch_velocity_commands(self):
        self._cmd_hold_steps += 1
        switch_mask = torch.rand(self.num_envs, device=self.device) < self._cmd_switch_prob
        hold_mask = self._cmd_hold_steps < self._cmd_switch_interval
        switch_mask = torch.logical_and(switch_mask, torch.logical_not(hold_mask))
        if switch_mask.any():
            ids = switch_mask.nonzero(as_tuple=False).flatten()
            self._randomize_velocity_commands(ids)

    def _compute_reward(self, actions):
        root_vel = self._root_states[:, 7:10]
        forward_vel = root_vel[:, 0]
        vel_err = forward_vel - self._velocity_cmd
        w = self._vel_reward_w
        r_task = torch.exp(-w * vel_err * vel_err)

        root_h = self._root_states[:, 2]
        r_upright = torch.clamp((root_h - 0.5) / 0.5, 0.0, 1.0)

        lateral_vel = root_vel[:, 1]
        r_lat = torch.exp(-4.0 * lateral_vel * lateral_vel)

        self.rew_buf[:] = 0.6 * r_task + 0.2 * r_upright + 0.2 * r_lat
        return
