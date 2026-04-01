# Copyright (c) 2021-2023, NVIDIA Corporation
# HumanMimic-style unified AMP: 0~3 m/s grid, walk/run demo only in configured bands;
# gap velocities use random walk/run for discriminator positives; extras["velocity_cmd"] for AMP agent blending.
# Backup baseline: HumanoidAMPUnifiedCurriculum (unchanged).

import numpy as np
import torch

from isaacgymenvs.tasks.humanoid_amp import build_amp_observations
from isaacgymenvs.tasks.humanoid_amp_unified import HumanoidAMPUnified


class HumanoidAMPUnifiedHumanMimic(HumanoidAMPUnified):
    """Velocity grid 0..velocityMax; reference walk/run bands only; gap learns from task + style-weighted AMP."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        env = cfg["env"]
        self._velocity_grid_step = float(env.get("velocityGridStep", 0.1))
        self._velocity_max = float(env.get("velocityMax", 3.0))
        self._walk_vel_min = float(env.get("walkVelMin", 0.5))
        self._walk_vel_max = float(env.get("walkVelMax", 1.5))
        self._run_vel_min = float(env.get("runVelMin", 2.0))
        self._run_vel_max = float(env.get("runVelMax", 3.0))
        self._stand_velocity_epsilon = float(env.get("standVelocityEpsilon", 0.05))
        self._walk_motion_index = int(env.get("walkMotionIndex", 0))
        self._run_motion_index = int(env.get("runMotionIndex", 1))
        self._enable_cmd_switch = bool(env.get("enableCmdSwitch", True))

        n_steps = int(round(self._velocity_max / self._velocity_grid_step)) + 1
        self._velocity_levels_np = (np.arange(n_steps, dtype=np.float32) * self._velocity_grid_step).clip(
            max=self._velocity_max
        )
        self._num_vel_levels = int(self._velocity_levels_np.shape[0])
        self._velocity_levels_t = None

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _ensure_velocity_levels_t(self):
        if self._velocity_levels_t is None:
            self._velocity_levels_t = torch.tensor(self._velocity_levels_np, device=self.device, dtype=torch.float)

    def _randomize_velocity_commands(self, env_ids):
        self._ensure_velocity_levels_t()
        idx = torch.randint(0, self._num_vel_levels, (env_ids.shape[0],), device=self.device)
        self._velocity_cmd[env_ids] = self._velocity_levels_t[idx]
        self._cmd_hold_steps[env_ids] = 0

    def _maybe_switch_velocity_commands(self):
        if not self._enable_cmd_switch:
            return
        super()._maybe_switch_velocity_commands()

    def post_physics_step(self):
        super().post_physics_step()
        self.extras["velocity_cmd"] = self._velocity_cmd.clone()

    def _demo_motion_ids_for_v(self, v_vals):
        motion_ids = np.empty(len(v_vals), dtype=np.int64)
        for i, v in enumerate(v_vals):
            if abs(v) < self._stand_velocity_epsilon:
                motion_ids[i] = self._walk_motion_index
            elif self._walk_vel_min <= v <= self._walk_vel_max:
                motion_ids[i] = self._walk_motion_index
            elif self._run_vel_min <= v <= self._run_vel_max:
                motion_ids[i] = self._run_motion_index
            else:
                motion_ids[i] = self._run_motion_index if np.random.randint(0, 2) == 1 else self._walk_motion_index
        return motion_ids

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        vel_idx = np.random.randint(0, self._num_vel_levels, size=num_samples)
        v_vals = self._velocity_levels_np[vel_idx]
        motion_ids = self._demo_motion_ids_for_v(v_vals)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert self._amp_obs_demo_buf.shape[0] == num_samples

        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(
            motion_ids, motion_times
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos, self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat
