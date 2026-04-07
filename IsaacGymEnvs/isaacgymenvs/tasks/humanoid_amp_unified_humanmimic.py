# Copyright (c) 2021-2023, NVIDIA Corporation
# HumanMimic-style unified AMP: 0~3 m/s grid, walk/run demo only in configured bands;
# gap velocities use random walk/run for discriminator positives; extras["velocity_cmd"] for AMP agent blending.

import os
import sys
import time
from os.path import join

import numpy as np
import torch

from isaacgym import gymapi

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
        # 键盘控速（与 scripts/interactive_unified.py / Stage2 类似）：关闭随机换速，仅听键盘
        self._manual_velocity_control = bool(env.get("manualVelocityControl", False))
        self._interactive_keyboard = bool(env.get("interactiveKeyboard", False))
        self._interactive_initial_vel = float(env.get("interactiveInitialVelocity", 1.0))
        # 与 velocityGridStep 对齐时，↑/↓ 每次一格（默认 0.1 m/s）
        self._interactive_vel_step = float(env.get("interactiveVelocityStep", 0.1))

        n_steps = int(round(self._velocity_max / self._velocity_grid_step)) + 1
        self._velocity_levels_np = (np.arange(n_steps, dtype=np.float32) * self._velocity_grid_step).clip(
            max=self._velocity_max
        )
        self._num_vel_levels = int(self._velocity_levels_np.shape[0])
        self._velocity_levels_t = None

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        if self._manual_velocity_control:
            if self._interactive_keyboard:
                # 与训练一致：从 0~velocityMax、步长 velocityGridStep 的离散网格随机取初速
                self._ensure_velocity_levels_t()
                all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                self._randomize_velocity_commands(all_ids)
            else:
                v0 = max(0.0, min(self._velocity_max, self._interactive_initial_vel))
                self._velocity_cmd[:] = v0
                self._cmd_hold_steps[:] = 0
        if self._interactive_keyboard and self.viewer is not None:
            self._setup_interactive_keyboard_subscriptions()

    def _ensure_velocity_levels_t(self):
        if self._velocity_levels_t is None:
            self._velocity_levels_t = torch.tensor(self._velocity_levels_np, device=self.device, dtype=torch.float)

    def _randomize_velocity_commands(self, env_ids):
        self._ensure_velocity_levels_t()
        idx = torch.randint(0, self._num_vel_levels, (env_ids.shape[0],), device=self.device)
        self._velocity_cmd[env_ids] = self._velocity_levels_t[idx]
        self._cmd_hold_steps[env_ids] = 0

    def _maybe_switch_velocity_commands(self):
        if self._manual_velocity_control:
            return
        if not self._enable_cmd_switch:
            return
        super()._maybe_switch_velocity_commands()

    def reset_idx(self, env_ids):
        if not self._manual_velocity_control:
            self._randomize_velocity_commands(env_ids)
        elif self._interactive_keyboard:
            self._randomize_velocity_commands(env_ids)
        super(HumanoidAMPUnified, self).reset_idx(env_ids)

    def _setup_interactive_keyboard_subscriptions(self):
        """与 interactive_unified.py 一致：↑/↓、W/F/S、0-9、Q；F=跑(2.5) 避免与 viewer 的 R=录屏 冲突。"""
        gym = self.gym
        viewer = self.viewer
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "hm_vel_up")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "hm_vel_down")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "hm_walk")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "hm_run")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "hm_stand")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "hm_max")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "hm_quit")
        num_keys = [
            gymapi.KEY_0, gymapi.KEY_1, gymapi.KEY_2, gymapi.KEY_3, gymapi.KEY_4,
            gymapi.KEY_5, gymapi.KEY_6, gymapi.KEY_7, gymapi.KEY_8, gymapi.KEY_9,
        ]
        for i, key in enumerate(num_keys):
            gym.subscribe_viewer_keyboard_event(viewer, key, f"hm_num_{i}")

    def _clamp_cmd(self, v):
        return max(0.0, min(float(v), self._velocity_max))

    def _snap_to_velocity_grid(self, v):
        """与训练速度离散网格一致（velocityGridStep / velocityMax）。"""
        v = self._clamp_cmd(v)
        i = int(round(v / self._velocity_grid_step + 1e-6))
        i = max(0, min(i, self._num_vel_levels - 1))
        return float(self._velocity_levels_np[i])

    def _handle_interactive_key(self, action):
        if action == "hm_quit":
            print("\n[HumanMimic interactive] Q — 退出")
            sys.exit(0)
        step = self._interactive_vel_step
        cur = self._velocity_cmd[0].item()
        new_v = None
        if action == "hm_vel_up":
            cur_s = self._snap_to_velocity_grid(cur)
            new_v = self._snap_to_velocity_grid(cur_s + step)
        elif action == "hm_vel_down":
            cur_s = self._snap_to_velocity_grid(cur)
            new_v = self._snap_to_velocity_grid(cur_s - step)
        elif action == "hm_walk":
            new_v = self._snap_to_velocity_grid(1.0)
        elif action == "hm_run":
            new_v = self._snap_to_velocity_grid(2.5)
        elif action == "hm_stand":
            new_v = 0.0
        elif action == "hm_max":
            new_v = self._velocity_max
        elif action.startswith("hm_num_"):
            num = int(action.split("_")[-1])
            new_v = self._snap_to_velocity_grid(num * step)
        if new_v is not None and abs(new_v - cur) > 1e-6:
            self._velocity_cmd[:] = new_v
            self._cmd_hold_steps[:] = 0

    def render(self):
        # VecTask.render 会先 query_viewer_action_events 并只处理 Q/V/R，队列被清空，
        # hm_* 必须在同一轮循环里处理，否则 post_physics_step 里读不到按键。
        if self.viewer and self._interactive_keyboard:
            if self.viewer and self.camera_follow:
                self._update_camera()
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.value > 0:
                    if evt.action == "QUIT":
                        sys.exit()
                    elif evt.action == "toggle_viewer_sync":
                        self.enable_viewer_sync = not self.enable_viewer_sync
                    elif evt.action == "record_frames":
                        self.record_frames = not self.record_frames
                    elif evt.action.startswith("hm_"):
                        self._handle_interactive_key(evt.action)
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    render_dt = self.dt * self.control_freq_inv
                else:
                    render_dt = 1.0 / self.render_fps
                if delta < render_dt:
                    time.sleep(render_dt - delta)
                self.last_frame_time = time.time()
            else:
                self.gym.poll_viewer_events(self.viewer)
            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)
                self.gym.write_viewer_image_to_file(
                    self.viewer, join(self.record_frames_dir, f"frame_{self.control_steps}.png")
                )
            if self.virtual_display:
                img = self.virtual_display.grab()
                return np.array(img)
            return
        super().render()

    def post_physics_step(self):
        super().post_physics_step()
        self.extras["velocity_cmd"] = self._velocity_cmd.clone()
        if self._interactive_keyboard:
            v = self._velocity_cmd[0].item()
            print(f"\r[v_cmd] {v:.2f} m/s (0~{self._velocity_max:.1f})   ", end="", flush=True)

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
