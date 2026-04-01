# Copyright (c) 2018-2023, NVIDIA Corporation
# HumanMimic-style AMP: imitation + style heads; gap velocities down-weight imitation, up-weight style.
# Backup baseline: amp_continuous.py (unchanged).

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

import numpy as np
import torch
from torch import nn

import isaacgymenvs.learning.amp_continuous as amp_continuous


class AMPAgentHumanMimic(amp_continuous.AMPAgent):
    """Extends AMPAgent with style discriminator and velocity-conditioned reward blending."""

    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._style_reward_w = float(config.get("style_reward_w", 0.5))
        self._style_reward_scale = float(config.get("style_reward_scale", 2.0))
        self._gap_imit_scale = float(config.get("gapImitationScale", 0.25))
        self._gap_style_scale = float(config.get("gapStyleScale", 1.5))
        self._walk_vel_min = float(config.get("walkVelMin", 0.5))
        self._walk_vel_max = float(config.get("walkVelMax", 1.5))
        self._run_vel_min = float(config.get("runVelMin", 2.0))
        self._run_vel_max = float(config.get("runVelMax", 3.0))

    def _build_amp_buffers(self):
        super()._build_amp_buffers()
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["velocity_cmd"] = torch.zeros(batch_shape + (1,), device=self.ppo_device)

    def play_steps(self):
        self.set_eval()

        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data("obses", n, self.obs["obs"])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)
            self.experience_buffer.update_data("amp_obs", n, infos["amp_obs"])

            if "velocity_cmd" in infos:
                vc = infos["velocity_cmd"]
                if vc.dim() == 1:
                    vc = vc.unsqueeze(-1)
                self.experience_buffer.update_data("velocity_cmd", n, vc)
            else:
                z = torch.zeros_like(self.experience_buffer.tensor_dict["rewards"][n])
                self.experience_buffer.update_data("velocity_cmd", n, z)

            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= 1.0 - terminated
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if self.vec_env.env.viewer and (n == (self.horizon_length - 1)):
                self._amp_debug(infos)

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]

        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]
        mb_velocity_cmd = self.experience_buffer.tensor_dict["velocity_cmd"]
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards, mb_velocity_cmd)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = input_dict["amp_obs"][0 : self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = input_dict["amp_obs_replay"][0 : self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict["amp_obs_demo"][0 : self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
            "amp_obs": amp_obs,
            "amp_obs_replay": amp_obs_replay,
            "amp_obs_demo": amp_obs_demo,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]
            disc_agent_logit = res_dict["disc_agent_logit"]
            disc_agent_replay_logit = res_dict["disc_agent_replay_logit"]
            disc_demo_logit = res_dict["disc_demo_logit"]
            style_agent_logit = res_dict["style_agent_logit"]
            style_agent_replay_logit = res_dict["style_agent_replay_logit"]
            style_demo_logit = res_dict["style_demo_logit"]

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info["actor_loss"]

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info["critic_loss"]

            b_loss = self.bound_loss(mu)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            style_agent_cat_logit = torch.cat([style_agent_logit, style_agent_replay_logit], dim=0)

            disc_info = self._disc_loss_humanmimic(
                disc_agent_cat_logit,
                disc_demo_logit,
                style_agent_cat_logit,
                style_demo_logit,
                amp_obs_demo,
            )
            disc_loss = disc_info["disc_loss"]

            loss = (
                a_loss
                + self.critic_coef * c_loss
                - self.entropy_coef * entropy
                + self.bounds_loss_coef * b_loss
                + self._disc_coef * disc_loss
            )

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.train_result = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
            "b_loss": b_loss,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)

    def _disc_loss_humanmimic(self, disc_agent_logit, disc_demo_logit, style_agent_logit, style_demo_logit, obs_demo):
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        imit_pred = 0.5 * (disc_loss_agent + disc_loss_demo)

        style_loss_agent = self._disc_loss_neg(style_agent_logit)
        style_loss_demo = self._disc_loss_pos(style_demo_logit)
        style_pred = 0.5 * (style_loss_agent + style_loss_demo)

        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss = imit_pred + style_pred + self._disc_logit_reg * disc_logit_loss

        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gp_imit = torch.mean(torch.sum(torch.square(disc_demo_grad), dim=-1))

        style_demo_grad = torch.autograd.grad(
            style_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(style_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gp_style = torch.mean(torch.sum(torch.square(style_demo_grad), dim=-1))

        disc_loss = disc_loss + self._disc_grad_penalty * (gp_imit + gp_style)

        if self._disc_weight_decay != 0:
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss = disc_loss + self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)
        style_agent_acc, style_demo_acc = self._compute_disc_acc(style_agent_logit, style_demo_logit)

        return {
            "disc_loss": disc_loss,
            "disc_grad_penalty": gp_imit + gp_style,
            "disc_logit_loss": disc_logit_loss,
            "disc_agent_acc": disc_agent_acc,
            "disc_demo_acc": disc_demo_acc,
            "disc_agent_logit": disc_agent_logit,
            "disc_demo_logit": disc_demo_logit,
            "style_agent_acc": style_agent_acc,
            "style_demo_acc": style_demo_acc,
            "style_agent_logit": style_agent_logit,
            "style_demo_logit": style_demo_logit,
        }

    def _velocity_blend(self, velocity_cmd):
        v = velocity_cmd.squeeze(-1)
        walk = (v >= self._walk_vel_min) & (v <= self._walk_vel_max)
        run = (v >= self._run_vel_min) & (v <= self._run_vel_max)
        ref = walk | run
        im = torch.where(
            ref,
            torch.ones_like(v, dtype=torch.float32),
            torch.full_like(v, self._gap_imit_scale, dtype=torch.float32),
        )
        sm = torch.where(
            ref,
            torch.ones_like(v, dtype=torch.float32),
            torch.full_like(v, self._gap_style_scale, dtype=torch.float32),
        )
        return im.unsqueeze(-1), sm.unsqueeze(-1)

    def _combine_rewards(self, task_rewards, amp_rewards, velocity_cmd):
        disc_r = amp_rewards["disc_rewards"]
        style_r = amp_rewards["style_rewards"]
        im_m, sm_m = self._velocity_blend(velocity_cmd)
        return (
            self._task_reward_w * task_rewards
            + self._disc_reward_w * im_m * disc_r
            + self._style_reward_w * sm_m * style_r
        )

    def _eval_style_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_style_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        style_r = self._calc_style_rewards(amp_obs)
        return {"disc_rewards": disc_r, "style_rewards": style_r}

    def _calc_style_rewards(self, amp_obs):
        with torch.no_grad():
            logits = self._eval_style_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-logits))
            style_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            style_r *= self._style_reward_scale
        return style_r

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info["style_rewards"] = batch_dict.get("style_rewards")

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        self.writer.add_scalar("info/style_agent_acc", torch_ext.mean_list(train_info["style_agent_acc"]).item(), frame)
        self.writer.add_scalar("info/style_demo_acc", torch_ext.mean_list(train_info["style_demo_acc"]).item(), frame)
        if train_info.get("style_rewards") is not None:
            sr_std, sr_mean = torch.std_mean(train_info["style_rewards"])
            self.writer.add_scalar("info/style_reward_mean", sr_mean.item(), frame)
            self.writer.add_scalar("info/style_reward_std", sr_std.item(), frame)
