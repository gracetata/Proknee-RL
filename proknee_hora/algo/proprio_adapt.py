"""
ProprioAdapt: Hora-faithful Stage 2 supervised distillation trainer.

Based on hora/hora/algo/padapt/padapt.py
Key differences from PPO-based training:
  - NO PPO loss, NO rollout buffer
  - Only adapt_tconv (student encoder) is trained
  - Everything else (backbone, actor, critic, priv_mlp) is frozen from Stage 1
  - Pure MSE loss: L = || adapt_tconv(proprio_hist) - priv_mlp(priv_info) ||^2
  - Online: each env step = one gradient update
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import logging

from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.algo.models.running_mean_std import RunningMeanStd
from proknee_hora.envs.constants import (
    OBS_DIM, TEACHER_PRIV_INFO_DIM, STUDENT_PROPRIO_DIM,
    LATENT_DIM, PROPRIO_HISTORY_LEN,
)

logger = logging.getLogger(__name__)


class AverageScalarMeter:
    """Rolling average meter for scalar values."""

    def __init__(self, window_size: int = 20000):
        self.window_size = window_size
        self.values = []
        self.sum = 0.0

    def update(self, values):
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy().flatten()
        for v in values:
            if len(self.values) >= self.window_size:
                self.sum -= self.values.pop(0)
            self.values.append(float(v))
            self.sum += float(v)

    def get_mean(self):
        if len(self.values) == 0:
            return 0.0
        return self.sum / len(self.values)


class ProprioAdapt:
    """Hora-faithful Stage 2 trainer: supervised latent distillation.

    Loads a Stage 1 teacher model, adds the adapt_tconv module,
    freezes everything except adapt_tconv, then trains with MSE loss
    on the latent representations.
    """

    def __init__(
        self,
        env,
        output_dir: str,
        device: str = 'cuda:0',
        lr: float = 3e-4,
        max_agent_steps: int = int(1e9),
        save_interval_steps: int = int(1e8),
        log_interval_steps: int = int(1e6),
    ):
        self.device = device
        self.env = env
        self.num_actors = env.num_envs
        self.max_agent_steps = max_agent_steps
        self.save_interval_steps = save_interval_steps
        self.log_interval_steps = log_interval_steps

        # ── Build model (same architecture as Stage 1, student mode) ──
        self.model = ProKneePolicy(
            obs_dim=OBS_DIM,
            action_dim=env.num_actions,
            priv_info_dim=TEACHER_PRIV_INFO_DIM,
            proprio_dim=STUDENT_PROPRIO_DIM,
            history_len=PROPRIO_HISTORY_LEN,
            latent_dim=LATENT_DIM,
            hidden_dims=[256, 128, 64],  # Must match Stage 1
            mode='student',
        )
        self.model.to(self.device)
        self.model.eval()  # Frozen parts in eval; adapt_tconv grads still flow

        # ── Running mean/std for obs normalization (frozen from Stage 1) ──
        self.running_mean_std = RunningMeanStd(OBS_DIM).to(self.device)
        self.running_mean_std.eval()

        # ── Running mean/std for proprio_hist normalization (trainable) ──
        self.sa_mean_std = RunningMeanStd((PROPRIO_HISTORY_LEN, STUDENT_PROPRIO_DIM)).to(self.device)
        self.sa_mean_std.train()

        # ── Output directories ──
        self.output_dir = output_dir
        self.nn_dir = os.path.join(output_dir, 'checkpoints')
        self.tb_dir = os.path.join(output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # ── TensorBoard ──
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.tb_dir)
        except ImportError:
            self.writer = None
            logger.warning("TensorBoard not available")

        # ── Freeze all except adapt_tconv ──
        adapt_params = []
        for name, p in self.model.named_parameters():
            if 'adapt_tconv' in name:
                adapt_params.append(p)
                p.requires_grad = True
            else:
                p.requires_grad = False

        n_adapt = sum(p.numel() for p in adapt_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params (adapt_tconv): {n_adapt:,} / {n_total:,} total")

        # ── Optimizer (only adapt_tconv params) ──
        self.optim = torch.optim.Adam(adapt_params, lr=lr)

        # ── Statistics ──
        self.agent_steps = 0
        self.best_rewards = -1e6
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.step_reward = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.latent_loss_accum = 0.0
        self.latent_loss_count = 0

    def restore_stage1(self, checkpoint_path: str):
        """Load Stage 1 teacher model weights (strict=False to allow missing adapt_tconv)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Stage 1 PPO checkpoint has 'policy_state_dict' key
        if 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Load with strict=False: adapt_tconv weights are randomly initialized
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded Stage 1 model from {checkpoint_path}")
        if missing:
            logger.info(f"  Missing keys (expected for adapt_tconv): {len(missing)}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected}")

        # If checkpoint has running_mean_std, load it too
        if 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            logger.info("  Loaded running_mean_std from checkpoint")

    def restore_test(self, checkpoint_path: str):
        """Load Stage 2 checkpoint for testing."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['policy_state_dict'])
        if 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'sa_mean_std' in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])

    def save(self, name: str):
        """Save checkpoint."""
        weights = {
            'model': self.model.state_dict(),
            'running_mean_std': self.running_mean_std.state_dict(),
            'sa_mean_std': self.sa_mean_std.state_dict(),
            'agent_steps': self.agent_steps,
            'best_rewards': self.best_rewards,
        }
        path = f'{name}.pth'
        torch.save(weights, path)
        return path

    def _forward_stage2(self, obs_dict):
        """Stage 2 forward: run both student (adapt_tconv) and teacher (priv_mlp).

        Following Hora's _actor_critic exactly:
        1. Student latent = adapt_tconv(proprio_hist) — used for action
        2. Teacher latent = priv_mlp(priv_info) — ground truth for loss
        3. obs + student_latent → frozen backbone → frozen actor → action
        """
        obs = obs_dict['obs']
        priv_info = obs_dict['priv_info']
        proprio_hist = obs_dict['proprio_hist']

        # Student latent (trainable)
        student_latent = self.model.adaptation.adapt_tconv(proprio_hist)

        # Teacher latent (frozen, detached)
        with torch.no_grad():
            teacher_latent = self.model.adaptation.priv_mlp(priv_info)

        # Action from frozen backbone using student latent
        action_mean, action_std, value = self.model.actor_critic(obs, student_latent)

        return action_mean, student_latent, teacher_latent

    def train(self):
        """Main training loop: Hora-faithful supervised distillation."""
        _t = time.time()
        _last_t = time.time()
        _last_log_steps = 0

        obs_dict = self.env.reset()
        self.agent_steps += self.num_actors

        while self.agent_steps <= self.max_agent_steps:
            # ── Normalize inputs ──
            obs_norm = self.running_mean_std(obs_dict['obs'].detach()).detach()
            hist_norm = self.sa_mean_std(obs_dict['proprio_hist'].detach())

            input_dict = {
                'obs': obs_norm,
                'priv_info': obs_dict['priv_info'],
                'proprio_hist': hist_norm,
            }

            # ── Forward pass (both student and teacher) ──
            mu, e, e_gt = self._forward_stage2(input_dict)

            # ── Loss: MSE on latent ──
            loss = ((e - e_gt.detach()) ** 2).mean()

            # ── Backward (only adapt_tconv gets gradients) ──
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # ── Step environment ──
            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.num_actors

            # ── Statistics ──
            self.latent_loss_accum += loss.item()
            self.latent_loss_count += 1

            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            # ── TensorBoard logging ──
            if self.writer and (self.agent_steps - _last_log_steps) >= self.log_interval_steps:
                _last_log_steps = self.agent_steps
                self._log_tensorboard()

            # ── Save checkpoints ──
            if self.agent_steps % self.save_interval_steps < self.num_actors:
                step_label = f'{int(self.agent_steps // 1e6)}M'
                self.save(os.path.join(self.nn_dir, f'step_{step_label}'))
                self.save(os.path.join(self.nn_dir, 'last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards and len(self.mean_eps_reward.values) > 100:
                self.save(os.path.join(self.nn_dir, 'best'))
                self.best_rewards = mean_rewards

            # ── Console output ──
            all_fps = self.agent_steps / (time.time() - _t + 1e-8)
            last_fps = self.num_actors / (time.time() - _last_t + 1e-8)
            _last_t = time.time()
            avg_loss = self.latent_loss_accum / max(self.latent_loss_count, 1)

            steps_m = int(self.agent_steps // 1e6)
            print(
                f'\r  Steps: {steps_m:04d}M | FPS: {all_fps:.0f} | '
                f'Reward: {mean_rewards:.2f} | Latent MSE: {avg_loss:.4f} | '
                f'Best: {self.best_rewards:.2f}',
                end='', flush=True,
            )

    def _log_tensorboard(self):
        if not self.writer:
            return
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
        avg_loss = self.latent_loss_accum / max(self.latent_loss_count, 1)
        self.writer.add_scalar('latent_mse/step', avg_loss, self.agent_steps)
        # Reset accumulators
        self.latent_loss_accum = 0.0
        self.latent_loss_count = 0

    def test(self):
        """Test loop: run student policy (no training)."""
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

        obs_dict = self.env.reset()
        while True:
            obs_norm = self.running_mean_std(obs_dict['obs']).detach()
            hist_norm = self.sa_mean_std(obs_dict['proprio_hist'].detach())

            input_dict = {
                'obs': obs_norm,
                'priv_info': obs_dict.get('priv_info', None),
                'proprio_hist': hist_norm,
            }
            mu, _, _ = self._forward_stage2(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
