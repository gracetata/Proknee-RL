"""
Teacher-Student Training for ProKnee-Hora.

Implements distillation training where the student learns to match
the teacher's latent representation from proprioceptive history.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import os

from .ppo import PPO, RolloutBuffer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TeacherStudentTrainer:
    """Teacher-Student training with distillation.
    
    Training process:
    1. Load pre-trained teacher (Stage 1)
    2. Freeze teacher's privileged info encoder
    3. Train student's proprioceptive adapter to match teacher's latent
    4. Combine PPO loss with distillation loss
    """
    
    def __init__(
        self,
        student_policy,
        teacher_policy,
        env,
        # PPO hyperparameters
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.95,
        e_clip: float = 0.2,
        entropy_coef: float = 0.005,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        # Distillation
        distill_coef: float = 1.0,
        # Rollout settings
        horizon_length: int = 16,
        minibatch_size: int = 8192,
        mini_epochs: int = 5,
        # Device
        device: str = 'cuda:0',
        # Proprioceptive settings
        proprio_dim: int = 10,
        history_len: int = 30,
    ):
        """Initialize Teacher-Student trainer.
        
        Args:
            student_policy: Student policy network.
            teacher_policy: Pre-trained teacher policy (will be frozen).
            env: Environment (ProKneeStudent).
            learning_rate: Learning rate for student.
            gamma: Discount factor.
            tau: GAE lambda.
            e_clip: PPO clip parameter.
            entropy_coef: Entropy bonus coefficient.
            value_loss_coef: Value loss coefficient.
            max_grad_norm: Maximum gradient norm.
            distill_coef: Distillation loss coefficient (lambda).
            horizon_length: Steps per rollout.
            minibatch_size: Minibatch size.
            mini_epochs: PPO epochs per update.
            device: PyTorch device.
            proprio_dim: Proprioceptive dimension per timestep.
            history_len: Proprioceptive history length.
        """
        self.student_policy = student_policy
        self.teacher_policy = teacher_policy
        self.env = env
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.e_clip = e_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.distill_coef = distill_coef
        self.mini_epochs = mini_epochs
        self.minibatch_size = minibatch_size
        
        # Freeze teacher
        self._freeze_teacher()
        
        # Optimizer (only student parameters)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, student_policy.parameters()),
            lr=learning_rate
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_envs=env.num_envs,
            horizon_length=horizon_length,
            obs_dim=env.num_obs,
            action_dim=env.num_actions,
            device=device,
            priv_info_dim=env.priv_info_dim,
            proprio_dim=proprio_dim,
            history_len=history_len,
        )
        
        # Latent buffer for distillation
        self.student_latents = torch.zeros(
            horizon_length, env.num_envs, student_policy.adaptation.latent_dim,
            device=device
        )
        self.teacher_latents = torch.zeros(
            horizon_length, env.num_envs, teacher_policy.adaptation.latent_dim,
            device=device
        )
        
        # Running reward normalization
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self.reward_count = 1e-4
        
        # Statistics
        self.epoch = 0
        self.total_steps = 0
        
    def _freeze_teacher(self):
        """Freeze teacher policy parameters."""
        for param in self.teacher_policy.parameters():
            param.requires_grad = False
        self.teacher_policy.eval()
        logger.info("Teacher policy frozen")
    
    def collect_rollout(self) -> Dict[str, float]:
        """Collect rollout with both student and teacher latents.
        
        Returns:
            Statistics dictionary.
        """
        self.buffer.reset()
        
        # Get current observations WITHOUT resetting all envs.
        # (Individual env resets happen inside step() when envs terminate.)
        if not hasattr(self, '_initialized') or not self._initialized:
            full_obs = self.env.reset()
            self._initialized = True
        else:
            full_obs = self.env.get_observations()
        
        obs_dict = {
            'obs': full_obs['obs'],
            'priv_info': full_obs['priv_info'],
            'proprio_hist': full_obs['proprio_hist'],
        }
        
        total_reward = 0
        episode_count = 0
        
        for step in range(self.buffer.horizon_length):
            obs = obs_dict['obs']
            priv_info = obs_dict['priv_info']
            proprio_hist = obs_dict['proprio_hist']
            
            # Get student action and latent
            with torch.no_grad():
                student_output = self.student_policy(obs, proprio_hist=proprio_hist)
                action_mean = student_output['action_mean']
                action_std = student_output['action_std']
                value = student_output['value'].squeeze(-1)
                student_latent = student_output['latent']
                
                # Get teacher latent
                teacher_output = self.teacher_policy(obs, priv_info=priv_info)
                teacher_latent = teacher_output['latent']
                
                # Sample action from student
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Store latents
            self.student_latents[step] = student_latent
            self.teacher_latents[step] = teacher_latent
            
            # Step environment
            next_obs_dict, reward, done, info = self.env.step(action)
            
            # Normalize reward (running mean/std)
            batch_mean = reward.mean().item()
            batch_var = reward.var().item()
            batch_count = reward.numel()
            delta = batch_mean - self.reward_running_mean
            tot_count = self.reward_count + batch_count
            self.reward_running_mean += delta * batch_count / tot_count
            self.reward_running_var += batch_var * batch_count + delta ** 2 * self.reward_count * batch_count / tot_count
            self.reward_count = tot_count
            reward_std = max((self.reward_running_var / self.reward_count) ** 0.5, 1e-4)
            normalized_reward = reward / reward_std

            # Store transition
            self.buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=normalized_reward,
                done=done,
                value=value,
                priv_info=priv_info,
                proprio_hist=proprio_hist,
            )
            
            # Update observations - step returns student_input format but info has priv_info
            obs_dict = {
                'obs': next_obs_dict['obs'],
                'priv_info': info.get('priv_info', priv_info),  # Get from info or use previous
                'proprio_hist': next_obs_dict['proprio_hist'],
            }
            
            total_reward += reward.sum().item()
            episode_count += done.sum().item()
        
        # Compute last value
        with torch.no_grad():
            obs = obs_dict['obs']
            proprio_hist = obs_dict['proprio_hist']
            student_output = self.student_policy(obs, proprio_hist=proprio_hist)
            last_value = student_output['value'].squeeze(-1)
        
        # Compute returns
        self.buffer.compute_returns(last_value, self.gamma, self.tau)
        
        self.total_steps += self.buffer.horizon_length * self.env.num_envs
        
        return {
            'reward_mean': total_reward / (self.buffer.horizon_length * self.env.num_envs),
            'episode_count': episode_count,
        }
    
    def update(self) -> Dict[str, float]:
        """Perform PPO + Distillation update.
        
        Returns:
            Loss statistics dictionary.
        """
        # Normalize advantages
        advantages = self.buffer.advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.view(
            self.buffer.horizon_length, self.env.num_envs
        )
        
        total_policy_loss = 0
        total_value_loss = 0
        total_distill_loss = 0
        total_entropy = 0
        update_count = 0
        
        for _ in range(self.mini_epochs):
            batches = self.buffer.get_batches(self.minibatch_size)
            
            for batch_idx, batch in enumerate(batches):
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                proprio_hist = batch['proprio_hist']
                priv_info = batch['priv_info']
                
                # Student forward pass
                student_output = self.student_policy(obs, proprio_hist=proprio_hist)
                action_mean = student_output['action_mean']
                action_std = student_output['action_std']
                value = student_output['value'].squeeze(-1)
                student_latent = student_output['latent']
                
                # Get teacher latent (no grad)
                with torch.no_grad():
                    teacher_output = self.teacher_policy(obs, priv_info=priv_info)
                    teacher_latent = teacher_output['latent']
                
                # Policy loss
                dist = torch.distributions.Normal(action_mean, action_std)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.e_clip, 1 + self.e_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                old_values = batch['values']
                value_clipped = old_values + torch.clamp(value - old_values, -self.e_clip, self.e_clip)
                value_loss_unclipped = (value - returns) ** 2
                value_loss_clipped = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Distillation loss (MSE between student and teacher latents)
                distill_loss = ((student_latent - teacher_latent.detach()) ** 2).mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_loss_coef * value_loss 
                    - self.entropy_coef * entropy
                    + self.distill_coef * distill_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.student_policy.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_distill_loss += distill_loss.item()
                total_entropy += entropy.item()
                update_count += 1
        
        return {
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'distill_loss': total_distill_loss / update_count,
            'entropy': total_entropy / update_count,
        }
    
    def train(self, num_epochs: int, callback=None) -> Dict[str, List[float]]:
        """Train for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs.
            callback: Optional callback function.
            
        Returns:
            Training history dictionary.
        """
        history = {
            'reward_mean': [],
            'policy_loss': [],
            'value_loss': [],
            'distill_loss': [],
            'entropy': [],
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Collect rollout
            rollout_stats = self.collect_rollout()
            
            # Update policy
            update_stats = self.update()
            
            # Record history
            history['reward_mean'].append(rollout_stats['reward_mean'])
            history['policy_loss'].append(update_stats['policy_loss'])
            history['value_loss'].append(update_stats['value_loss'])
            history['distill_loss'].append(update_stats['distill_loss'])
            history['entropy'].append(update_stats['entropy'])
            
            # Callback
            if callback is not None:
                callback(epoch, {**rollout_stats, **update_stats})
            
            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: reward={rollout_stats['reward_mean']:.4f}, "
                    f"policy_loss={update_stats['policy_loss']:.4f}, "
                    f"distill_loss={update_stats['distill_loss']:.4f}"
                )
        
        return history
    
    def save(self, path: str):
        """Save student checkpoint.
        
        Args:
            path: Save path.
        """
        torch.save({
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'student_state_dict': self.student_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved student checkpoint to {path}")
    
    def load(self, path: str):
        """Load student checkpoint.
        
        Args:
            path: Checkpoint path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.total_steps = checkpoint['total_steps']
        self.student_policy.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded student checkpoint from {path}")


def load_teacher_from_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """Load teacher policy from checkpoint.
    
    Args:
        checkpoint_path: Path to teacher checkpoint.
        device: PyTorch device.
        
    Returns:
        Loaded teacher policy.
    """
    from ..algo.models.actor_critic import ProKneePolicy
    from ..envs.constants import ACTIVE_PROSTHESIS_JOINTS, STUDENT_PROPRIO_DIM, TEACHER_PRIV_INFO_DIM, OBS_DIM

    # Create teacher policy (must match training dimensions)
    teacher = ProKneePolicy(
        obs_dim=OBS_DIM,
        action_dim=len(ACTIVE_PROSTHESIS_JOINTS),
        priv_info_dim=TEACHER_PRIV_INFO_DIM,
        proprio_dim=STUDENT_PROPRIO_DIM,
        history_len=30,
        latent_dim=8,
        hidden_dims=[256, 128, 64],
        mode='teacher'
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    teacher.load_state_dict(checkpoint['policy_state_dict'])
    
    return teacher
