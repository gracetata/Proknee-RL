"""
PPO Algorithm for ProKnee-Hora.

Implements Proximal Policy Optimization for training prosthetic knee policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(
        self,
        num_envs: int,
        horizon_length: int,
        obs_dim: int,
        action_dim: int,
        device: str = 'cuda:0',
        priv_info_dim: int = 0,
        proprio_dim: int = 0,
        history_len: int = 0,
    ):
        """Initialize rollout buffer.
        
        Args:
            num_envs: Number of parallel environments.
            horizon_length: Number of steps per rollout.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            device: PyTorch device.
            priv_info_dim: Privileged info dimension (0 if not used).
            proprio_dim: Proprioceptive dimension per timestep.
            history_len: Proprioceptive history length.
        """
        self.num_envs = num_envs
        self.horizon_length = horizon_length
        self.device = device
        
        # Core buffers
        self.obs = torch.zeros(horizon_length, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon_length, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(horizon_length, num_envs, device=device)
        self.rewards = torch.zeros(horizon_length, num_envs, device=device)
        self.dones = torch.zeros(horizon_length, num_envs, device=device)
        self.values = torch.zeros(horizon_length, num_envs, device=device)
        
        # Optional buffers for teacher-student
        self.priv_info = None
        self.proprio_hist = None
        self.latents = None
        self.teacher_latents = None
        
        if priv_info_dim > 0:
            self.priv_info = torch.zeros(horizon_length, num_envs, priv_info_dim, device=device)
        
        if proprio_dim > 0 and history_len > 0:
            self.proprio_hist = torch.zeros(horizon_length, num_envs, history_len, proprio_dim, device=device)
        
        # Computed advantages
        self.advantages = torch.zeros(horizon_length, num_envs, device=device)
        self.returns = torch.zeros(horizon_length, num_envs, device=device)
        
        self.step = 0
        
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        priv_info: Optional[torch.Tensor] = None,
        proprio_hist: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        teacher_latent: Optional[torch.Tensor] = None,
    ):
        """Add a transition to the buffer."""
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.values[self.step] = value
        
        if priv_info is not None and self.priv_info is not None:
            self.priv_info[self.step] = priv_info
        
        if proprio_hist is not None and self.proprio_hist is not None:
            self.proprio_hist[self.step] = proprio_hist
        
        self.step += 1
        
    def compute_returns(self, last_value: torch.Tensor, gamma: float, tau: float):
        """Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for the last state.
            gamma: Discount factor.
            tau: GAE lambda parameter.
        """
        last_gae = 0
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * tau * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns = self.advantages + self.values
        
    def get_batches(self, minibatch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get minibatches for training.
        
        Args:
            minibatch_size: Size of each minibatch.
            
        Returns:
            List of batch dictionaries.
        """
        total_size = self.horizon_length * self.num_envs
        indices = torch.randperm(total_size, device=self.device)
        
        batches = []
        for start in range(0, total_size, minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]
            
            # Convert to 2D indices
            t_idx = batch_indices // self.num_envs
            e_idx = batch_indices % self.num_envs
            
            batch = {
                'obs': self.obs[t_idx, e_idx],
                'actions': self.actions[t_idx, e_idx],
                'log_probs': self.log_probs[t_idx, e_idx],
                'advantages': self.advantages[t_idx, e_idx],
                'returns': self.returns[t_idx, e_idx],
                'values': self.values[t_idx, e_idx],
            }
            
            if self.priv_info is not None:
                batch['priv_info'] = self.priv_info[t_idx, e_idx]
            
            if self.proprio_hist is not None:
                batch['proprio_hist'] = self.proprio_hist[t_idx, e_idx]
            
            batches.append(batch)
        
        return batches
    
    def reset(self):
        """Reset buffer for new rollout."""
        self.step = 0


class PPO:
    """Proximal Policy Optimization algorithm."""
    
    def __init__(
        self,
        policy,
        env,
        # PPO hyperparameters
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.95,
        e_clip: float = 0.2,
        entropy_coef: float = 0.0,
        value_loss_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        # Rollout settings
        horizon_length: int = 16,
        minibatch_size: int = 16384,
        mini_epochs: int = 4,
        # Device
        device: str = 'cuda:0',
        # Teacher-student
        use_priv_info: bool = False,
        priv_info_dim: int = 0,
        proprio_dim: int = 0,
        history_len: int = 0,
    ):
        """Initialize PPO algorithm.
        
        Args:
            policy: Policy network.
            env: Environment.
            learning_rate: Learning rate.
            gamma: Discount factor.
            tau: GAE lambda.
            e_clip: PPO clip parameter.
            entropy_coef: Entropy bonus coefficient.
            value_loss_coef: Value loss coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            horizon_length: Steps per rollout.
            minibatch_size: Minibatch size.
            mini_epochs: Number of PPO epochs per update.
            device: PyTorch device.
            use_priv_info: Whether to use privileged information.
            priv_info_dim: Privileged info dimension.
            proprio_dim: Proprioceptive dimension.
            history_len: History length for student.
        """
        self.policy = policy
        self.env = env
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.e_clip = e_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.mini_epochs = mini_epochs
        self.minibatch_size = minibatch_size
        
        # Teacher-student settings
        self.use_priv_info = use_priv_info
        
        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_envs=env.num_envs,
            horizon_length=horizon_length,
            obs_dim=env.num_obs,
            action_dim=env.num_actions,
            device=device,
            priv_info_dim=priv_info_dim if use_priv_info else 0,
            proprio_dim=proprio_dim,
            history_len=history_len,
        )
        
        # Statistics
        self.epoch = 0
        self.total_steps = 0
        
    def collect_rollout(self) -> Dict[str, float]:
        """Collect rollout data from environment.
        
        Returns:
            Statistics dictionary.
        """
        self.buffer.reset()
        
        # Get initial observations
        if hasattr(self.env, 'get_teacher_input'):
            obs_dict = self.env.get_teacher_input()
        elif hasattr(self.env, 'get_student_input'):
            obs_dict = self.env.get_student_input()
        else:
            obs_dict = self.env.get_observations()
        
        total_reward = 0
        episode_count = 0
        ep_len_sum = 0.0
        
        for step in range(self.buffer.horizon_length):
            obs = obs_dict['obs']
            priv_info = obs_dict.get('priv_info', None)
            proprio_hist = obs_dict.get('proprio_hist', None)
            
            # Get action from policy
            with torch.no_grad():
                if self.use_priv_info:
                    output = self.policy(obs, priv_info=priv_info)
                else:
                    output = self.policy(obs, proprio_hist=proprio_hist)
                
                action_mean = output['action_mean']
                action_std = output['action_std']
                value = output['value'].squeeze(-1)
                
                # Sample action
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Step environment
            next_obs_dict, reward, done, info = self.env.step(action)
            
            # Store transition
            self.buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                priv_info=priv_info,
                proprio_hist=proprio_hist,
            )
            
            # Update observations
            obs_dict = {
                'obs': next_obs_dict['obs'] if isinstance(next_obs_dict, dict) else next_obs_dict,
            }
            if 'priv_info' in next_obs_dict:
                obs_dict['priv_info'] = next_obs_dict['priv_info']
            if 'proprio_hist' in next_obs_dict:
                obs_dict['proprio_hist'] = next_obs_dict['proprio_hist']
            
            total_reward += reward.sum().item()
            episode_count += done.sum().item()
            if 'finished_ep_len' in info:
                ep_len_sum += info['finished_ep_len'].sum().item()
        
        # Compute last value for bootstrapping
        with torch.no_grad():
            obs = obs_dict['obs']
            priv_info = obs_dict.get('priv_info', None)
            proprio_hist = obs_dict.get('proprio_hist', None)
            
            if self.use_priv_info:
                output = self.policy(obs, priv_info=priv_info)
            else:
                output = self.policy(obs, proprio_hist=proprio_hist)
            last_value = output['value'].squeeze(-1)
        
        # Compute returns and advantages
        self.buffer.compute_returns(last_value, self.gamma, self.tau)
        
        # Update step count
        self.total_steps += self.buffer.horizon_length * self.env.num_envs
        
        return {
            'reward_mean': total_reward / (self.buffer.horizon_length * self.env.num_envs),
            'episode_count': episode_count,
            'avg_ep_len': ep_len_sum / max(episode_count, 1),
        }
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update.
        
        Returns:
            Loss statistics dictionary.
        """
        # Normalize advantages
        advantages = self.buffer.advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.view(self.buffer.horizon_length, self.env.num_envs)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        for _ in range(self.mini_epochs):
            batches = self.buffer.get_batches(self.minibatch_size)
            
            for batch in batches:
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                
                priv_info = batch.get('priv_info', None)
                proprio_hist = batch.get('proprio_hist', None)
                
                # Forward pass
                if self.use_priv_info:
                    output = self.policy(obs, priv_info=priv_info)
                else:
                    output = self.policy(obs, proprio_hist=proprio_hist)
                
                action_mean = output['action_mean']
                action_std = output['action_std']
                value = output['value'].squeeze(-1)
                
                # Policy loss
                dist = torch.distributions.Normal(action_mean, action_std)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.e_clip, 1 + self.e_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((value - returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1
        
        return {
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count,
        }
    
    def train(self, num_epochs: int, callback=None) -> Dict[str, List[float]]:
        """Train for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs.
            callback: Optional callback function called each epoch.
            
        Returns:
            Training history dictionary.
        """
        history = {
            'reward_mean': [],
            'policy_loss': [],
            'value_loss': [],
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
            history['entropy'].append(update_stats['entropy'])
            
            # Callback
            if callback is not None:
                callback(epoch, {**rollout_stats, **update_stats})
            
            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: reward={rollout_stats['reward_mean']:.4f}, "
                    f"policy_loss={update_stats['policy_loss']:.4f}, "
                    f"value_loss={update_stats['value_loss']:.4f}"
                )
        
        return history
    
    def save(self, path: str):
        """Save policy checkpoint.
        
        Args:
            path: Save path.
        """
        torch.save({
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load policy checkpoint.
        
        Args:
            path: Checkpoint path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.total_steps = checkpoint['total_steps']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
