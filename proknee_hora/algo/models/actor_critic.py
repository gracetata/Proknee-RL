"""
Actor-Critic networks for ProKnee-Hora.

Supports both shared and separate actor-critic architectures.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from .adaptation import AdaptationModule


class ActorCritic(nn.Module):
    """Actor-Critic network with shared backbone.
    
    Architecture:
        obs + latent -> shared_mlp -> actor_head -> actions
                                   -> critic_head -> value
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0
    ):
        """Initialize ActorCritic.
        
        Args:
            obs_dim: Dimension of observation input.
            action_dim: Dimension of action output.
            latent_dim: Dimension of latent input from adaptation module.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function.
            init_noise_std: Initial standard deviation for action noise.
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        input_dim = obs_dim + latent_dim
        activation_fn = self._get_activation(activation)
        
        # Shared backbone
        backbone_layers = []
        dims = [input_dim] + hidden_dims[:-1]
        for i in range(len(dims) - 1):
            backbone_layers.append(nn.Linear(dims[i], dims[i + 1]))
            backbone_layers.append(activation_fn())
        self.backbone = nn.Sequential(*backbone_layers)
        
        backbone_out_dim = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dims[-1]),
            activation_fn(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dims[-1]),
            activation_fn(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Action noise (learnable log std)
        self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(init_noise_std)))
        
        # Initialize weights
        self._init_weights()
        
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'selu': nn.SELU,
        }
        return activations.get(name, nn.ELU)
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: Observation tensor (B, obs_dim).
            latent: Latent tensor from adaptation module (B, latent_dim).
            
        Returns:
            Tuple of (action_mean, action_std, value).
        """
        # Concatenate observation and latent
        x = torch.cat([obs, latent], dim=-1)
        
        # Shared backbone
        features = self.backbone(x)
        
        # Actor output
        action_mean = self.actor(features)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def act(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action for given observation.
        
        Args:
            obs: Observation tensor.
            latent: Latent tensor.
            deterministic: If True, return mean action.
            
        Returns:
            Tuple of (action, log_prob).
        """
        action_mean, action_std, _ = self.forward(obs, latent)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action for given observation.
        
        Args:
            obs: Observation tensor.
            latent: Latent tensor.
            action: Action tensor to evaluate.
            
        Returns:
            Tuple of (log_prob, value, entropy).
        """
        action_mean, action_std, value = self.forward(obs, latent)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(-1), entropy


class ActorCriticSeparate(nn.Module):
    """Actor-Critic network with separate backbones.
    
    Architecture:
        obs + latent -> actor_mlp -> actions
        obs + latent -> critic_mlp -> value
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        actor_hidden_dims: List[int] = [512, 256, 128],
        critic_hidden_dims: List[int] = [512, 256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0
    ):
        """Initialize ActorCriticSeparate.
        
        Args:
            obs_dim: Dimension of observation input.
            action_dim: Dimension of action output.
            latent_dim: Dimension of latent input.
            actor_hidden_dims: Hidden dimensions for actor.
            critic_hidden_dims: Hidden dimensions for critic.
            activation: Activation function.
            init_noise_std: Initial noise standard deviation.
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        input_dim = obs_dim + latent_dim
        activation_fn = self._get_activation(activation)
        
        # Actor network
        actor_layers = []
        dims = [input_dim] + actor_hidden_dims
        for i in range(len(dims) - 1):
            actor_layers.append(nn.Linear(dims[i], dims[i + 1]))
            actor_layers.append(activation_fn())
        actor_layers.append(nn.Linear(dims[-1], action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        dims = [input_dim] + critic_hidden_dims
        for i in range(len(dims) - 1):
            critic_layers.append(nn.Linear(dims[i], dims[i + 1]))
            critic_layers.append(activation_fn())
        critic_layers.append(nn.Linear(dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise
        self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(init_noise_std)))
        
        self._init_weights()
    
    def _get_activation(self, name: str):
        activations = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'selu': nn.SELU,
        }
        return activations.get(name, nn.ELU)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = torch.cat([obs, latent], dim=-1)
        
        action_mean = self.actor(x)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        value = self.critic(x)
        
        return action_mean, action_std, value
    
    def act(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action for given observation."""
        action_mean, action_std, _ = self.forward(obs, latent)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action."""
        action_mean, action_std, value = self.forward(obs, latent)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(-1), entropy


class ProKneePolicy(nn.Module):
    """Complete policy for ProKnee with adaptation module.
    
    Combines adaptation module (Teacher/Student) with Actor-Critic network.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        priv_info_dim: int = 113,
        proprio_dim: int = 10,
        history_len: int = 30,
        latent_dim: int = 8,
        hidden_dims: List[int] = [512, 256, 128],
        mode: str = 'teacher',
        separate_actor_critic: bool = False
    ):
        """Initialize ProKneePolicy.
        
        Args:
            obs_dim: Dimension of base observation.
            action_dim: Dimension of action (1 for knee only).
            priv_info_dim: Dimension of privileged information.
            proprio_dim: Dimension of proprioceptive input per timestep.
            history_len: Number of history timesteps.
            latent_dim: Dimension of latent representation.
            hidden_dims: Hidden dimensions for actor-critic.
            mode: 'teacher' or 'student'.
            separate_actor_critic: If True, use separate networks.
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mode = mode
        
        # Adaptation module
        self.adaptation = AdaptationModule(
            priv_info_dim=priv_info_dim,
            proprio_dim=proprio_dim,
            history_len=history_len,
            latent_dim=latent_dim,
            mode=mode
        )
        
        # Actor-Critic network
        if separate_actor_critic:
            self.actor_critic = ActorCriticSeparate(
                obs_dim=obs_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                actor_hidden_dims=hidden_dims,
                critic_hidden_dims=hidden_dims
            )
        else:
            self.actor_critic = ActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims
            )
    
    def forward(
        self,
        obs: torch.Tensor,
        priv_info: Optional[torch.Tensor] = None,
        proprio_hist: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: Base observation (B, obs_dim).
            priv_info: Privileged information (B, priv_info_dim).
            proprio_hist: Proprioceptive history (B, history_len, proprio_dim).
            
        Returns:
            Dictionary with action_mean, action_std, value, latent, teacher_latent.
        """
        # Get latent from adaptation module
        latent, teacher_latent = self.adaptation(priv_info, proprio_hist)
        
        # Get action and value from actor-critic
        action_mean, action_std, value = self.actor_critic(obs, latent)
        
        return {
            'action_mean': action_mean,
            'action_std': action_std,
            'value': value,
            'latent': latent,
            'teacher_latent': teacher_latent
        }
    
    def act(
        self,
        obs: torch.Tensor,
        priv_info: Optional[torch.Tensor] = None,
        proprio_hist: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action for given inputs."""
        latent, _ = self.adaptation(priv_info, proprio_hist)
        return self.actor_critic.act(obs, latent, deterministic)
    
    def set_mode(self, mode: str):
        """Set policy mode (teacher/student)."""
        self.mode = mode
        self.adaptation.set_mode(mode)
    
    def freeze_teacher(self):
        """Freeze teacher module for student training."""
        self.adaptation.freeze_teacher()
    
    def freeze_actor_critic(self):
        """Freeze actor-critic for distillation-only training."""
        for param in self.actor_critic.parameters():
            param.requires_grad = False
