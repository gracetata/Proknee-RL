"""
ProprioAdaptTConv and PrivilegedMLP modules for Teacher-Student architecture.

Based on the Hora architecture:
- PrivilegedMLP: Encodes privileged information to latent (Teacher)
- ProprioAdaptTConv: Encodes proprioceptive history to latent (Student)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class PrivilegedMLP(nn.Module):
    """MLP encoder for privileged information (Teacher).
    
    Encodes privileged information (GRF, contacts, terrain, etc.) into
    a low-dimensional latent representation.
    
    Input: (batch, priv_info_dim)  e.g., (B, 18)
    Output: (batch, latent_dim)    e.g., (B, 8)
    """
    
    def __init__(
        self,
        input_dim: int = 113,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 8,
        activation: str = 'elu'
    ):
        """Initialize PrivilegedMLP.
        
        Args:
            input_dim: Dimension of privileged information input.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Dimension of latent output.
            activation: Activation function ('elu', 'relu', 'tanh').
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        activation_fn = self._get_activation(activation)
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(activation_fn())
        
        self.mlp = nn.Sequential(*layers)
        
        # Final tanh to bound latent
        self.final_activation = nn.Tanh()
        
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'selu': nn.SELU,
        }
        return activations.get(name, nn.ELU)
    
    def forward(self, priv_info: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            priv_info: Privileged information tensor (B, priv_info_dim).
            
        Returns:
            Latent tensor (B, output_dim).
        """
        x = self.mlp(priv_info)
        return self.final_activation(x)


class ProprioAdaptTConv(nn.Module):
    """Temporal Convolution adapter for proprioceptive history (Student).
    
    Compresses proprioceptive history using 1D convolutions to extract
    temporal patterns and produce a latent representation.
    
    Based on Hora architecture:
    - Input: (batch, history_len, proprio_dim) e.g., (B, 30, 10)
    - Output: (batch, latent_dim) e.g., (B, 8)
    
    Dimension reduction path:
    (B, 30, 10) → channel_transform → (B, 30, 32)
    → TConv1d(k=9,s=2) → (B, 11, 32)
    → TConv1d(k=5,s=1) → (B, 7, 32)
    → TConv1d(k=3,s=1) → (B, 3, 32)
    → flatten → (B, 96)
    → Linear → (B, 8)
    """
    
    def __init__(
        self,
        proprio_dim: int = 10,
        history_len: int = 30,
        hidden_channels: int = 32,
        output_dim: int = 8,
        activation: str = 'elu'
    ):
        """Initialize ProprioAdaptTConv.
        
        Args:
            proprio_dim: Dimension of proprioceptive input per timestep.
            history_len: Number of history timesteps.
            hidden_channels: Number of channels in conv layers.
            output_dim: Dimension of latent output.
            activation: Activation function.
        """
        super().__init__()
        
        self.proprio_dim = proprio_dim
        self.history_len = history_len
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        
        activation_fn = self._get_activation(activation)
        
        # Channel transform: (B, T, proprio_dim) -> (B, T, hidden_channels)
        self.channel_transform = nn.Sequential(
            nn.Linear(proprio_dim, hidden_channels),
            activation_fn()
        )
        
        # Temporal convolutions: operate on (B, C, T) format
        # Conv1d: (B, hidden_channels, T) -> compress temporal dimension
        self.tconv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9, stride=2, padding=4)
        self.tconv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=2)
        self.tconv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        
        self.activation = activation_fn()
        
        # Calculate flattened dimension after convolutions
        # Input: T=30 -> after tconv1 (k=9,s=2,p=4): (30+2*4-9)/2+1 = 15
        # After tconv2 (k=5,s=1,p=2): 15
        # After tconv3 (k=3,s=1,p=1): 15
        # But we want to compress more, so adjust padding
        self._calc_output_size()
        
        # Final projection to latent
        self.low_dim_proj = nn.Linear(self.flat_dim, output_dim)
        
        # Final tanh to bound latent
        self.final_activation = nn.Tanh()
        
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'elu': nn.ELU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'selu': nn.SELU,
        }
        return activations.get(name, nn.ELU)
    
    def _calc_output_size(self):
        """Calculate output size after convolutions."""
        # Simulate forward pass to get dimensions
        with torch.no_grad():
            x = torch.zeros(1, self.history_len, self.proprio_dim)
            x = self.channel_transform(x)  # (1, T, C)
            x = x.permute(0, 2, 1)  # (1, C, T)
            x = self.activation(self.tconv1(x))
            x = self.activation(self.tconv2(x))
            x = self.activation(self.tconv3(x))
            self.flat_dim = x.flatten(1).shape[1]
    
    def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            proprio_hist: Proprioceptive history tensor (B, history_len, proprio_dim).
            
        Returns:
            Latent tensor (B, output_dim).
        """
        # Channel transform
        x = self.channel_transform(proprio_hist)  # (B, T, C)
        
        # Permute for Conv1d: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # Temporal convolutions
        x = self.activation(self.tconv1(x))
        x = self.activation(self.tconv2(x))
        x = self.activation(self.tconv3(x))
        
        # Flatten and project to latent
        x = x.flatten(1)
        x = self.low_dim_proj(x)
        
        return self.final_activation(x)


class AdaptationModule(nn.Module):
    """Combined adaptation module supporting both Teacher and Student modes.
    
    In Teacher mode: Uses PrivilegedMLP
    In Student mode: Uses ProprioAdaptTConv
    """
    
    def __init__(
        self,
        priv_info_dim: int = 113,
        proprio_dim: int = 10,
        history_len: int = 30,
        latent_dim: int = 8,
        hidden_dims: List[int] = [256, 128],
        hidden_channels: int = 32,
        mode: str = 'teacher'
    ):
        """Initialize AdaptationModule.
        
        Args:
            priv_info_dim: Dimension of privileged info (teacher).
            proprio_dim: Dimension of proprioceptive input (student).
            history_len: Number of history timesteps (student).
            latent_dim: Dimension of latent output.
            hidden_dims: Hidden dimensions for PrivilegedMLP.
            hidden_channels: Hidden channels for ProprioAdaptTConv.
            mode: 'teacher' or 'student'.
        """
        super().__init__()
        
        self.mode = mode
        self.latent_dim = latent_dim
        
        # Teacher module (privileged info encoder)
        self.priv_mlp = PrivilegedMLP(
            input_dim=priv_info_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim
        )
        
        # Student module (proprioceptive history encoder)
        self.adapt_tconv = ProprioAdaptTConv(
            proprio_dim=proprio_dim,
            history_len=history_len,
            hidden_channels=hidden_channels,
            output_dim=latent_dim
        )
    
    def forward(
        self,
        priv_info: torch.Tensor = None,
        proprio_hist: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            priv_info: Privileged information (B, priv_info_dim).
            proprio_hist: Proprioceptive history (B, history_len, proprio_dim).
            
        Returns:
            Tuple of (active_latent, teacher_latent).
            - In teacher mode: (teacher_latent, teacher_latent)
            - In student mode: (student_latent, teacher_latent) for distillation
        """
        teacher_latent = None
        student_latent = None
        
        if priv_info is not None:
            teacher_latent = self.priv_mlp(priv_info)
            
        if proprio_hist is not None:
            student_latent = self.adapt_tconv(proprio_hist)
        
        if self.mode == 'teacher':
            return teacher_latent, teacher_latent
        else:
            return student_latent, teacher_latent
    
    def set_mode(self, mode: str):
        """Set adaptation mode.
        
        Args:
            mode: 'teacher' or 'student'.
        """
        assert mode in ['teacher', 'student']
        self.mode = mode
        
    def freeze_teacher(self):
        """Freeze teacher module parameters (for student training)."""
        for param in self.priv_mlp.parameters():
            param.requires_grad = False
            
    def unfreeze_teacher(self):
        """Unfreeze teacher module parameters."""
        for param in self.priv_mlp.parameters():
            param.requires_grad = True
