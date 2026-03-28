"""
Tests for ProKnee-Hora models.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proknee_hora.algo.models.adaptation import (
    PrivilegedMLP, 
    ProprioAdaptTConv, 
    AdaptationModule
)
from proknee_hora.algo.models.actor_critic import (
    ActorCritic,
    ActorCriticSeparate,
    ProKneePolicy
)


class TestPrivilegedMLP:
    """Tests for PrivilegedMLP module."""
    
    def test_forward(self):
        """Test forward pass."""
        module = PrivilegedMLP(input_dim=18, hidden_dims=[128, 64], output_dim=8)
        
        priv_info = torch.randn(32, 18)
        latent = module(priv_info)
        
        assert latent.shape == (32, 8)
        # Output should be bounded by tanh
        assert latent.min() >= -1.0
        assert latent.max() <= 1.0
    
    def test_different_dims(self):
        """Test with different dimensions."""
        module = PrivilegedMLP(input_dim=10, hidden_dims=[64, 32], output_dim=4)
        
        priv_info = torch.randn(16, 10)
        latent = module(priv_info)
        
        assert latent.shape == (16, 4)


class TestProprioAdaptTConv:
    """Tests for ProprioAdaptTConv module."""
    
    def test_forward(self):
        """Test forward pass."""
        module = ProprioAdaptTConv(
            proprio_dim=10,
            history_len=30,
            hidden_channels=32,
            output_dim=8
        )
        
        proprio_hist = torch.randn(32, 30, 10)
        latent = module(proprio_hist)
        
        assert latent.shape == (32, 8)
        # Output should be bounded by tanh
        assert latent.min() >= -1.0
        assert latent.max() <= 1.0
    
    def test_temporal_compression(self):
        """Test that temporal dimension is compressed."""
        module = ProprioAdaptTConv(
            proprio_dim=10,
            history_len=30,
            hidden_channels=32,
            output_dim=8
        )
        
        # Verify internal dimension calculation
        assert module.flat_dim > 0
        
    def test_different_history_lengths(self):
        """Test with different history lengths."""
        for history_len in [10, 30, 50]:
            module = ProprioAdaptTConv(
                proprio_dim=10,
                history_len=history_len,
                hidden_channels=32,
                output_dim=8
            )
            
            proprio_hist = torch.randn(16, history_len, 10)
            latent = module(proprio_hist)
            
            assert latent.shape == (16, 8)


class TestAdaptationModule:
    """Tests for AdaptationModule."""
    
    def test_teacher_mode(self):
        """Test adaptation in teacher mode."""
        module = AdaptationModule(
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            latent_dim=8,
            mode='teacher'
        )
        
        priv_info = torch.randn(32, 18)
        proprio_hist = torch.randn(32, 30, 10)
        
        latent, teacher_latent = module(priv_info, proprio_hist)
        
        assert latent.shape == (32, 8)
        assert teacher_latent.shape == (32, 8)
        # In teacher mode, latent should equal teacher_latent
        assert torch.allclose(latent, teacher_latent)
    
    def test_student_mode(self):
        """Test adaptation in student mode."""
        module = AdaptationModule(
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            latent_dim=8,
            mode='student'
        )
        
        priv_info = torch.randn(32, 18)
        proprio_hist = torch.randn(32, 30, 10)
        
        latent, teacher_latent = module(priv_info, proprio_hist)
        
        assert latent.shape == (32, 8)
        assert teacher_latent.shape == (32, 8)
        # In student mode, latent comes from proprio, teacher_latent from priv_info
        assert not torch.allclose(latent, teacher_latent)
    
    def test_freeze_teacher(self):
        """Test freezing teacher module."""
        module = AdaptationModule(
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            latent_dim=8,
            mode='student'
        )
        
        module.freeze_teacher()
        
        for param in module.priv_mlp.parameters():
            assert param.requires_grad == False


class TestActorCritic:
    """Tests for ActorCritic network."""
    
    def test_forward(self):
        """Test forward pass."""
        network = ActorCritic(
            obs_dim=105,
            action_dim=1,
            latent_dim=8,
            hidden_dims=[256, 128, 64]
        )
        
        obs = torch.randn(32, 105)
        latent = torch.randn(32, 8)
        
        action_mean, action_std, value = network(obs, latent)
        
        assert action_mean.shape == (32, 1)
        assert action_std.shape == (32, 1)
        assert value.shape == (32, 1)
    
    def test_act(self):
        """Test action sampling."""
        network = ActorCritic(obs_dim=105, action_dim=1, latent_dim=8)
        
        obs = torch.randn(32, 105)
        latent = torch.randn(32, 8)
        
        action, log_prob = network.act(obs, latent)
        
        assert action.shape == (32, 1)
        assert log_prob.shape == (32,)
    
    def test_evaluate(self):
        """Test action evaluation."""
        network = ActorCritic(obs_dim=105, action_dim=1, latent_dim=8)
        
        obs = torch.randn(32, 105)
        latent = torch.randn(32, 8)
        action = torch.randn(32, 1)
        
        log_prob, value, entropy = network.evaluate(obs, latent, action)
        
        assert log_prob.shape == (32,)
        assert value.shape == (32,)
        assert entropy.shape == (32,)


class TestProKneePolicy:
    """Tests for ProKneePolicy."""
    
    def test_teacher_mode(self):
        """Test policy in teacher mode."""
        policy = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            latent_dim=8,
            mode='teacher'
        )
        
        obs = torch.randn(32, 105)
        priv_info = torch.randn(32, 18)
        
        output = policy(obs, priv_info=priv_info)
        
        assert 'action_mean' in output
        assert 'action_std' in output
        assert 'value' in output
        assert 'latent' in output
        assert output['action_mean'].shape == (32, 1)
    
    def test_student_mode(self):
        """Test policy in student mode."""
        policy = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            latent_dim=8,
            mode='student'
        )
        
        obs = torch.randn(32, 105)
        proprio_hist = torch.randn(32, 30, 10)
        
        output = policy(obs, proprio_hist=proprio_hist)
        
        assert 'action_mean' in output
        assert 'latent' in output
        assert output['action_mean'].shape == (32, 1)
    
    def test_mode_switch(self):
        """Test switching between teacher and student modes."""
        policy = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            latent_dim=8,
            mode='teacher'
        )
        
        assert policy.mode == 'teacher'
        
        policy.set_mode('student')
        assert policy.mode == 'student'
        
        policy.set_mode('teacher')
        assert policy.mode == 'teacher'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
