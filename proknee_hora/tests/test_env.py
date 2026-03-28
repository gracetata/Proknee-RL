"""
Tests for ProKnee-Hora environments.
"""

import pytest
import sys
import os

# Isaac Gym must be imported before torch
isaacgym = pytest.importorskip("isaacgym", reason="Isaac Gym required for env tests")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proknee_hora.envs import ProKneeBase, ProKneeTeacher, ProKneeStudent
from proknee_hora.envs import (
    ACTIVE_PROSTHESIS_JOINTS, 
    PASSIVE_PROSTHESIS_JOINTS,
    STUDENT_PROPRIO_DIM,
    TEACHER_PRIV_INFO_DIM,
    LATENT_DIM
)


class TestProKneeBase:
    """Tests for ProKneeBase environment."""
    
    def test_init(self):
        """Test environment initialization."""
        env = ProKneeBase(num_envs=16, device='cpu')
        assert env.num_envs == 16
        assert env.num_dofs == 28
        assert env.base_obs_dim == 105
        
    def test_reset(self):
        """Test environment reset."""
        env = ProKneeBase(num_envs=16, device='cpu')
        obs = env.reset()
        
        assert 'obs' in obs
        assert 'priv_info' in obs
        assert 'proprio_hist' in obs
        assert obs['obs'].shape == (16, 105)
        assert obs['priv_info'].shape == (16, 18)
        assert obs['proprio_hist'].shape == (16, 30, 10)
    
    def test_step_full_body(self):
        """Test step with full body action."""
        env = ProKneeBase(num_envs=16, device='cpu', prosthesis_only=False)
        env.reset()
        
        action = torch.randn(16, 28)
        obs, reward, done, info = env.step(action)
        
        assert obs['obs'].shape == (16, 105)
        assert reward.shape == (16,)
        assert done.shape == (16,)
    
    def test_step_prosthesis_only(self):
        """Test step with prosthesis-only action."""
        env = ProKneeBase(num_envs=16, device='cpu', prosthesis_only=True)
        env.reset()
        
        action = torch.randn(16, 1)  # Only knee joint
        obs, reward, done, info = env.step(action)
        
        assert obs['obs'].shape == (16, 105)
        assert reward.shape == (16,)
        

class TestProKneeTeacher:
    """Tests for ProKneeTeacher environment."""
    
    def test_init(self):
        """Test teacher environment initialization."""
        env = ProKneeTeacher(num_envs=16, device='cpu')
        assert env.prosthesis_only == True
        assert env.use_priv_info == True
    
    def test_get_teacher_input(self):
        """Test getting teacher input."""
        env = ProKneeTeacher(num_envs=16, device='cpu')
        env.reset()
        
        teacher_input = env.get_teacher_input()
        
        assert 'obs' in teacher_input
        assert 'priv_info' in teacher_input
        assert teacher_input['obs'].shape == (16, 105)
        assert teacher_input['priv_info'].shape == (16, 18)
    
    def test_step(self):
        """Test teacher step."""
        env = ProKneeTeacher(num_envs=16, device='cpu')
        env.reset()
        
        action = torch.randn(16, 1)
        teacher_input, reward, done, info = env.step(action)
        
        assert 'obs' in teacher_input
        assert 'priv_info' in teacher_input
        assert 'proprio_hist' in info  # For potential distillation


class TestProKneeStudent:
    """Tests for ProKneeStudent environment."""
    
    def test_init(self):
        """Test student environment initialization."""
        env = ProKneeStudent(num_envs=16, device='cpu')
        assert env.prosthesis_only == True
        assert env.use_priv_info == False
    
    def test_get_student_input(self):
        """Test getting student input."""
        env = ProKneeStudent(num_envs=16, device='cpu')
        env.reset()
        
        student_input = env.get_student_input()
        
        assert 'obs' in student_input
        assert 'proprio_hist' in student_input
        assert student_input['obs'].shape == (16, 105)
        assert student_input['proprio_hist'].shape == (16, 30, 10)
    
    def test_step(self):
        """Test student step with distillation info."""
        env = ProKneeStudent(num_envs=16, device='cpu')
        env.reset()
        
        action = torch.randn(16, 1)
        student_input, reward, done, info = env.step(action)
        
        assert 'obs' in student_input
        assert 'proprio_hist' in student_input
        assert 'teacher_latent' in info
        assert info['teacher_latent'].shape == (16, 8)
    
    def test_distillation_loss(self):
        """Test distillation loss computation."""
        env = ProKneeStudent(num_envs=16, device='cpu', distill_coef=1.0)
        
        student_latent = torch.randn(16, 8)
        teacher_latent = torch.randn(16, 8)
        
        loss = env.compute_distillation_loss(student_latent, teacher_latent)
        
        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0


class TestJointConfiguration:
    """Tests for joint configuration constants."""
    
    def test_active_prosthesis_joints(self):
        """Test active prosthesis joint indices."""
        assert ACTIVE_PROSTHESIS_JOINTS == [21]  # Right knee
    
    def test_passive_prosthesis_joints(self):
        """Test passive prosthesis joint indices."""
        assert PASSIVE_PROSTHESIS_JOINTS == [22, 23, 24]  # Right ankle
    
    def test_dimensions(self):
        """Test dimension constants."""
        assert STUDENT_PROPRIO_DIM == 10
        assert TEACHER_PRIV_INFO_DIM == 18
        assert LATENT_DIM == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
