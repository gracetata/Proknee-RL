"""
Tests for ProKnee-Hora training components.
"""

import pytest
import sys
import os

# Isaac Gym must be imported before torch
isaacgym = pytest.importorskip("isaacgym", reason="Isaac Gym required for training tests")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proknee_hora.algo.ppo import PPO, RolloutBuffer
from proknee_hora.algo.teacher_student import TeacherStudentTrainer
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.envs import ProKneeBase, ProKneeTeacher, ProKneeStudent


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""
    
    def test_init(self):
        """Test buffer initialization."""
        buffer = RolloutBuffer(
            num_envs=16,
            horizon_length=8,
            obs_dim=105,
            action_dim=1,
            device='cpu'
        )
        
        assert buffer.obs.shape == (8, 16, 105)
        assert buffer.actions.shape == (8, 16, 1)
        assert buffer.rewards.shape == (8, 16)
    
    def test_add(self):
        """Test adding transitions."""
        buffer = RolloutBuffer(
            num_envs=16,
            horizon_length=8,
            obs_dim=105,
            action_dim=1,
            device='cpu'
        )
        
        for _ in range(8):
            buffer.add(
                obs=torch.randn(16, 105),
                action=torch.randn(16, 1),
                log_prob=torch.randn(16),
                reward=torch.randn(16),
                done=torch.zeros(16),
                value=torch.randn(16)
            )
        
        assert buffer.step == 8
    
    def test_compute_returns(self):
        """Test GAE computation."""
        buffer = RolloutBuffer(
            num_envs=16,
            horizon_length=8,
            obs_dim=105,
            action_dim=1,
            device='cpu'
        )
        
        for _ in range(8):
            buffer.add(
                obs=torch.randn(16, 105),
                action=torch.randn(16, 1),
                log_prob=torch.randn(16),
                reward=torch.ones(16),
                done=torch.zeros(16),
                value=torch.randn(16)
            )
        
        last_value = torch.randn(16)
        buffer.compute_returns(last_value, gamma=0.99, tau=0.95)
        
        assert buffer.advantages.shape == (8, 16)
        assert buffer.returns.shape == (8, 16)
    
    def test_get_batches(self):
        """Test minibatch generation."""
        buffer = RolloutBuffer(
            num_envs=16,
            horizon_length=8,
            obs_dim=105,
            action_dim=1,
            device='cpu'
        )
        
        for _ in range(8):
            buffer.add(
                obs=torch.randn(16, 105),
                action=torch.randn(16, 1),
                log_prob=torch.randn(16),
                reward=torch.ones(16),
                done=torch.zeros(16),
                value=torch.randn(16)
            )
        
        buffer.compute_returns(torch.randn(16), gamma=0.99, tau=0.95)
        batches = buffer.get_batches(minibatch_size=32)
        
        assert len(batches) == (8 * 16) // 32
        assert batches[0]['obs'].shape == (32, 105)


class TestPPOIntegration:
    """Integration tests for PPO with environment."""
    
    def test_ppo_init(self):
        """Test PPO initialization."""
        env = ProKneeBase(num_envs=16, device='cpu', prosthesis_only=True)
        policy = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            mode='teacher'
        )
        
        ppo = PPO(
            policy=policy,
            env=env,
            horizon_length=8,
            minibatch_size=32,
            device='cpu',
            use_priv_info=True,
            priv_info_dim=18
        )
        
        assert ppo.buffer.horizon_length == 8
    
    def test_collect_rollout(self):
        """Test rollout collection."""
        env = ProKneeBase(num_envs=16, device='cpu', prosthesis_only=True)
        policy = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            mode='teacher'
        )
        
        ppo = PPO(
            policy=policy,
            env=env,
            horizon_length=4,
            minibatch_size=16,
            device='cpu',
            use_priv_info=True,
            priv_info_dim=18
        )
        
        stats = ppo.collect_rollout()
        
        assert 'reward_mean' in stats
        assert ppo.buffer.step == 4  # Buffer filled with horizon_length steps
    
    def test_single_update(self):
        """Test single PPO update."""
        env = ProKneeBase(num_envs=16, device='cpu', prosthesis_only=True)
        policy = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            mode='teacher'
        )
        
        ppo = PPO(
            policy=policy,
            env=env,
            horizon_length=4,
            minibatch_size=16,
            mini_epochs=2,
            device='cpu',
            use_priv_info=True,
            priv_info_dim=18
        )
        
        ppo.collect_rollout()
        stats = ppo.update()
        
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats


class TestTeacherStudentIntegration:
    """Integration tests for Teacher-Student training."""
    
    def test_trainer_init(self):
        """Test trainer initialization."""
        env = ProKneeStudent(num_envs=16, device='cpu')
        
        student = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            mode='student'
        )
        
        teacher = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            mode='teacher'
        )
        
        trainer = TeacherStudentTrainer(
            student_policy=student,
            teacher_policy=teacher,
            env=env,
            horizon_length=4,
            minibatch_size=16,
            device='cpu',
            proprio_dim=10,
            history_len=30
        )
        
        # Teacher should be frozen
        for param in teacher.parameters():
            assert param.requires_grad == False
    
    def test_distillation_update(self):
        """Test update with distillation loss."""
        env = ProKneeStudent(num_envs=16, device='cpu')
        
        student = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            mode='student'
        )
        
        teacher = ProKneePolicy(
            obs_dim=105,
            action_dim=1,
            priv_info_dim=18,
            proprio_dim=10,
            history_len=30,
            mode='teacher'
        )
        
        trainer = TeacherStudentTrainer(
            student_policy=student,
            teacher_policy=teacher,
            env=env,
            horizon_length=4,
            minibatch_size=16,
            mini_epochs=2,
            distill_coef=1.0,
            device='cpu',
            proprio_dim=10,
            history_len=30
        )
        
        trainer.collect_rollout()
        stats = trainer.update()
        
        assert 'policy_loss' in stats
        assert 'distill_loss' in stats
        assert stats['distill_loss'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
