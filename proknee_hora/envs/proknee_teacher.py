"""
ProKnee Teacher Environment.

Teacher environment with privileged information for Stage 1 training.
"""

# Isaac Gym MUST be imported before torch
from isaacgym import gymapi  # noqa: F401 — ensure native libs loaded first

import torch
from typing import Dict, Tuple, Optional
from .proknee_base import ProKneeBase


class ProKneeTeacher(ProKneeBase):
    """Teacher environment with privileged information.
    
    In Stage 1 training, the teacher has access to privileged information
    (GRF, contacts, orientation error, velocity error) and controls only
    the left knee joint (1 DOF).  Other joints use the frozen Stage-0 body
    policy; left ankle joints are locked.
    """
    
    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda:0',
        headless: bool = True,
        body_policy_checkpoint: Optional[str] = None,
        proprio_hist_len: int = 30,
        **kwargs
    ):
        super().__init__(
            num_envs=num_envs,
            device=device,
            headless=headless,
            prosthesis_only=True,
            freeze_body=True,
            body_policy_checkpoint=body_policy_checkpoint,
            proprio_hist_len=proprio_hist_len,
            **kwargs
        )
        self.use_priv_info = True
        
    def get_teacher_input(self) -> Dict[str, torch.Tensor]:
        """Get inputs for teacher policy (obs + priv_info)."""
        observations = self.get_observations()
        return {
            'obs': observations['obs'],
            'priv_info': observations['priv_info'],
        }
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Step with teacher action (1-D knee action)."""
        obs, rewards, dones, info = super().step(action)
        
        teacher_input = {
            'obs': obs['obs'],
            'priv_info': obs['priv_info'],
        }
        info['proprio_hist'] = obs['proprio_hist']
        return teacher_input, rewards, dones, info


class ProKneeTeacherFullBody(ProKneeBase):
    """Full body environment for Stage 0 training (28-DOF, no freezing)."""
    
    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda:0',
        headless: bool = True,
        **kwargs
    ):
        super().__init__(
            num_envs=num_envs,
            device=device,
            headless=headless,
            prosthesis_only=False,
            freeze_body=False,
            **kwargs
        )


# Test code
if __name__ == "__main__":
    print("Testing ProKneeTeacher environment...")
    
    # Test Teacher environment
    env = ProKneeTeacher(
        num_envs=16,
        device='cpu',
    )
    
    # Test reset
    obs = env.reset()
    teacher_input = env.get_teacher_input()
    print(f"Teacher input shapes:")
    print(f"  obs: {teacher_input['obs'].shape}")
    print(f"  priv_info: {teacher_input['priv_info'].shape}")
    
    # Test step
    action = torch.randn(16, 1)  # Knee action only
    teacher_input, reward, done, info = env.step(action)
    print(f"\nAfter step:")
    print(f"  reward: {reward.mean():.4f}")
    print(f"  proprio_hist in info: {'proprio_hist' in info}")
    
    print("\nProKneeTeacher test passed!")
