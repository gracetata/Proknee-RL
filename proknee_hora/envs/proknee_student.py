"""
ProKnee Student Environment.

Student environment with limited sensor input for Stage 2 training (distillation).
"""

# Isaac Gym MUST be imported before torch
from isaacgym import gymapi  # noqa: F401 — ensure native libs loaded first

import torch
from typing import Dict, Tuple, Optional
from .proknee_base import ProKneeBase


class ProKneeStudent(ProKneeBase):
    """Student environment with limited proprioceptive input.
    
    The student only sees:
      - Left knee position / velocity  (2D)
      - Left hip positions / velocities (6D)
      - Left foot Fz                   (1D)
      - Command velocity               (1D)
    → 10-D per timestep × 30 frames history.
    
    A frozen teacher provides latent supervision for distillation.
    """
    
    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda:0',
        headless: bool = True,
        body_policy_checkpoint: Optional[str] = None,
        teacher_checkpoint: Optional[str] = None,
        proprio_hist_len: int = 30,
        distill_coef: float = 1.0,
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
        self.use_priv_info = False
        self.distill_coef = distill_coef

        # Load teacher for distillation supervision
        self.teacher = None
        if teacher_checkpoint:
            self._load_teacher(teacher_checkpoint)
    
    def _load_teacher(self, checkpoint_path: str):
        """Load teacher model for distillation."""
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
        
        from ..algo.teacher_student import load_teacher_from_checkpoint
        self.teacher = load_teacher_from_checkpoint(checkpoint_path, self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        print(f"[ProKneeStudent] Loaded teacher from: {checkpoint_path}")
    
    def get_student_input(self) -> Dict[str, torch.Tensor]:
        """Get inputs for student policy (obs + proprio_hist)."""
        observations = self.get_observations()
        return {
            'obs': observations['obs'],
            'proprio_hist': observations['proprio_hist'],
        }
    
    def get_teacher_latent(self, priv_info: torch.Tensor) -> torch.Tensor:
        """Get teacher latent for distillation supervision."""
        if self.teacher is None:
            return torch.zeros(self.num_envs, 8, device=self.device)
        with torch.no_grad():
            output = self.teacher(
                torch.zeros(priv_info.shape[0], self.num_obs, device=self.device),
                priv_info=priv_info,
            )
            return output['latent']
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Step with student action. Returns student_input + teacher info in info dict."""
        obs, rewards, dones, info = super().step(action)
        
        student_input = {
            'obs': obs['obs'],
            'proprio_hist': obs['proprio_hist'],
        }
        info['priv_info'] = obs['priv_info']
        info['teacher_latent'] = self.get_teacher_latent(obs['priv_info'])
        return student_input, rewards, dones, info
    
    def compute_distillation_loss(
        self,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor
    ) -> torch.Tensor:
        """MSE distillation loss with stop-gradient on teacher."""
        loss = torch.mean((student_latent - teacher_latent.detach()) ** 2)
        return self.distill_coef * loss


class ProKneeStudentEval(ProKneeStudent):
    """Student environment for evaluation (no teacher required)."""
    
    def __init__(self, *args, **kwargs):
        kwargs['teacher_checkpoint'] = None
        super().__init__(*args, **kwargs)
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        obs, rewards, dones, info = ProKneeBase.step(self, action)
        student_input = {
            'obs': obs['obs'],
            'proprio_hist': obs['proprio_hist'],
        }
        return student_input, rewards, dones, info
