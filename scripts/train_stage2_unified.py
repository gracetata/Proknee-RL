#!/usr/bin/env python3
"""
Stage 2 Unified: Hora-faithful supervised latent distillation for velocity-conditioned policy.

Uses ProKneeUnifiedEnv with the unified velocity-conditioned body policy (Phase 11+).
Trains ONLY the adapt_tconv module to predict latent from proprio_hist (16D).
Everything else (backbone, actor, critic, priv_mlp) is frozen from Stage 1 Unified.

The Student must infer velocity from proprio_hist — it does NOT receive velocity_cmd as input.
This matches the deployment scenario where the prosthesis has no external velocity signal.

Does NOT modify the original train_stage2.py or train_stage2_multi.py.

Usage:
    # CPU test
    python scripts/train_stage2_unified.py --device cpu --num-envs 4 --max-steps 100

    # GPU training
    PYTHONUNBUFFERED=1 nohup python scripts/train_stage2_unified.py \
        --device cuda:0 --num-envs 4096 \
        > outputs/stage2_unified_train.log 2>&1 &
"""

import os
import sys
import argparse
import datetime
import shutil
import time
import logging
import numpy as np

# Fix numpy deprecations
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

from isaacgym import gymapi  # noqa: F401

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from proknee_hora.envs.proknee_unified import ProKneeUnifiedEnv
from proknee_hora.envs.constants_unified import (
    OBS_DIM_UNIFIED,
    TEACHER_PRIV_INFO_DIM_UNIFIED,
    STUDENT_PROPRIO_DIM_UNIFIED,
    VELOCITY_LEVELS,
)
from proknee_hora.envs.constants import (
    LATENT_DIM, PROPRIO_HISTORY_LEN,
)
from proknee_hora.algo.models.actor_critic import ProKneePolicy
from proknee_hora.algo.models.running_mean_std import RunningMeanStd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('train_stage2_unified')


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  ProKnee Stage 2 Unified: Hora-Faithful Distillation         ║
║  adapt_tconv(proprio_hist_16D) → latent ≈ priv_mlp(priv)    ║
║  Student infers velocity from history — no explicit v_cmd    ║
║  Pure MSE loss — Only adapt_tconv trained                    ║
╚══════════════════════════════════════════════════════════════╝
"""


class AverageScalarMeter:
    """Rolling average meter."""
    def __init__(self, window_size=20000):
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
        return self.sum / len(self.values) if self.values else 0.0


class ProprioAdaptUnified:
    """Stage 2 unified trainer: supervised latent distillation.

    Student learns to predict teacher latent from proprio_hist alone.
    Velocity is implicitly encoded in the proprio_hist patterns.
    """

    def __init__(
        self,
        env: ProKneeUnifiedEnv,
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

        obs_dim = OBS_DIM_UNIFIED  # 16
        priv_dim = TEACHER_PRIV_INFO_DIM_UNIFIED  # 114
        proprio_dim = STUDENT_PROPRIO_DIM_UNIFIED  # 16

        # Build model with unified dimensions
        self.model = ProKneePolicy(
            obs_dim=obs_dim,
            action_dim=env.num_actions,
            priv_info_dim=priv_dim,
            proprio_dim=proprio_dim,
            history_len=PROPRIO_HISTORY_LEN,
            latent_dim=LATENT_DIM,
            hidden_dims=[256, 128, 64],
            mode='student',
        )
        self.model.to(self.device)
        self.model.eval()

        # Running mean/std for obs normalization (frozen from Stage 1)
        self.running_mean_std = RunningMeanStd(obs_dim).to(self.device)
        self.running_mean_std.eval()

        # Running mean/std for proprio_hist normalization (trainable)
        self.sa_mean_std = RunningMeanStd((PROPRIO_HISTORY_LEN, proprio_dim)).to(self.device)
        self.sa_mean_std.train()

        # Output directories
        self.output_dir = output_dir
        self.nn_dir = os.path.join(output_dir, 'checkpoints')
        self.tb_dir = os.path.join(output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.tb_dir)
        except ImportError:
            self.writer = None

        # Freeze all except adapt_tconv
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

        self.optim = torch.optim.Adam(adapt_params, lr=lr)

        # Statistics
        self.agent_steps = 0
        self.best_rewards = -1e6
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.step_reward = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.latent_loss_accum = 0.0
        self.latent_loss_count = 0

    def restore_stage1(self, checkpoint_path: str):
        """Load Stage 1 unified teacher model weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded Stage 1 unified model from {checkpoint_path}")
        if missing:
            logger.info(f"  Missing keys (expected for adapt_tconv): {len(missing)}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected}")

        if 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            logger.info("  Loaded running_mean_std from checkpoint")

    def save(self, name: str):
        """Save checkpoint."""
        weights = {
            'model': self.model.state_dict(),
            'running_mean_std': self.running_mean_std.state_dict(),
            'sa_mean_std': self.sa_mean_std.state_dict(),
            'agent_steps': self.agent_steps,
            'best_rewards': self.best_rewards,
            'mode': 'unified',
        }
        path = f'{name}.pth'
        torch.save(weights, path)
        return path

    def _forward_stage2(self, obs_dict):
        """Stage 2 forward pass."""
        obs = obs_dict['obs']
        priv_info = obs_dict['priv_info']
        proprio_hist = obs_dict['proprio_hist']

        # Student: infer latent from proprio_hist (no velocity input)
        student_latent = self.model.adaptation.adapt_tconv(proprio_hist)

        # Teacher: compute latent from priv_info (includes velocity)
        with torch.no_grad():
            teacher_latent = self.model.adaptation.priv_mlp(priv_info)

        # Actor uses student latent
        action_mean, action_std, value = self.model.actor_critic(obs, student_latent)

        return action_mean, student_latent, teacher_latent

    def train(self):
        """Main training loop."""
        _t = time.time()
        _last_t = time.time()
        _last_log_steps = 0

        obs_dict = self.env.reset()
        self.agent_steps += self.num_actors

        while self.agent_steps <= self.max_agent_steps:
            obs_norm = self.running_mean_std(obs_dict['obs'].detach()).detach()
            hist_norm = self.sa_mean_std(obs_dict['proprio_hist'].detach())

            input_dict = {
                'obs': obs_norm,
                'priv_info': obs_dict['priv_info'],
                'proprio_hist': hist_norm,
            }

            mu, e, e_gt = self._forward_stage2(input_dict)

            loss = ((e - e_gt.detach()) ** 2).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu = mu.detach().clamp(-1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.num_actors

            # Statistics
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

            # TensorBoard logging
            if self.writer and (self.agent_steps - _last_log_steps) >= self.log_interval_steps:
                _last_log_steps = self.agent_steps
                self._log_tensorboard()

            # Save checkpoints
            if self.agent_steps % self.save_interval_steps < self.num_actors:
                step_label = f'{int(self.agent_steps // 1e6)}M'
                self.save(os.path.join(self.nn_dir, f'step_{step_label}'))
                self.save(os.path.join(self.nn_dir, 'last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards and len(self.mean_eps_reward.values) > 100:
                self.save(os.path.join(self.nn_dir, 'best'))
                self.best_rewards = mean_rewards

            # Console output
            all_fps = self.agent_steps / (time.time() - _t + 1e-8)
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
        # Log velocity distribution
        vel_cmd = self.env.velocity_cmd
        for v in VELOCITY_LEVELS:
            frac = (vel_cmd == v).float().mean().item()
            self.writer.add_scalar(f'velocity/frac_{v:.1f}', frac, self.agent_steps)
        self.latent_loss_accum = 0.0
        self.latent_loss_count = 0


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 Unified Latent Distillation")
    p.add_argument("--teacher-ckpt", type=str,
                    default="outputs/humanmimic_policy_knee_ankle_vel/stage1_unified/checkpoints/best.pth",
                    help="Stage 1 unified teacher checkpoint")
    p.add_argument("--body-policy", type=str,
                    default="outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth",
                    help="HumanMimic Unified Stage0 checkpoint")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=int(5e8))
    p.add_argument("--save-interval", type=int, default=int(5e7))
    p.add_argument("--log-interval", type=int, default=int(1e6))
    p.add_argument("--episode-length", type=int, default=300)
    p.add_argument("--vel-switch-prob", type=float, default=0.005)
    p.add_argument("--vel-switch-interval", type=int, default=100)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Stage2 输出根目录（含 checkpoints/、tb/）。默认 outputs/stage2_unified_<时间戳>",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print(BANNER)

    print(f"[Config] device={args.device}, num_envs={args.num_envs}")
    print(f"[Config] lr={args.lr}, max_steps={args.max_steps:.0e}")
    print(f"[Config] body_policy={args.body_policy}")
    print(f"[Config] teacher_ckpt={args.teacher_ckpt}")
    print()

    # Check paths exist
    if not os.path.exists(args.body_policy):
        print(f"ERROR: Body policy not found: {args.body_policy}")
        sys.exit(1)
    if not os.path.exists(args.teacher_ckpt):
        print(f"ERROR: Teacher checkpoint not found: {args.teacher_ckpt}")
        print("Run Stage 1 Unified training first.")
        sys.exit(1)

    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(ROOT, "outputs", f"stage2_unified_{timestamp}")
    else:
        out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Config] output_dir={out_dir}")
    print()

    # Create environment
    print("[1/3] Creating ProKneeUnifiedEnv...")
    t0 = time.time()
    env = ProKneeUnifiedEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=True,
        body_policy_checkpoint=args.body_policy,
        proprio_hist_len=PROPRIO_HISTORY_LEN,
        episode_length=args.episode_length,
        velocity_switch_prob=args.vel_switch_prob,
        velocity_switch_interval=args.vel_switch_interval,
    )
    print(f"  Environment created in {time.time() - t0:.1f}s")

    # Create trainer
    print("[2/3] Creating ProprioAdaptUnified trainer...")
    trainer = ProprioAdaptUnified(
        env=env,
        output_dir=out_dir,
        device=args.device,
        lr=args.lr,
        max_agent_steps=args.max_steps,
        save_interval_steps=args.save_interval,
        log_interval_steps=args.log_interval,
    )

    # Load Stage 1 teacher
    print("[3/3] Loading Stage 1 unified teacher model...")
    trainer.restore_stage1(args.teacher_ckpt)

    n_adapt = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Total params:     {n_total:,}")
    print(f"  Frozen params:    {n_total - n_adapt:,}")
    print(f"  Trainable (tconv): {n_adapt:,}")

    print()
    print("=" * 60)
    print("  Unified Stage 2 Training started!")
    print(f"  Target: {args.max_steps:.0e} agent steps ({args.num_envs} envs)")
    print(f"  TensorBoard: tensorboard --logdir {os.path.join(out_dir, 'tb')}")
    print("=" * 60)
    print()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")

    # Final save
    final_path = trainer.save(os.path.join(trainer.nn_dir, 'final'))
    print(f"\n  Final checkpoint: {final_path}")

    # Copy best to canonical location
    canonical_dir = os.path.join(ROOT, "outputs", "checkpoints", "stage2_unified")
    os.makedirs(canonical_dir, exist_ok=True)
    best_src = os.path.join(trainer.nn_dir, 'best.pth')
    if os.path.exists(best_src):
        shutil.copy2(best_src, os.path.join(canonical_dir, "best.pth"))
        print(f"  Best checkpoint → {canonical_dir}/best.pth")

    print()
    print("=" * 60)
    print("  Unified Stage 2 Training complete!")
    print(f"  Agent steps:   {trainer.agent_steps:,}")
    print(f"  Best reward:   {trainer.best_rewards:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
