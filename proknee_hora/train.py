#!/usr/bin/env python3
"""
ProKnee-Hora Training Script.

Main entry point for training prosthetic knee policies using
Teacher-Student architecture.

Usage:
    # Stage 0: Train full body AMP walking
    python train.py stage=0
    
    # Stage 1: Train teacher with privileged info
    python train.py stage=1 load.body_policy=path/to/stage0.pth
    
    # Stage 2: Train student with distillation
    python train.py stage=2 load.body_policy=path/to/stage0.pth load.teacher=path/to/stage1.pth
"""

import os
import sys
import argparse
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from proknee_hora.envs import ProKneeBase, ProKneeTeacher, ProKneeStudent
from proknee_hora.envs.proknee_teacher import ProKneeTeacherFullBody
from proknee_hora.algo import PPO, TeacherStudentTrainer
from proknee_hora.algo.models import ProKneePolicy
from proknee_hora.utils.logger import setup_logger, TrainingLogger
from proknee_hora.utils.config import save_config


def create_output_dir(cfg: DictConfig) -> str:
    """Create output directory for experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        cfg.output_dir,
        f"stage{cfg.stage}_{cfg.stage_name}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    return output_dir


def train_stage0(cfg: DictConfig, output_dir: str, logger):
    """Stage 0: Train full body AMP walking policy.
    
    Args:
        cfg: Configuration.
        output_dir: Output directory.
        logger: Logger instance.
    """
    logger.info("=" * 50)
    logger.info("Stage 0: Full Body AMP Walking Training")
    logger.info("=" * 50)
    
    # Create environment
    env = ProKneeTeacherFullBody(
        num_envs=cfg.env.num_envs,
        device=cfg.device,
        headless=cfg.headless,
    )
    logger.info(f"Created environment with {env.num_envs} envs")
    
    # Create policy
    policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=env.num_actions,
        priv_info_dim=0,  # No priv info in Stage 0
        proprio_dim=0,
        history_len=0,
        latent_dim=8,
        hidden_dims=cfg.network.actor.units,
        mode='teacher',
        separate_actor_critic=cfg.network.separate,
    ).to(cfg.device)
    logger.info(f"Created policy: {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Create PPO trainer
    trainer = PPO(
        policy=policy,
        env=env,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        tau=cfg.ppo.tau,
        e_clip=cfg.ppo.e_clip,
        entropy_coef=cfg.ppo.entropy_coef,
        value_loss_coef=cfg.ppo.value_loss_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        horizon_length=cfg.ppo.horizon_length,
        minibatch_size=cfg.ppo.minibatch_size,
        mini_epochs=cfg.ppo.mini_epochs,
        device=cfg.device,
        use_priv_info=False,
    )
    
    # Training loop
    training_logger = TrainingLogger(output_dir, "stage0")
    best_reward = float('-inf')
    
    def callback(epoch, stats):
        training_logger.log_metrics(epoch, stats)
        
        nonlocal best_reward
        if stats['reward_mean'] > best_reward:
            best_reward = stats['reward_mean']
            trainer.save(os.path.join(output_dir, "checkpoints", "best.pth"))
        
        if epoch % cfg.train.save_interval == 0:
            trainer.save(os.path.join(output_dir, "checkpoints", f"epoch_{epoch}.pth"))
    
    # Train
    history = trainer.train(cfg.train.max_epochs, callback=callback)
    
    # Save final checkpoint
    trainer.save(os.path.join(output_dir, "checkpoints", "final.pth"))
    logger.info(f"Stage 0 training complete. Best reward: {best_reward:.4f}")
    
    return history


def train_stage1(cfg: DictConfig, output_dir: str, logger):
    """Stage 1: Train teacher with privileged information.
    
    Args:
        cfg: Configuration.
        output_dir: Output directory.
        logger: Logger instance.
    """
    logger.info("=" * 50)
    logger.info("Stage 1: Teacher Policy Training")
    logger.info("=" * 50)
    
    # Load body policy checkpoint
    body_policy_path = cfg.load.body_policy
    if not os.path.exists(body_policy_path):
        raise FileNotFoundError(f"Body policy not found: {body_policy_path}")
    logger.info(f"Loading body policy from: {body_policy_path}")
    
    # Create environment
    env = ProKneeTeacher(
        num_envs=cfg.env.num_envs,
        device=cfg.device,
        headless=cfg.headless,
        body_policy_checkpoint=body_policy_path,
        proprio_hist_len=cfg.env.hora.proprio_hist_len,
    )
    logger.info(f"Created teacher environment with {env.num_envs} envs")
    
    # Create policy
    policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=env.num_actions,  # 1 for knee only
        priv_info_dim=cfg.env.hora.priv_info_dim,
        proprio_dim=cfg.env.hora.proprio_dim,
        history_len=cfg.env.hora.proprio_hist_len,
        latent_dim=cfg.env.hora.latent_dim,
        hidden_dims=cfg.network.actor.units,
        mode='teacher',
    ).to(cfg.device)
    logger.info(f"Created teacher policy: {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Create PPO trainer
    trainer = PPO(
        policy=policy,
        env=env,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        tau=cfg.ppo.tau,
        e_clip=cfg.ppo.e_clip,
        entropy_coef=cfg.ppo.entropy_coef,
        value_loss_coef=cfg.ppo.value_loss_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        horizon_length=cfg.ppo.horizon_length,
        minibatch_size=cfg.ppo.minibatch_size,
        mini_epochs=cfg.ppo.mini_epochs,
        device=cfg.device,
        use_priv_info=True,
        priv_info_dim=cfg.env.hora.priv_info_dim,
    )
    
    # Training loop
    training_logger = TrainingLogger(output_dir, "stage1")
    best_reward = float('-inf')
    
    def callback(epoch, stats):
        training_logger.log_metrics(epoch, stats)
        
        nonlocal best_reward
        if stats['reward_mean'] > best_reward:
            best_reward = stats['reward_mean']
            trainer.save(os.path.join(output_dir, "checkpoints", "best.pth"))
        
        if epoch % cfg.train.save_interval == 0:
            trainer.save(os.path.join(output_dir, "checkpoints", f"epoch_{epoch}.pth"))
    
    # Train
    history = trainer.train(cfg.train.max_epochs, callback=callback)
    
    # Save final checkpoint
    trainer.save(os.path.join(output_dir, "checkpoints", "final.pth"))
    logger.info(f"Stage 1 training complete. Best reward: {best_reward:.4f}")
    
    return history


def train_stage2(cfg: DictConfig, output_dir: str, logger):
    """Stage 2: Train student with distillation.
    
    Args:
        cfg: Configuration.
        output_dir: Output directory.
        logger: Logger instance.
    """
    logger.info("=" * 50)
    logger.info("Stage 2: Student Policy Training (Distillation)")
    logger.info("=" * 50)
    
    # Load checkpoints
    body_policy_path = cfg.load.body_policy
    teacher_path = cfg.load.teacher
    
    if not os.path.exists(body_policy_path):
        raise FileNotFoundError(f"Body policy not found: {body_policy_path}")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    
    logger.info(f"Loading body policy from: {body_policy_path}")
    logger.info(f"Loading teacher from: {teacher_path}")
    
    # Create environment
    env = ProKneeStudent(
        num_envs=cfg.env.num_envs,
        device=cfg.device,
        headless=cfg.headless,
        body_policy_checkpoint=body_policy_path,
        teacher_checkpoint=teacher_path,
        proprio_hist_len=cfg.env.hora.proprio_hist_len,
        distill_coef=cfg.distillation.coef,
    )
    logger.info(f"Created student environment with {env.num_envs} envs")
    
    # Create student policy
    student_policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=env.num_actions,
        priv_info_dim=cfg.env.hora.priv_info_dim,
        proprio_dim=cfg.env.hora.proprio_dim,
        history_len=cfg.env.hora.proprio_hist_len,
        latent_dim=cfg.env.hora.latent_dim,
        hidden_dims=cfg.network.actor.units,
        mode='student',
    ).to(cfg.device)
    
    # Load teacher policy
    teacher_policy = ProKneePolicy(
        obs_dim=env.num_obs,
        action_dim=env.num_actions,
        priv_info_dim=cfg.env.hora.priv_info_dim,
        proprio_dim=cfg.env.hora.proprio_dim,
        history_len=cfg.env.hora.proprio_hist_len,
        latent_dim=cfg.env.hora.latent_dim,
        hidden_dims=cfg.network.actor.units,
        mode='teacher',
    ).to(cfg.device)
    
    # Load teacher weights
    teacher_checkpoint = torch.load(teacher_path, map_location=cfg.device)
    teacher_policy.load_state_dict(teacher_checkpoint['policy_state_dict'])
    
    logger.info(f"Created student policy: {sum(p.numel() for p in student_policy.parameters())} parameters")
    
    # Create Teacher-Student trainer
    trainer = TeacherStudentTrainer(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        env=env,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        tau=cfg.ppo.tau,
        e_clip=cfg.ppo.e_clip,
        entropy_coef=cfg.ppo.entropy_coef,
        value_loss_coef=cfg.ppo.value_loss_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        distill_coef=cfg.distillation.coef,
        horizon_length=cfg.ppo.horizon_length,
        minibatch_size=cfg.ppo.minibatch_size,
        mini_epochs=cfg.ppo.mini_epochs,
        device=cfg.device,
        proprio_dim=cfg.env.hora.proprio_dim,
        history_len=cfg.env.hora.proprio_hist_len,
    )
    
    # Training loop
    training_logger = TrainingLogger(output_dir, "stage2")
    best_reward = float('-inf')
    
    def callback(epoch, stats):
        training_logger.log_metrics(epoch, stats)
        
        nonlocal best_reward
        if stats['reward_mean'] > best_reward:
            best_reward = stats['reward_mean']
            trainer.save(os.path.join(output_dir, "checkpoints", "best.pth"))
        
        if epoch % cfg.train.save_interval == 0:
            trainer.save(os.path.join(output_dir, "checkpoints", f"epoch_{epoch}.pth"))
    
    # Train
    history = trainer.train(cfg.train.max_epochs, callback=callback)
    
    # Save final checkpoint
    trainer.save(os.path.join(output_dir, "checkpoints", "final.pth"))
    logger.info(f"Stage 2 training complete. Best reward: {best_reward:.4f}")
    
    return history


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training entry point."""
    # Setup logging
    logger = setup_logger("proknee_hora")
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create output directory
    output_dir = create_output_dir(cfg)
    logger.info(f"Output directory: {output_dir}")
    
    # Save configuration
    save_config(cfg, os.path.join(output_dir, "config.yaml"))
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Train based on stage
    stage = cfg.stage
    
    if stage == 0:
        history = train_stage0(cfg, output_dir, logger)
    elif stage == 1:
        history = train_stage1(cfg, output_dir, logger)
    elif stage == 2:
        history = train_stage2(cfg, output_dir, logger)
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 0, 1, or 2.")
    
    logger.info("Training complete!")
    return history


if __name__ == "__main__":
    main()
