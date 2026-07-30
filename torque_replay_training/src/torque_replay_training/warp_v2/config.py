"""Validated configuration for the 4096-world MuJoCo Warp trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReplayAnnealingConfig:
    hold_steps: int = 2_000_000
    end_steps: int = 20_000_000

    def validate(self) -> None:
        if self.hold_steps < 0:
            raise ValueError("replay hold_steps must be non-negative")
        if self.end_steps <= self.hold_steps:
            raise ValueError("replay end_steps must be greater than hold_steps")


@dataclass(frozen=True)
class RewardConfig:
    alive: float = 1.0
    upright: float = 1.0
    root_height: float = 0.5
    root_velocity: float = 1.0
    yaw_rate: float = 0.5
    foot_slip: float = -0.1
    both_feet_airborne: float = -0.25
    torque_l2: float = -2e-6
    joint_acceleration_l2: float = -1e-7
    action_rate_l2: float = -0.006
    joint_limit: float = -1.0
    fall: float = -200.0
    upright_sigma: float = 0.10
    height_sigma: float = 0.05
    velocity_sigma: float = 0.40
    yaw_rate_sigma: float = 0.40

    def validate(self) -> None:
        for name in (
            "upright_sigma",
            "height_sigma",
            "velocity_sigma",
            "yaw_rate_sigma",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fall >= 0:
            raise ValueError("fall reward weight must be negative")


@dataclass(frozen=True)
class WarpEnvironmentConfig:
    num_envs: int = 4096
    episode_steps: int = 512
    history_steps: int = 4
    random_start: bool = True
    nconmax_per_world: int = 96
    njmax_per_world: int = 512
    ccd_iterations: int = 100
    torque_limits: tuple[float, ...] = (260.0, 280.0, 80.0, 15.0)
    fall_height: float = 0.55
    fall_up_z: float = 0.35
    seed: int = 0

    def validate(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if self.history_steps != 4:
            raise ValueError("warp-v2 deployable actor requires exactly four history frames")
        if len(self.torque_limits) != 4 or any(value <= 0 for value in self.torque_limits):
            raise ValueError("four positive torque limits are required")
        if self.nconmax_per_world <= 0 or self.njmax_per_world <= 0:
            raise ValueError("contact and constraint capacities must be positive")
        if self.ccd_iterations <= 0:
            raise ValueError("ccd_iterations must be positive")


@dataclass(frozen=True)
class PPOConfig:
    device: str = "cuda:0"
    max_agent_steps: int = 100_000_000
    horizon_length: int = 32
    minibatch_size: int = 32_768
    mini_epochs: int = 4
    learning_rate: float = 3e-4
    target_kl: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    critic_coefficient: float = 2.0
    entropy_coefficient: float = 5e-4
    max_grad_norm: float = 1.0
    reward_scale: float = 0.01
    initial_log_std: float = -3.5
    save_frequency: int = 25
    actor_units: tuple[int, ...] = (256, 256, 128)
    critic_units: tuple[int, ...] = (512, 256, 128)

    def validate(self, num_envs: int) -> None:
        if self.max_agent_steps <= 0 or self.horizon_length <= 0:
            raise ValueError("training steps and horizon must be positive")
        batch_size = num_envs * self.horizon_length
        if self.minibatch_size <= 0 or batch_size % self.minibatch_size:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by "
                f"minibatch_size={self.minibatch_size}"
            )
        if self.mini_epochs <= 0:
            raise ValueError("mini_epochs must be positive")
        if not self.actor_units or not self.critic_units:
            raise ValueError("actor and critic networks require hidden layers")


@dataclass(frozen=True)
class WarpTrainingConfig:
    environment: WarpEnvironmentConfig = field(default_factory=WarpEnvironmentConfig)
    replay_annealing: ReplayAnnealingConfig = field(
        default_factory=ReplayAnnealingConfig
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self) -> None:
        self.environment.validate()
        self.replay_annealing.validate()
        self.reward.validate()
        self.ppo.validate(self.environment.num_envs)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _tuples(raw: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        if name in raw:
            raw[name] = tuple(raw[name])
    return raw


def load_training_config(path: str | Path) -> WarpTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("training config must be a mapping")
    environment = WarpEnvironmentConfig(
        **_tuples(_section(raw, "environment"), ("torque_limits",))
    )
    annealing = ReplayAnnealingConfig(**_section(raw, "replay_annealing"))
    reward = RewardConfig(**_section(raw, "reward"))
    ppo = PPOConfig(
        **_tuples(
            _section(raw, "ppo"),
            ("actor_units", "critic_units"),
        )
    )
    result = WarpTrainingConfig(
        environment=environment,
        replay_annealing=annealing,
        reward=reward,
        ppo=ppo,
    )
    result.validate()
    return result
