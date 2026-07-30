"""Replay-torque exploration initialization without imitation learning."""

from __future__ import annotations

import torch

from .config import ReplayAnnealingConfig


def replay_beta(
    agent_steps: int | torch.Tensor,
    config: ReplayAnnealingConfig,
) -> float | torch.Tensor:
    """Return the replay baseline coefficient.

    The coefficient is one through ``hold_steps``, decreases linearly, and is
    exactly zero at and after ``end_steps``.
    """

    config.validate()
    if isinstance(agent_steps, torch.Tensor):
        steps = agent_steps.to(dtype=torch.float32)
        fraction = (config.end_steps - steps) / (
            config.end_steps - config.hold_steps
        )
        return fraction.clamp(0.0, 1.0)
    steps = int(agent_steps)
    if steps <= config.hold_steps:
        return 1.0
    if steps >= config.end_steps:
        return 0.0
    return (config.end_steps - steps) / (
        config.end_steps - config.hold_steps
    )


def mix_replay_action(
    policy_action: torch.Tensor,
    replay_torque: torch.Tensor,
    torque_limits: torch.Tensor,
    beta: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix replay torque with the PPO action and enforce physical limits.

    ``policy_action`` remains the action stored by PPO. Replay torque affects
    only the environment transition while beta is nonzero.
    """

    if policy_action.shape[-1] != 4 or replay_torque.shape[-1] != 4:
        raise ValueError("policy action and replay torque must end in four values")
    if torque_limits.shape != (4,):
        raise ValueError("torque_limits must have shape [4]")
    replay_action = replay_torque / torque_limits
    mixed_unclipped = replay_action * beta + policy_action
    executed_action = mixed_unclipped.clamp(-1.0, 1.0)
    saturation = (mixed_unclipped.abs() > 1.0).to(policy_action.dtype)
    return executed_action, executed_action * torque_limits, saturation
