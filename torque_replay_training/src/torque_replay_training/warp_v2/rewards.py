"""Task reward with no replay-torque or imitation term."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import RewardConfig


@dataclass(frozen=True)
class RewardInputs:
    projected_gravity: torch.Tensor
    root_height: torch.Tensor
    target_root_height: torch.Tensor
    root_velocity: torch.Tensor
    command_velocity: torch.Tensor
    yaw_rate: torch.Tensor
    command_yaw_rate: torch.Tensor
    foot_contact: torch.Tensor
    foot_horizontal_speed: torch.Tensor
    prosthesis_torque: torch.Tensor
    prosthesis_acceleration: torch.Tensor
    policy_action: torch.Tensor
    previous_policy_action: torch.Tensor
    joint_limit_violation: torch.Tensor
    fell: torch.Tensor


def compute_reward(
    inputs: RewardInputs,
    config: RewardConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute raw per-environment reward and individually logged components."""

    config.validate()
    up_error_sq = (
        inputs.projected_gravity[:, :2].square().sum(dim=-1)
        + (inputs.projected_gravity[:, 2] + 1.0).square()
    )
    upright = torch.exp(-up_error_sq / config.upright_sigma**2)
    height_error = inputs.root_height - inputs.target_root_height
    root_height = torch.exp(
        -height_error.square() / config.height_sigma**2
    )
    velocity_error = inputs.root_velocity - inputs.command_velocity
    root_velocity = torch.exp(
        -velocity_error.square().sum(dim=-1) / config.velocity_sigma**2
    )
    yaw_error = inputs.yaw_rate - inputs.command_yaw_rate
    yaw_rate = torch.exp(-yaw_error.square() / config.yaw_rate_sigma**2)
    foot_slip = (
        inputs.foot_contact.to(inputs.foot_horizontal_speed.dtype)
        * inputs.foot_horizontal_speed.square()
    ).sum(dim=-1)
    both_airborne = (~inputs.foot_contact.any(dim=-1)).to(
        inputs.root_height.dtype
    )
    torque_l2 = inputs.prosthesis_torque.square().sum(dim=-1)
    acceleration_l2 = inputs.prosthesis_acceleration.square().sum(dim=-1)
    action_rate_l2 = (
        inputs.policy_action - inputs.previous_policy_action
    ).square().mean(dim=-1)
    joint_limit = inputs.joint_limit_violation.sum(dim=-1)
    alive = (~inputs.fell).to(inputs.root_height.dtype)
    fall = inputs.fell.to(inputs.root_height.dtype)

    weighted = {
        "reward_alive": config.alive * alive,
        "reward_upright": config.upright * upright,
        "reward_root_height": config.root_height * root_height,
        "reward_root_velocity": config.root_velocity * root_velocity,
        "reward_yaw_rate": config.yaw_rate * yaw_rate,
        "penalty_foot_slip": config.foot_slip * foot_slip,
        "penalty_both_feet_airborne": (
            config.both_feet_airborne * both_airborne
        ),
        "penalty_torque_l2": config.torque_l2 * torque_l2,
        "penalty_joint_acceleration_l2": (
            config.joint_acceleration_l2 * acceleration_l2
        ),
        "penalty_action_rate_l2": config.action_rate_l2 * action_rate_l2,
        "penalty_joint_limit": config.joint_limit * joint_limit,
        "penalty_fall": config.fall * fall,
    }
    total = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
    return total, weighted
