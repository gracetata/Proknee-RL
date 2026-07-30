"""Observations, rewards, terminations, and metrics for the MJLAB task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from ..warp_v2.observations import actor_frame, projected_gravity
from .action import ReplayProsthesisAction


def replay_action(env: ManagerBasedRlEnv) -> ReplayProsthesisAction:
    term = env.action_manager.get_term("prosthesis")
    if not isinstance(term, ReplayProsthesisAction):
        raise TypeError("prosthesis action term has an unexpected type")
    return term


def root_up_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    return -projected_gravity(env.sim.data.qpos[:, 3:7])[:, 2]


def actor_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = replay_action(env)
    body_linear, body_angular = term.body_velocity()
    del body_linear, body_angular
    return actor_frame(
        env.sim.data.qpos[:, :],
        env.sim.data.qvel[:, :],
        term.prosthesis_qpos,
        term.prosthesis_dofs,
        term.command,
        term.command[:, 2],
        env.action_manager.prev_action,
    )


def critic_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = replay_action(env)
    progress = term.episode_steps.float() / term.episode_limits.clamp_min(1).float()
    return torch.cat(
        (
            actor_observation(env),
            env.sim.data.qpos[:, :],
            env.sim.data.qvel[:, :],
            root_up_z(env).unsqueeze(1),
            progress.unsqueeze(1),
        ),
        dim=1,
    )


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    gravity = projected_gravity(env.sim.data.qpos[:, 3:7])
    error = gravity[:, :2].square().sum(dim=1) + (gravity[:, 2] + 1.0).square()
    return torch.exp(-error / std**2)


def minimum_height(env: ManagerBasedRlEnv, minimum: float, margin: float) -> torch.Tensor:
    return ((env.sim.data.qpos[:, 2] - minimum) / margin).clamp(0.0, 1.0)


def forward_velocity(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    term = replay_action(env)
    body_linear, _ = term.body_velocity()
    error = body_linear[:, 0] - term.command[:, 0]
    return torch.exp(-error.square() / std**2)


def lateral_velocity_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    body_linear, _ = replay_action(env).body_velocity()
    return body_linear[:, 1].square()


def yaw_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    _, body_angular = replay_action(env).body_velocity()
    return body_angular[:, 2].square()


def prosthesis_torque_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return replay_action(env).executed_torque.square().sum(dim=1)


def prosthesis_acceleration_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = replay_action(env)
    return env.sim.data.qacc[:, term.prosthesis_dofs].square().sum(dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=1)


def prosthesis_joint_limit(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = replay_action(env)
    joint_ids = torch.as_tensor(
        env.sim.mj_model.dof_jntid[term.prosthesis_dofs.detach().cpu().numpy()],
        dtype=torch.long,
        device=env.device,
    )
    limited = torch.as_tensor(
        env.sim.mj_model.jnt_limited[joint_ids.detach().cpu().numpy()],
        dtype=torch.bool,
        device=env.device,
    )
    ranges = torch.as_tensor(
        env.sim.mj_model.jnt_range[joint_ids.detach().cpu().numpy()],
        dtype=torch.float32,
        device=env.device,
    )
    position = env.sim.data.qpos[:, term.prosthesis_qpos]
    violation = (ranges[:, 0] - position).clamp_min(0.0) + (
        position - ranges[:, 1]
    ).clamp_min(0.0)
    return (violation * limited).sum(dim=1)


def fallen(env: ManagerBasedRlEnv, minimum_height: float, minimum_up_z: float) -> torch.Tensor:
    finite = torch.isfinite(env.sim.data.qpos[:, :]).all(dim=1) & torch.isfinite(
        env.sim.data.qvel[:, :]
    ).all(dim=1)
    return (
        (env.sim.data.qpos[:, 2] < minimum_height)
        | (root_up_z(env) < minimum_up_z)
        | (~finite)
    )


def replay_exhausted(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = replay_action(env)
    return term.episode_steps >= term.episode_limits


def replay_beta_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    value = replay_action(env).beta
    return torch.full((env.num_envs,), value, device=env.device)


def tracking_error_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    return replay_action(env).max_tracking_error


def root_height_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.sim.data.qpos[:, 2]


def root_up_z_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    return root_up_z(env)
