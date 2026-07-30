"""Replay-force and four-DoF prosthesis action term for official MJLAB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from ..compact_schema import CompactTorqueReplayDataset
from ..warp_v2.observations import quaternion_rotate_inverse
from ..warp_v2.replay import PackedReplay

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ReplayProsthesisActionCfg(ActionTermCfg):
    replay_paths: tuple[str, ...]
    episode_steps: int = 512
    random_start: bool = True
    torque_limits: tuple[float, float, float, float] = (260.0, 280.0, 80.0, 15.0)
    replay_hold_agent_steps: int = 2_000_000
    replay_end_agent_steps: int = 20_000_000
    fixed_replay_beta: float | None = None
    target_forward_velocity: float = 1.0
    healthy_kp: float = 0.0
    healthy_kd: float = 0.0
    healthy_correction_limit: float = 0.0
    root_position_kp: float = 0.0
    root_velocity_kd: float = 0.0
    root_force_limit: float = 0.0
    root_orientation_kp: float = 0.0
    root_angular_velocity_kd: float = 0.0
    root_torque_limit: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "ReplayProsthesisAction":
        return ReplayProsthesisAction(self, env)


class ReplayProsthesisAction(ActionTerm):
    """Inject recorded generalized forces and replace four prosthesis entries.

    The initial action prior is ``beta * replay + (1-beta) * policy``. At
    ``beta=1`` the stochastic policy cannot perturb exact replay. Replay torque
    is not used as a behavior-cloning target or reward term.
    """

    cfg: ReplayProsthesisActionCfg

    def __init__(self, cfg: ReplayProsthesisActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.replay = PackedReplay.load(cfg.replay_paths, device=self.device)
        if env.sim.mj_model.nq != self.replay.nq or env.sim.mj_model.nv != self.replay.nv:
            raise ValueError(
                "MJLAB model/replay dimensions differ: "
                f"model=({env.sim.mj_model.nq},{env.sim.mj_model.nv}), "
                f"replay=({self.replay.nq},{self.replay.nv})"
            )
        if not np.isclose(
            env.cfg.sim.mujoco.timestep,
            self.replay.dt_physics,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("MJLAB physics timestep differs from replay data")
        if env.cfg.decimation != self.replay.n_substeps:
            raise ValueError("MJLAB decimation differs from replay substep count")

        first = CompactTorqueReplayDataset.load(Path(cfg.replay_paths[0]))
        prosthesis = set(first.metadata["prosthesis_dof_indices"])
        scalar_pairs = [
            (int(row["dofadr"]), int(row["qposadr"]))
            for row in first.metadata["joint_metadata"]
            if int(row["nq"]) == 1
            and int(row["nv"]) == 1
            and int(row["dofadr"]) not in prosthesis
        ]
        self.healthy_dofs = torch.tensor(
            [row[0] for row in scalar_pairs],
            dtype=torch.long,
            device=self.device,
        )
        self.healthy_qpos = torch.tensor(
            [row[1] for row in scalar_pairs],
            dtype=torch.long,
            device=self.device,
        )
        self.prosthesis_dofs = self.replay.prosthesis_dofs
        self.prosthesis_qpos = self.replay.prosthesis_qpos
        self.torque_limits = torch.tensor(
            cfg.torque_limits,
            dtype=torch.float32,
            device=self.device,
        )
        self._raw_actions = torch.zeros((self.num_envs, 4), device=self.device)
        self.executed_actions = torch.zeros_like(self._raw_actions)
        self.executed_torque = torch.zeros_like(self._raw_actions)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_indices = torch.zeros_like(self.motion_ids)
        self.episode_steps = torch.zeros_like(self.motion_ids)
        self.episode_limits = torch.zeros_like(self.motion_ids)
        self.episode_start = torch.zeros_like(self.motion_ids)
        self.command = torch.zeros((self.num_envs, 3), device=self.device)
        self.command[:, 0] = cfg.target_forward_velocity
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(env.cfg.seed or 0))
        self.beta = 1.0
        self.max_tracking_error = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def _beta(self) -> float:
        if self.cfg.fixed_replay_beta is not None:
            return float(self.cfg.fixed_replay_beta)
        steps = int(self._env.common_step_counter) * self.num_envs
        if steps <= self.cfg.replay_hold_agent_steps:
            return 1.0
        if steps >= self.cfg.replay_end_agent_steps:
            return 0.0
        span = self.cfg.replay_end_agent_steps - self.cfg.replay_hold_agent_steps
        return 1.0 - (steps - self.cfg.replay_hold_agent_steps) / span

    def process_actions(self, actions: torch.Tensor) -> None:
        qpos_ref, _, _ = self.replay.gather_state(
            self.motion_ids,
            self.frame_indices,
        )
        error = torch.linalg.vector_norm(
            self._env.sim.data.qpos[:, :] - qpos_ref,
            dim=1,
        )
        self.max_tracking_error[:] = torch.maximum(self.max_tracking_error, error)
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self.beta = self._beta()

    def _reference_state(self, substep: int) -> tuple[torch.Tensor, torch.Tensor]:
        qpos0, qvel0, _ = self.replay.gather_state(
            self.motion_ids, self.frame_indices
        )
        qpos1, qvel1, _ = self.replay.gather_state(
            self.motion_ids, self.frame_indices + 1
        )
        alpha = float(substep) / float(self.replay.n_substeps)
        return (1.0 - alpha) * qpos0 + alpha * qpos1, (
            (1.0 - alpha) * qvel0 + alpha * qvel1
        )

    def _add_optional_tracking_stabilizer(
        self,
        recorded: torch.Tensor,
        substep: int,
    ) -> None:
        qpos_ref, qvel_ref = self._reference_state(substep)
        data = self._env.sim.data
        assistance_scale = float(self.beta)
        if self.cfg.healthy_kp or self.cfg.healthy_kd:
            correction = (
                self.cfg.healthy_kp
                * (
                    qpos_ref.index_select(1, self.healthy_qpos)
                    - data.qpos[:, self.healthy_qpos]
                )
                + self.cfg.healthy_kd
                * (
                    qvel_ref.index_select(1, self.healthy_dofs)
                    - data.qvel[:, self.healthy_dofs]
                )
            )
            correction = correction.clamp(
                -self.cfg.healthy_correction_limit,
                self.cfg.healthy_correction_limit,
            )
            recorded[:, self.healthy_dofs] += assistance_scale * correction
        if self.cfg.root_position_kp or self.cfg.root_velocity_kd:
            root_force = (
                self.cfg.root_position_kp * (qpos_ref[:, :3] - data.qpos[:, :3])
                + self.cfg.root_velocity_kd * (qvel_ref[:, :3] - data.qvel[:, :3])
            ).clamp(-self.cfg.root_force_limit, self.cfg.root_force_limit)
            recorded[:, :3] += assistance_scale * root_force
        if self.cfg.root_orientation_kp or self.cfg.root_angular_velocity_kd:
            current = data.qpos[:, 3:7]
            reference = qpos_ref[:, 3:7]
            # Relative quaternion current^-1 * reference; its vector part is
            # the small-angle orientation error in the root/body frame.
            current_w = current[:, :1]
            current_xyz = current[:, 1:]
            reference_w = reference[:, :1]
            reference_xyz = reference[:, 1:]
            error_w = current_w * reference_w + (
                current_xyz * reference_xyz
            ).sum(dim=1, keepdim=True)
            error_xyz = (
                current_w * reference_xyz
                - reference_w * current_xyz
                - torch.linalg.cross(current_xyz, reference_xyz, dim=1)
            )
            error_xyz = 2.0 * torch.where(
                error_w < 0.0,
                -error_xyz,
                error_xyz,
            )
            root_torque = (
                self.cfg.root_orientation_kp * error_xyz
                + self.cfg.root_angular_velocity_kd
                * (qvel_ref[:, 3:6] - data.qvel[:, 3:6])
            ).clamp(-self.cfg.root_torque_limit, self.cfg.root_torque_limit)
            recorded[:, 3:6] += assistance_scale * root_torque

    def apply_actions(self) -> None:
        substep = (int(self._env._sim_step_counter) - 1) % self.replay.n_substeps
        recorded = self.replay.gather_force(
            self.motion_ids,
            self.frame_indices,
            substep,
        ).clone()
        self._add_optional_tracking_stabilizer(recorded, substep)
        replay_torque = recorded.index_select(1, self.prosthesis_dofs)
        replay_action = replay_torque / self.torque_limits
        self.executed_actions[:] = (
            self.beta * replay_action + (1.0 - self.beta) * self._raw_actions
        ).clamp(-1.0, 1.0)
        self.executed_torque[:] = self.executed_actions * self.torque_limits
        recorded[:, self.prosthesis_dofs] = self.executed_torque
        self._env.sim.data.qfrc_applied[:] = recorded

        if substep == self.replay.n_substeps - 1:
            self.frame_indices += 1
            self.episode_steps += 1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            selected = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            selected = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            selected = env_ids
        if selected.numel() == 0:
            return
        selection = self.replay.sample(
            int(selected.numel()),
            self.cfg.episode_steps,
            generator=self.generator,
            random_start=self.cfg.random_start,
        )
        self.motion_ids[selected] = selection.motion_ids
        self.frame_indices[selected] = selection.frame_indices
        self.episode_start[selected] = selection.frame_indices
        self.episode_steps[selected] = 0
        self.episode_limits[selected] = selection.episode_steps
        qpos, qvel, warmstart = self.replay.gather_state(
            selection.motion_ids,
            selection.frame_indices,
        )
        data = self._env.sim.data
        data.qpos[selected] = qpos
        data.qvel[selected] = qvel
        data.qacc_warmstart[selected] = warmstart
        data.qacc[selected] = 0.0
        data.qfrc_applied[selected] = 0.0
        data.time[selected] = (
            selection.frame_indices.float() * self.replay.dt_control
        )
        self._raw_actions[selected] = 0.0
        self.executed_actions[selected] = 0.0
        self.executed_torque[selected] = 0.0
        self.max_tracking_error[selected] = 0.0

    def body_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self._env.sim.data.qpos
        qvel = self._env.sim.data.qvel
        return (
            quaternion_rotate_inverse(qpos[:, 3:7], qvel[:, :3]),
            quaternion_rotate_inverse(qpos[:, 3:7], qvel[:, 3:6]),
        )
