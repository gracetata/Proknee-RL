"""4096-world MuJoCo Warp environment for replay-initialized prosthesis PPO."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Self

import mujoco
import torch

from .config import (
    ReplayAnnealingConfig,
    RewardConfig,
    WarpEnvironmentConfig,
)
from .curriculum import mix_replay_action, replay_beta
from .observations import (
    ACTOR_OBSERVATION_SIZE,
    ObservationHistory,
    actor_frame,
    critic_observation,
    projected_gravity,
    quaternion_rotate_inverse,
)
from .replay import PackedReplay
from .rewards import RewardInputs, compute_reward


def _name_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"MuJoCo model is missing {name!r}")
    return value


class WarpReplayEnv:
    """GPU-resident vector environment.

    MuJoCo Warp is imported lazily so CPU schema/tests do not initialize CUDA.
    The launcher must expose only physical GPU 5, making ``cuda:0`` the single
    visible training device.
    """

    action_size = 4
    actor_observation_size = ACTOR_OBSERVATION_SIZE

    def __init__(
        self,
        model_path: str | Path,
        replay: PackedReplay,
        config: WarpEnvironmentConfig,
        annealing: ReplayAnnealingConfig,
        reward_config: RewardConfig,
        *,
        device: str,
    ) -> None:
        config.validate()
        annealing.validate()
        reward_config.validate()
        self.config = config
        self.annealing = annealing
        self.reward_config = reward_config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("MuJoCo Warp training requires a CUDA device")
        if replay.device != self.device:
            raise ValueError(
                f"replay is on {replay.device}, environment is on {self.device}"
            )
        self.replay = replay
        self.num_envs = config.num_envs
        self.model_path = Path(model_path).resolve()
        self.mj_model = mujoco.MjModel.from_binary_path(str(self.model_path))
        self.mj_model.opt.ccd_iterations = config.ccd_iterations
        if self.mj_model.nq != replay.nq or self.mj_model.nv != replay.nv:
            raise ValueError("MuJoCo model and compact replay dimensions differ")
        if not torch.isclose(
            torch.tensor(self.mj_model.opt.timestep),
            torch.tensor(replay.dt_physics),
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError("MuJoCo model and replay physics timestep differ")

        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as error:
            raise RuntimeError(
                "mujoco-warp and warp-lang are required for WarpReplayEnv"
            ) from error
        self.mjw = mjw
        self.wp = wp
        wp.init()
        warp_device = f"cuda:{self.device.index or 0}"
        wp.set_device(warp_device)
        self.warp_device = warp_device
        with wp.ScopedDevice(warp_device):
            self.model = mjw.put_model(self.mj_model)
            host_data = mujoco.MjData(self.mj_model)
            self.data = mjw.put_data(
                self.mj_model,
                host_data,
                nworld=self.num_envs,
                nconmax=config.nconmax_per_world,
                njmax=config.njmax_per_world,
            )

        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.qacc = wp.to_torch(self.data.qacc)
        self.qacc_warmstart = wp.to_torch(self.data.qacc_warmstart)
        self.qfrc_applied = wp.to_torch(self.data.qfrc_applied)
        self.time = wp.to_torch(self.data.time)
        self.cvel = wp.to_torch(self.data.cvel)
        self.contact_worldid = wp.to_torch(self.data.contact.worldid)
        self.contact_geom = wp.to_torch(self.data.contact.geom)
        self.contact_distance = wp.to_torch(self.data.contact.dist)
        self.nacon = wp.to_torch(self.data.nacon)
        self.nefc = wp.to_torch(self.data.nefc)

        self.torque_limits = torch.tensor(
            config.torque_limits,
            device=self.device,
            dtype=torch.float32,
        )
        self.prosthesis_dofs = replay.prosthesis_dofs
        self.prosthesis_qpos = replay.prosthesis_qpos
        self._configure_model_indices()

        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.frame_indices = torch.zeros_like(self.motion_ids)
        self.episode_steps = torch.zeros_like(self.motion_ids)
        self.episode_limits = torch.zeros_like(self.motion_ids)
        self.previous_policy_action = torch.zeros(
            (self.num_envs, self.action_size), device=self.device
        )
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.history = ObservationHistory(
            self.num_envs,
            device=self.device,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)
        self.agent_steps = 0
        self.beta = 1.0
        self._last_saturation = torch.zeros(
            (self.num_envs, self.action_size), device=self.device
        )
        self._all_mask = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.critic_observation_size = (
            self.actor_observation_size
            + self.mj_model.nq
            + self.mj_model.nv
            + 1
            + 2
            + 1
        )

    def _configure_model_indices(self) -> None:
        left_bodies = {
            _name_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("talus_l", "calcn_l", "toes_l")
        }
        right_bodies = {
            _name_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("talus_r", "calcn_r", "toes_r")
        }
        left_geoms = [
            index
            for index, body in enumerate(self.mj_model.geom_bodyid)
            if int(body) in left_bodies
        ]
        right_geoms = [
            index
            for index, body in enumerate(self.mj_model.geom_bodyid)
            if int(body) in right_bodies
        ]
        if not left_geoms or not right_geoms:
            raise ValueError("could not identify bilateral foot collision geoms")
        self.foot_geom_ids = (
            torch.tensor(left_geoms, device=self.device),
            torch.tensor(right_geoms, device=self.device),
        )
        self.foot_body_ids = torch.tensor(
            (
                _name_id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "calcn_l",
                ),
                _name_id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "calcn_r",
                ),
            ),
            dtype=torch.long,
            device=self.device,
        )
        ranges: list[tuple[float, float]] = []
        for dof in self.prosthesis_dofs.detach().cpu().tolist():
            joint = int(self.mj_model.dof_jntid[dof])
            if not self.mj_model.jnt_limited[joint]:
                ranges.append((-torch.inf, torch.inf))
            else:
                ranges.append(tuple(float(x) for x in self.mj_model.jnt_range[joint]))
        self.prosthesis_ranges = torch.tensor(
            ranges,
            dtype=torch.float32,
            device=self.device,
        )

    def set_agent_steps(self, steps: int) -> None:
        self.agent_steps = int(steps)
        self.beta = float(replay_beta(self.agent_steps, self.annealing))

    def _warp_call(self, function: Any, *args: Any) -> None:
        torch_stream = torch.cuda.current_stream(self.device)
        warp_stream = self.wp.stream_from_torch(torch_stream)
        with self.wp.ScopedStream(warp_stream):
            function(*args)

    def _reference_command(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference_qpos, reference_qvel, _warmstart = self.replay.gather_state(
            self.motion_ids,
            self.frame_indices,
        )
        root_quaternion = reference_qpos[:, 3:7]
        body_linear = quaternion_rotate_inverse(
            root_quaternion,
            reference_qvel[:, :3],
        )
        body_angular = quaternion_rotate_inverse(
            root_quaternion,
            reference_qvel[:, 3:6],
        )
        return body_linear, body_angular[:, 2], reference_qpos[:, 2]

    def _foot_contact(self) -> torch.Tensor:
        capacity = self.contact_worldid.shape[0]
        contact_index = torch.arange(capacity, device=self.device)
        valid = (
            (contact_index < self.nacon[0])
            & (self.contact_distance <= 0.0)
            & (self.contact_worldid >= 0)
            & (self.contact_worldid < self.num_envs)
        )
        result = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=self.device,
        )
        for side, geom_ids in enumerate(self.foot_geom_ids):
            touches = (
                torch.isin(self.contact_geom[:, 0], geom_ids)
                | torch.isin(self.contact_geom[:, 1], geom_ids)
            )
            world_ids = self.contact_worldid[valid & touches].long()
            result[world_ids, side] = True
        return result

    def _root_up_z(self) -> torch.Tensor:
        return -projected_gravity(self.qpos[:, 3:7])[:, 2]

    def _joint_limit_violation(self) -> torch.Tensor:
        values = self.qpos.index_select(1, self.prosthesis_qpos)
        lower = self.prosthesis_ranges[:, 0]
        upper = self.prosthesis_ranges[:, 1]
        span = (upper - lower).clamp_min(1e-6)
        return (
            (lower - values).clamp_min(0.0)
            + (values - upper).clamp_min(0.0)
        ) / span

    def _frame(self) -> torch.Tensor:
        command_velocity, command_yaw_rate, _height = self._reference_command()
        return actor_frame(
            self.qpos,
            self.qvel,
            self.prosthesis_qpos,
            self.prosthesis_dofs,
            command_velocity,
            command_yaw_rate,
            self.previous_policy_action,
        )

    def _observations(
        self,
        foot_contact: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if foot_contact is None:
            foot_contact = self._foot_contact()
        actor = self.history.observation()
        progress = self.episode_steps.float() / self.episode_limits.clamp_min(1).float()
        critic = critic_observation(
            actor,
            self.qpos,
            self.qvel,
            self._root_up_z(),
            foot_contact,
            progress,
        )
        return actor, critic

    def _reset_mask(self, mask: torch.Tensor) -> None:
        selected = mask.nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            return
        selection = self.replay.sample(
            int(selected.numel()),
            self.config.episode_steps,
            generator=self.generator,
            random_start=self.config.random_start,
        )
        self.motion_ids[selected] = selection.motion_ids
        self.frame_indices[selected] = selection.frame_indices
        self.episode_limits[selected] = selection.episode_steps
        self.episode_steps[selected] = 0
        qpos, qvel, warmstart = self.replay.gather_state(
            selection.motion_ids,
            selection.frame_indices,
        )
        self.qpos[selected] = qpos
        self.qvel[selected] = qvel
        self.qacc_warmstart[selected] = warmstart
        self.qacc[selected] = 0.0
        self.qfrc_applied[selected] = 0.0
        self.time[selected] = (
            selection.frame_indices.float() * self.replay.dt_control
        )
        self.previous_policy_action[selected] = 0.0
        self.episode_return[selected] = 0.0
        self._warp_call(self.mjw.forward, self.model, self.data)
        frame = self._frame()
        self.history.reset(mask, frame)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._reset_mask(self._all_mask)
        return self._observations()

    def step(
        self,
        policy_action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if policy_action.shape != (self.num_envs, self.action_size):
            raise ValueError(
                f"actions have shape {tuple(policy_action.shape)}, expected "
                f"({self.num_envs}, {self.action_size})"
            )
        policy_action = policy_action.clamp(-1.0, 1.0)
        previous_action = self.previous_policy_action.clone()
        saturation = torch.zeros_like(policy_action)
        executed_torque = torch.zeros_like(policy_action)
        for substep in range(self.replay.n_substeps):
            recorded = self.replay.gather_force(
                self.motion_ids,
                self.frame_indices,
                substep,
            )
            replay_torque = recorded.index_select(1, self.prosthesis_dofs)
            _executed_action, executed_torque, substep_saturation = (
                mix_replay_action(
                    policy_action,
                    replay_torque,
                    self.torque_limits,
                    self.beta,
                )
            )
            saturation = torch.maximum(saturation, substep_saturation)
            self.qfrc_applied.copy_(recorded)
            self.qfrc_applied[:, self.prosthesis_dofs] = executed_torque
            self._warp_call(self.mjw.step, self.model, self.data)

        self.frame_indices += 1
        self.episode_steps += 1
        command_velocity, command_yaw_rate, target_height = (
            self._reference_command()
        )
        foot_contact = self._foot_contact()
        body_linear = quaternion_rotate_inverse(
            self.qpos[:, 3:7],
            self.qvel[:, :3],
        )
        body_angular = quaternion_rotate_inverse(
            self.qpos[:, 3:7],
            self.qvel[:, 3:6],
        )
        root_up_z = self._root_up_z()
        finite = (
            torch.isfinite(self.qpos).all(dim=1)
            & torch.isfinite(self.qvel).all(dim=1)
            & torch.isfinite(self.qacc).all(dim=1)
        )
        contact_overflow = self.nacon[0] > self.data.naconmax
        constraint_overflow = (self.nefc > self.data.njmax).any()
        overflow = contact_overflow | constraint_overflow
        fell = (
            (self.qpos[:, 2] < self.config.fall_height)
            | (root_up_z < self.config.fall_up_z)
            | (~finite)
            | overflow
        )
        reward, reward_terms = compute_reward(
            RewardInputs(
                projected_gravity=projected_gravity(self.qpos[:, 3:7]),
                root_height=self.qpos[:, 2],
                target_root_height=target_height,
                root_velocity=body_linear,
                command_velocity=command_velocity,
                yaw_rate=body_angular[:, 2],
                command_yaw_rate=command_yaw_rate,
                foot_contact=foot_contact,
                foot_horizontal_speed=torch.linalg.vector_norm(
                    self.cvel.index_select(1, self.foot_body_ids)[:, :, 3:5],
                    dim=-1,
                ),
                prosthesis_torque=executed_torque,
                prosthesis_acceleration=self.qacc.index_select(
                    1,
                    self.prosthesis_dofs,
                ),
                policy_action=policy_action,
                previous_policy_action=previous_action,
                joint_limit_violation=self._joint_limit_violation(),
                fell=fell,
            ),
            self.reward_config,
        )
        self.episode_return += reward
        end_of_episode = self.episode_steps >= self.episode_limits
        done = fell | end_of_episode
        finished = done.nonzero(as_tuple=False).flatten()
        completed_return = self.episode_return.index_select(0, finished).clone()
        completed_fell = fell.index_select(0, finished).clone()
        root_height_metric = self.qpos[:, 2].clone()
        root_up_z_metric = root_up_z.clone()
        self.previous_policy_action.copy_(policy_action)
        frame = self._frame()
        self.history.append(frame)
        _terminal_actor, terminal_critic = self._observations(foot_contact)
        self._last_saturation.copy_(saturation)
        if finished.numel():
            self._reset_mask(done)
        actor_observation, critic_observation_value = self._observations()
        info: dict[str, torch.Tensor] = {
            "time_outs": end_of_episode & (~fell),
            "terminal_critic_observation": terminal_critic,
            "fell": fell,
            "completed_return": completed_return,
            "completed_fell": completed_fell,
            "replay_beta": torch.full(
                (self.num_envs,),
                self.beta,
                device=self.device,
            ),
            "action_saturation": saturation.mean(dim=1),
            "root_height": root_height_metric,
            "root_up_z": root_up_z_metric,
            "buffer_overflow": overflow.expand(self.num_envs),
            **reward_terms,
        }
        return (
            actor_observation,
            critic_observation_value,
            reward,
            done,
            info,
        )

    def close(self) -> None:
        # Warp/PyTorch own the storage; dropping references is sufficient.
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
