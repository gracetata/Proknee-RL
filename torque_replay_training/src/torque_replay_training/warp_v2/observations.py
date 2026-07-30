"""Deployable actor observations and privileged critic observations."""

from __future__ import annotations

import torch

ACTOR_FRAME_SIZE = 24
ACTOR_HISTORY_STEPS = 4
ACTOR_OBSERVATION_SIZE = ACTOR_FRAME_SIZE * ACTOR_HISTORY_STEPS


def quaternion_rotate_inverse(
    quaternion_wxyz: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame vectors into the quaternion's local frame."""

    quaternion = quaternion_wxyz / quaternion_wxyz.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    # Rotation by conjugate(q), without explicitly constructing a matrix.
    first = vector * (2.0 * scalar.square() - 1.0)
    second = 2.0 * xyz * (xyz * vector).sum(dim=-1, keepdim=True)
    third = -2.0 * scalar * torch.linalg.cross(xyz, vector, dim=-1)
    return first + second + third


def projected_gravity(root_quaternion: torch.Tensor) -> torch.Tensor:
    gravity = torch.zeros(
        (*root_quaternion.shape[:-1], 3),
        dtype=root_quaternion.dtype,
        device=root_quaternion.device,
    )
    gravity[..., 2] = -1.0
    return quaternion_rotate_inverse(root_quaternion, gravity)


def actor_frame(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    prosthesis_qpos_indices: torch.Tensor,
    prosthesis_dof_indices: torch.Tensor,
    command_velocity: torch.Tensor,
    command_yaw_rate: torch.Tensor,
    previous_policy_action: torch.Tensor,
) -> torch.Tensor:
    """Construct one 24-value deployable observation frame."""

    root_quaternion = qpos[:, 3:7]
    body_linear_velocity = quaternion_rotate_inverse(
        root_quaternion,
        qvel[:, :3],
    )
    body_angular_velocity = quaternion_rotate_inverse(
        root_quaternion,
        qvel[:, 3:6],
    )
    command = torch.cat(
        (command_velocity[:, :2], command_yaw_rate.unsqueeze(1)),
        dim=1,
    )
    frame = torch.cat(
        (
            qpos.index_select(1, prosthesis_qpos_indices),
            qvel.index_select(1, prosthesis_dof_indices),
            projected_gravity(root_quaternion),
            body_linear_velocity,
            body_angular_velocity,
            command,
            previous_policy_action,
        ),
        dim=1,
    )
    if frame.shape[1] != ACTOR_FRAME_SIZE:
        raise RuntimeError(
            f"actor frame has {frame.shape[1]} values, expected {ACTOR_FRAME_SIZE}"
        )
    return frame


class ObservationHistory:
    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.frames = torch.zeros(
            (num_envs, ACTOR_HISTORY_STEPS, ACTOR_FRAME_SIZE),
            dtype=dtype,
            device=device,
        )

    def reset(self, mask: torch.Tensor, frame: torch.Tensor) -> None:
        selected = mask.nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            return
        repeated = frame.index_select(0, selected).unsqueeze(1).expand(
            -1,
            ACTOR_HISTORY_STEPS,
            -1,
        )
        self.frames[selected] = repeated

    def append(self, frame: torch.Tensor) -> None:
        self.frames[:, :-1].copy_(self.frames[:, 1:].clone())
        self.frames[:, -1].copy_(frame)

    def observation(self) -> torch.Tensor:
        return self.frames.reshape(self.frames.shape[0], ACTOR_OBSERVATION_SIZE)


def critic_observation(
    actor_observation: torch.Tensor,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    root_up_z: torch.Tensor,
    foot_contact: torch.Tensor,
    episode_progress: torch.Tensor,
) -> torch.Tensor:
    """Build the asymmetric critic input without replay prosthesis torque."""

    return torch.cat(
        (
            actor_observation,
            qpos,
            qvel,
            root_up_z.unsqueeze(1),
            foot_contact.to(qpos.dtype),
            episode_progress.unsqueeze(1),
        ),
        dim=1,
    )
