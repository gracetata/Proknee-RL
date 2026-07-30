"""Packed GPU representation of compact-v1 torque replay trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..compact_schema import CompactTorqueReplayDataset


@dataclass(frozen=True)
class ReplaySelection:
    motion_ids: torch.Tensor
    frame_indices: torch.Tensor
    episode_steps: torch.Tensor


class PackedReplay:
    """Compact trajectories concatenated once and kept on the training device."""

    def __init__(
        self,
        *,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        qacc_warmstart: torch.Tensor,
        qfrc: torch.Tensor,
        state_offsets: torch.Tensor,
        torque_offsets: torch.Tensor,
        lengths: torch.Tensor,
        dt_control: float,
        dt_physics: float,
        n_substeps: int,
        prosthesis_dofs: torch.Tensor,
        prosthesis_qpos: torch.Tensor,
        motion_paths: tuple[str, ...],
    ) -> None:
        self.qpos = qpos
        self.qvel = qvel
        self.qacc_warmstart = qacc_warmstart
        self.qfrc = qfrc
        self.state_offsets = state_offsets
        self.torque_offsets = torque_offsets
        self.lengths = lengths
        self.dt_control = float(dt_control)
        self.dt_physics = float(dt_physics)
        self.n_substeps = int(n_substeps)
        self.prosthesis_dofs = prosthesis_dofs
        self.prosthesis_qpos = prosthesis_qpos
        self.motion_paths = motion_paths
        self.device = qpos.device
        self.n_motions = int(lengths.numel())
        self.nq = int(qpos.shape[1])
        self.nv = int(qvel.shape[1])

    @classmethod
    def load(
        cls,
        paths: Sequence[str | Path],
        *,
        device: str | torch.device,
    ) -> PackedReplay:
        resolved = tuple(Path(path).resolve() for path in paths)
        if not resolved:
            raise ValueError("at least one compact replay is required")
        datasets = [CompactTorqueReplayDataset.load(path) for path in resolved]
        first = datasets[0]
        lengths = np.asarray([dataset.n_steps for dataset in datasets], dtype=np.int64)
        state_counts = lengths + 1
        state_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(state_counts)[:-1])
        )
        torque_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(lengths)[:-1])
        )
        for dataset in datasets:
            if (
                dataset.nq != first.nq
                or dataset.nv != first.nv
                or dataset.n_substeps != first.n_substeps
            ):
                raise ValueError("all compact trajectories must have identical dimensions")
            if not np.isclose(dataset.dt_control, first.dt_control):
                raise ValueError("all compact trajectories must have identical control dt")
            if not np.isclose(dataset.dt_physics, first.dt_physics):
                raise ValueError("all compact trajectories must have identical physics dt")
            if (
                dataset.metadata["prosthesis_dof_indices"]
                != first.metadata["prosthesis_dof_indices"]
                or dataset.metadata["prosthesis_qpos_indices"]
                != first.metadata["prosthesis_qpos_indices"]
            ):
                raise ValueError("prosthesis indices differ between trajectories")

        # float32 is the native MJWarp state type. The source compact files
        # remain untouched float64 CPU truth.
        qpos = np.concatenate(
            [dataset.rollout_qpos.astype(np.float32) for dataset in datasets],
            axis=0,
        )
        qvel = np.concatenate(
            [dataset.rollout_qvel.astype(np.float32) for dataset in datasets],
            axis=0,
        )
        warmstart = np.concatenate(
            [dataset.qacc_warmstart.astype(np.float32) for dataset in datasets],
            axis=0,
        )
        qfrc = np.concatenate(
            [dataset.qfrc_actuator.astype(np.float32) for dataset in datasets],
            axis=0,
        )
        target = torch.device(device)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value).to(target)

        return cls(
            qpos=tensor(qpos),
            qvel=tensor(qvel),
            qacc_warmstart=tensor(warmstart),
            qfrc=tensor(qfrc),
            state_offsets=tensor(state_offsets),
            torque_offsets=tensor(torque_offsets),
            lengths=tensor(lengths),
            dt_control=first.dt_control,
            dt_physics=first.dt_physics,
            n_substeps=first.n_substeps,
            prosthesis_dofs=torch.tensor(
                first.metadata["prosthesis_dof_indices"],
                dtype=torch.long,
                device=target,
            ),
            prosthesis_qpos=torch.tensor(
                first.metadata["prosthesis_qpos_indices"],
                dtype=torch.long,
                device=target,
            ),
            motion_paths=tuple(dataset.motion_path for dataset in datasets),
        )

    def sample(
        self,
        num_envs: int,
        episode_steps: int,
        *,
        generator: torch.Generator | None = None,
        random_start: bool = True,
    ) -> ReplaySelection:
        motion_ids = torch.randint(
            self.n_motions,
            (num_envs,),
            device=self.device,
            generator=generator,
        )
        motion_lengths = self.lengths.index_select(0, motion_ids)
        maximum_start = (motion_lengths - min(episode_steps, int(self.lengths.max()))).clamp_min(0)
        if random_start:
            uniform = torch.rand(
                num_envs,
                device=self.device,
                generator=generator,
            )
            starts = torch.floor(uniform * (maximum_start + 1)).to(torch.long)
        else:
            starts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        available = torch.minimum(
            motion_lengths - starts,
            torch.full_like(motion_lengths, int(episode_steps)),
        )
        return ReplaySelection(
            motion_ids=motion_ids,
            frame_indices=starts,
            episode_steps=available,
        )

    def state_indices(
        self,
        motion_ids: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_offsets.index_select(0, motion_ids) + frame_indices

    def torque_indices(
        self,
        motion_ids: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.torque_offsets.index_select(0, motion_ids) + frame_indices

    def gather_state(
        self,
        motion_ids: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.state_indices(motion_ids, frame_indices)
        return (
            self.qpos.index_select(0, indices),
            self.qvel.index_select(0, indices),
            self.qacc_warmstart.index_select(0, indices),
        )

    def gather_force(
        self,
        motion_ids: torch.Tensor,
        frame_indices: torch.Tensor,
        substep: int,
    ) -> torch.Tensor:
        if not 0 <= substep < self.n_substeps:
            raise IndexError(substep)
        indices = self.torque_indices(motion_ids, frame_indices)
        return self.qfrc.index_select(0, indices)[:, substep]

    def mean_prosthesis_torque(
        self,
        motion_ids: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        indices = self.torque_indices(motion_ids, frame_indices)
        force = self.qfrc.index_select(0, indices)
        return force.index_select(2, self.prosthesis_dofs).mean(dim=1)
