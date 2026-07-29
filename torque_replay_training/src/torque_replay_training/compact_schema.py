"""Lossless compact schema containing only MuJoCo replay/training state."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import SCHEMA_VERSION, TorqueReplayDataset


COMPACT_SCHEMA_VERSION = 1
REQUIRED_METADATA_KEYS = (
    "joint_metadata",
    "joint_names",
    "prosthesis_dof_indices",
    "prosthesis_joint_names",
    "prosthesis_qpos_indices",
    "root_dof_indices",
)


def _compact_metadata(source: TorqueReplayDataset) -> dict[str, Any]:
    metadata = {key: source.metadata[key] for key in REQUIRED_METADATA_KEYS}
    metadata.update(
        {
            "compact_schema_version": COMPACT_SCHEMA_VERSION,
            "source_schema_version": int(source.metadata["schema_version"]),
            "nq": source.nq,
            "nv": source.nv,
            "actual_steps": source.n_steps,
            "qualification_fall_height": float(
                source.metadata.get("qualification_fall_height", 0.55)
            ),
            "qualification_fall_up_z": float(
                source.metadata.get("qualification_fall_up_z", 0.35)
            ),
        }
    )
    return metadata


@dataclass
class CompactTorqueReplayDataset:
    """Minimal lossless data required by torque replay and HORA PPO."""

    motion_path: str
    dt_control: float
    dt_physics: float
    rollout_qpos: np.ndarray
    rollout_qvel: np.ndarray
    qacc_warmstart: np.ndarray
    qfrc_actuator: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_steps(self) -> int:
        return int(self.qfrc_actuator.shape[0])

    @property
    def n_substeps(self) -> int:
        return int(self.qfrc_actuator.shape[1])

    @property
    def nq(self) -> int:
        return int(self.rollout_qpos.shape[1])

    @property
    def nv(self) -> int:
        return int(self.rollout_qvel.shape[1])

    @property
    def qfrc_actuator_mean(self) -> np.ndarray:
        return self.qfrc_actuator.mean(axis=1)

    def validate(self) -> None:
        t = self.n_steps
        if t <= 0:
            raise ValueError("compact dataset has no control steps")
        if self.qfrc_actuator.ndim != 3:
            raise ValueError("qfrc_actuator must have shape [T,S,nv]")
        if self.qfrc_actuator.shape[2] != self.nv:
            raise ValueError("qfrc_actuator and rollout_qvel dimensions differ")
        if self.rollout_qpos.shape != (t + 1, self.nq):
            raise ValueError("rollout_qpos must have shape [T+1,nq]")
        if self.rollout_qvel.shape != (t + 1, self.nv):
            raise ValueError("rollout_qvel must have shape [T+1,nv]")
        if self.qacc_warmstart.shape != (t + 1, self.nv):
            raise ValueError("qacc_warmstart must have shape [T+1,nv]")
        if not np.array_equal(self.qacc_warmstart[0], np.zeros(self.nv)):
            raise ValueError("qacc_warmstart[0] must be exactly zero")
        for name in (
            "rollout_qpos",
            "rollout_qvel",
            "qacc_warmstart",
            "qfrc_actuator",
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float64:
                raise ValueError(f"{name} must remain float64, got {value.dtype}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
        if int(self.metadata.get("compact_schema_version", -1)) != COMPACT_SCHEMA_VERSION:
            raise ValueError("unsupported compact schema version")
        if int(self.metadata.get("source_schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("compact source must be schema-v3")
        if int(self.metadata.get("nq", -1)) != self.nq:
            raise ValueError("metadata nq differs from array shape")
        if int(self.metadata.get("nv", -1)) != self.nv:
            raise ValueError("metadata nv differs from array shape")
        for key in REQUIRED_METADATA_KEYS:
            if key not in self.metadata:
                raise ValueError(f"compact metadata is missing {key}")
        prosthesis = [int(value) for value in self.metadata["prosthesis_dof_indices"]]
        if len(prosthesis) != 4 or len(set(prosthesis)) != 4:
            raise ValueError("expected four unique prosthesis DOFs")

    @classmethod
    def from_full(cls, source: TorqueReplayDataset) -> "CompactTorqueReplayDataset":
        warmstart = np.zeros((source.n_steps + 1, source.nv), dtype=np.float64)
        warmstart[1:] = source.rollout_qacc[:, -1, :]
        result = cls(
            motion_path=source.motion_path,
            dt_control=source.dt_control,
            dt_physics=source.dt_physics,
            rollout_qpos=np.asarray(source.rollout_qpos, dtype=np.float64),
            rollout_qvel=np.asarray(source.rollout_qvel, dtype=np.float64),
            qacc_warmstart=warmstart,
            qfrc_actuator=np.asarray(source.qfrc_actuator, dtype=np.float64),
            metadata=_compact_metadata(source),
        )
        result.validate()
        return result

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                motion_path=np.asarray(self.motion_path),
                dt_control=np.asarray(self.dt_control, dtype=np.float64),
                dt_physics=np.asarray(self.dt_physics, dtype=np.float64),
                rollout_qpos=self.rollout_qpos,
                rollout_qvel=self.rollout_qvel,
                qacc_warmstart=self.qacc_warmstart,
                qfrc_actuator=self.qfrc_actuator,
                metadata=np.asarray(
                    json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)
                ),
            )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CompactTorqueReplayDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                motion_path=str(data["motion_path"].item()),
                dt_control=float(data["dt_control"].item()),
                dt_physics=float(data["dt_physics"].item()),
                rollout_qpos=np.asarray(data["rollout_qpos"], dtype=np.float64),
                rollout_qvel=np.asarray(data["rollout_qvel"], dtype=np.float64),
                qacc_warmstart=np.asarray(data["qacc_warmstart"], dtype=np.float64),
                qfrc_actuator=np.asarray(data["qfrc_actuator"], dtype=np.float64),
                metadata=json.loads(str(data["metadata"].item())),
            )
        result.validate()
        return result
