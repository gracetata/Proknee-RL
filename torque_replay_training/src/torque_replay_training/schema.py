"""Versioned NPZ schema for full-body active generalized-force rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


@dataclass
class TorqueReplayDataset:
    motion_path: str
    dt_control: float
    dt_physics: float
    reference_qpos: np.ndarray
    reference_qvel: np.ndarray
    rollout_qpos: np.ndarray
    rollout_qvel: np.ndarray
    rollout_qacc: np.ndarray
    policy_action: np.ndarray
    actuator_ctrl: np.ndarray
    actuator_force: np.ndarray
    qfrc_actuator: np.ndarray
    qfrc_passive: np.ndarray
    qfrc_constraint: np.ndarray
    contact_ncon: np.ndarray
    reward: np.ndarray
    absorbing: np.ndarray
    done: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_steps(self) -> int:
        return int(self.policy_action.shape[0])

    @property
    def n_substeps(self) -> int:
        return int(self.qfrc_actuator.shape[1])

    @property
    def nq(self) -> int:
        return int(self.rollout_qpos.shape[-1])

    @property
    def nv(self) -> int:
        return int(self.rollout_qvel.shape[-1])

    @property
    def qfrc_actuator_mean(self) -> np.ndarray:
        return np.asarray(self.qfrc_actuator, dtype=np.float64).mean(axis=1)

    def validate(self) -> None:
        t = self.n_steps
        if t <= 0:
            raise ValueError("dataset has no control steps")
        expected_t = {
            "reference_qpos": self.reference_qpos,
            "reference_qvel": self.reference_qvel,
            "rollout_qacc": self.rollout_qacc,
            "actuator_ctrl": self.actuator_ctrl,
            "actuator_force": self.actuator_force,
            "qfrc_actuator": self.qfrc_actuator,
            "qfrc_passive": self.qfrc_passive,
            "qfrc_constraint": self.qfrc_constraint,
            "contact_ncon": self.contact_ncon,
            "reward": self.reward,
            "absorbing": self.absorbing,
            "done": self.done,
        }
        for name, value in expected_t.items():
            if int(value.shape[0]) != t:
                raise ValueError(f"{name}.shape[0]={value.shape[0]} != policy steps {t}")
        if self.rollout_qpos.shape != (t + 1, self.nq):
            raise ValueError(f"rollout_qpos must be [T+1,nq], got {self.rollout_qpos.shape}")
        if self.rollout_qvel.shape != (t + 1, self.nv):
            raise ValueError(f"rollout_qvel must be [T+1,nv], got {self.rollout_qvel.shape}")
        if self.reference_qpos.shape[1] != self.nq or self.reference_qvel.shape[1] != self.nv:
            raise ValueError("reference and rollout state dimensions differ")
        if self.qfrc_actuator.ndim != 3 or self.qfrc_actuator.shape[2] != self.nv:
            raise ValueError(f"qfrc_actuator must be [T,S,nv], got {self.qfrc_actuator.shape}")
        s = self.n_substeps
        for name in (
            "rollout_qacc",
            "actuator_ctrl",
            "actuator_force",
            "qfrc_passive",
            "qfrc_constraint",
            "contact_ncon",
        ):
            value = getattr(self, name)
            if value.shape[1] != s:
                raise ValueError(f"{name} substeps={value.shape[1]} != {s}")
        for name, value in self.numeric_arrays().items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
        if int(self.metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.metadata.get('schema_version')}")
        p = self.metadata.get("prosthesis_dof_indices", [])
        if len(p) != 4 or len(set(int(x) for x in p)) != 4:
            raise ValueError(f"expected four unique prosthesis DOF indices, got {p}")

    def numeric_arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name))
            for name in (
                "reference_qpos",
                "reference_qvel",
                "rollout_qpos",
                "rollout_qvel",
                "rollout_qacc",
                "policy_action",
                "actuator_ctrl",
                "actuator_force",
                "qfrc_actuator",
                "qfrc_passive",
                "qfrc_constraint",
                "contact_ncon",
                "reward",
                "absorbing",
                "done",
            )
        }

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            motion_path=np.asarray(self.motion_path),
            dt_control=np.asarray(self.dt_control, dtype=np.float64),
            dt_physics=np.asarray(self.dt_physics, dtype=np.float64),
            reference_qpos=self.reference_qpos.astype(np.float64),
            reference_qvel=self.reference_qvel.astype(np.float64),
            rollout_qpos=self.rollout_qpos.astype(np.float64),
            rollout_qvel=self.rollout_qvel.astype(np.float64),
            rollout_qacc=self.rollout_qacc.astype(np.float32),
            policy_action=self.policy_action.astype(np.float32),
            actuator_ctrl=self.actuator_ctrl.astype(np.float32),
            actuator_force=self.actuator_force.astype(np.float32),
            qfrc_actuator=self.qfrc_actuator.astype(np.float32),
            qfrc_actuator_mean=self.qfrc_actuator_mean.astype(np.float32),
            qfrc_passive=self.qfrc_passive.astype(np.float32),
            qfrc_constraint=self.qfrc_constraint.astype(np.float32),
            contact_ncon=self.contact_ncon.astype(np.int32),
            reward=self.reward.astype(np.float32),
            absorbing=self.absorbing.astype(np.bool_),
            done=self.done.astype(np.bool_),
            metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TorqueReplayDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            result = cls(
                motion_path=str(data["motion_path"].item()),
                dt_control=float(data["dt_control"].item()),
                dt_physics=float(data["dt_physics"].item()),
                reference_qpos=np.asarray(data["reference_qpos"], dtype=np.float64),
                reference_qvel=np.asarray(data["reference_qvel"], dtype=np.float64),
                rollout_qpos=np.asarray(data["rollout_qpos"], dtype=np.float64),
                rollout_qvel=np.asarray(data["rollout_qvel"], dtype=np.float64),
                rollout_qacc=np.asarray(data["rollout_qacc"], dtype=np.float64),
                policy_action=np.asarray(data["policy_action"], dtype=np.float32),
                actuator_ctrl=np.asarray(data["actuator_ctrl"], dtype=np.float32),
                actuator_force=np.asarray(data["actuator_force"], dtype=np.float32),
                qfrc_actuator=np.asarray(data["qfrc_actuator"], dtype=np.float64),
                qfrc_passive=np.asarray(data["qfrc_passive"], dtype=np.float64),
                qfrc_constraint=np.asarray(data["qfrc_constraint"], dtype=np.float64),
                contact_ncon=np.asarray(data["contact_ncon"], dtype=np.int32),
                reward=np.asarray(data["reward"], dtype=np.float32),
                absorbing=np.asarray(data["absorbing"], dtype=np.bool_),
                done=np.asarray(data["done"], dtype=np.bool_),
                metadata=metadata,
            )
        result.validate()
        return result

