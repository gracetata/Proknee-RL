from __future__ import annotations

import json

import numpy as np

from torque_replay_training.compact_schema import CompactTorqueReplayDataset


def _dataset() -> CompactTorqueReplayDataset:
    metadata = {
        "compact_schema_version": 1,
        "source_schema_version": 3,
        "nq": 3,
        "nv": 2,
        "actual_steps": 4,
        "joint_metadata": [
            {
                "dofadr": 0,
                "qposadr": 0,
                "nq": 1,
                "nv": 1,
                "name": "joint_a",
            },
            {
                "dofadr": 1,
                "qposadr": 1,
                "nq": 1,
                "nv": 1,
                "name": "joint_b",
            },
        ],
        "joint_names": ["joint_a", "joint_b"],
        "prosthesis_dof_indices": [0, 1, 0, 1],
        "prosthesis_joint_names": ["a", "b", "c", "d"],
        "prosthesis_qpos_indices": [0, 1, 0, 1],
        "root_dof_indices": [],
    }
    # Validation requires unique prosthesis indices; use an nv=4 fixture.
    metadata["nv"] = 4
    metadata["prosthesis_dof_indices"] = [0, 1, 2, 3]
    return CompactTorqueReplayDataset(
        motion_path="test/motion",
        dt_control=0.01,
        dt_physics=0.002,
        rollout_qpos=np.zeros((5, 3), dtype=np.float64),
        rollout_qvel=np.zeros((5, 4), dtype=np.float64),
        qacc_warmstart=np.zeros((5, 4), dtype=np.float64),
        qfrc_actuator=np.ones((4, 5, 4), dtype=np.float64),
        metadata=metadata,
    )


def test_compact_schema_roundtrip_excludes_unneeded_arrays(tmp_path) -> None:
    dataset = _dataset()
    path = dataset.save(tmp_path / "compact.npz")
    loaded = CompactTorqueReplayDataset.load(path)
    assert np.array_equal(loaded.qfrc_actuator, dataset.qfrc_actuator)
    with np.load(path, allow_pickle=False) as raw:
        assert set(raw.files) == {
            "motion_path",
            "dt_control",
            "dt_physics",
            "rollout_qpos",
            "rollout_qvel",
            "qacc_warmstart",
            "qfrc_actuator",
            "metadata",
        }
        metadata = json.loads(str(raw["metadata"].item()))
        assert "actuator_names" not in metadata
