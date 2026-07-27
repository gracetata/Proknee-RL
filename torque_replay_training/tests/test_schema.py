from __future__ import annotations

import numpy as np

from torque_replay_training.replay_env import _non_prosthesis_dofs, _warmstart_for_step
from torque_replay_training.schema import SCHEMA_VERSION, TorqueReplayDataset


def _dataset() -> TorqueReplayDataset:
    t, s, nq, nv, nu = 2, 2, 5, 4, 3
    qfrc = np.zeros((t, s, nv), dtype=np.float64)
    qfrc[0, 0, 0] = np.nextafter(1.0, 2.0)
    qacc = np.arange(t * s * nv, dtype=np.float64).reshape(t, s, nv) / 7.0
    return TorqueReplayDataset(
        motion_path="test/motion",
        dt_control=0.01,
        dt_physics=0.005,
        reference_qpos=np.zeros((t, nq)),
        reference_qvel=np.zeros((t, nv)),
        rollout_qpos=np.zeros((t + 1, nq)),
        rollout_qvel=np.zeros((t + 1, nv)),
        rollout_qacc=qacc,
        policy_action=np.zeros((t, 1)),
        actuator_ctrl=np.zeros((t, s, nu)),
        actuator_force=np.zeros((t, s, nu)),
        qfrc_actuator=qfrc,
        qfrc_passive=np.zeros((t, s, nv)),
        qfrc_constraint=np.zeros((t, s, nv)),
        contact_ncon=np.zeros((t, s), dtype=np.int32),
        reward=np.zeros(t),
        absorbing=np.zeros(t, dtype=bool),
        done=np.zeros(t, dtype=bool),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "prosthesis_dof_indices": [0, 1, 2, 3],
        },
    )


def test_physics_transition_arrays_round_trip_as_float64(tmp_path) -> None:
    dataset = _dataset()
    path = tmp_path / "dataset.npz"
    dataset.save(path)
    loaded = TorqueReplayDataset.load(path)

    assert np.array_equal(loaded.qfrc_actuator, dataset.qfrc_actuator)
    assert np.array_equal(loaded.rollout_qacc, dataset.rollout_qacc)
    with np.load(path, allow_pickle=False) as raw:
        assert raw["qfrc_actuator"].dtype == np.float64
        assert raw["rollout_qacc"].dtype == np.float64


def test_random_reset_warmstart_uses_previous_physics_substep() -> None:
    dataset = _dataset()
    assert np.array_equal(_warmstart_for_step(dataset, 0), np.zeros(dataset.nv))
    assert np.array_equal(_warmstart_for_step(dataset, 1), dataset.rollout_qacc[0, -1])


def test_non_prosthesis_replay_includes_root_dofs() -> None:
    replayed = _non_prosthesis_dofs(10, np.asarray([6, 7, 8, 9]))
    assert np.array_equal(replayed, np.arange(6, dtype=np.int32))
