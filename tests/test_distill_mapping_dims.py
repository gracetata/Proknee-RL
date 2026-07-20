"""Dimension checks for prosthesis distillation teacher/student mapping."""

from __future__ import annotations

import numpy as np
import pytest

from musclemimic.distill.config import load_fullbody_config, make_env
from musclemimic.distill.mapping import build_distill_mapping, spot_check_rollout_npz, validate_distill_dimensions
from musclemimic.distill.config import repo_root
from musclemimic.prosthesis.constants import DEFAULT_DISABLED_MUSCLE_NAMES, KNEE15_DISABLED_MUSCLE_NAMES


MOTION = "KIT/3/walk_6m_straight_line04_poses"
CONFIG_NAME = "conf_fullbody_prosthesis_gmr_resnet"
CONFIG_NAME_KNEE15 = "conf_fullbody_prosthesis_gmr_resnet_knee15"
DATASET_DIR = repo_root() / "data/teacher_rollouts/KIT_KINESIS_TRAINING_MOTIONS"


@pytest.fixture(scope="module")
def mapping():
    cfg = load_fullbody_config(CONFIG_NAME)
    teacher_env = make_env(cfg, env_name="MyoFullBody", motion_paths=[MOTION], use_mujoco=True, fixed_start=True)
    student_env = make_env(
        cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[MOTION], use_mujoco=True, fixed_start=True
    )
    return build_distill_mapping(teacher_env, student_env)


@pytest.fixture(scope="module")
def mapping_knee15():
    cfg = load_fullbody_config(CONFIG_NAME_KNEE15)
    teacher_env = make_env(cfg, env_name="MyoFullBody", motion_paths=[MOTION], use_mujoco=True, fixed_start=True)
    student_env = make_env(
        cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[MOTION], use_mujoco=True, fixed_start=True
    )
    return build_distill_mapping(teacher_env, student_env)


def test_disabled_muscle_count(mapping):
    assert len(mapping.disabled_muscle_names) == 19
    assert set(mapping.disabled_muscle_names) == set(DEFAULT_DISABLED_MUSCLE_NAMES)


def test_action_dimensions(mapping):
    assert mapping.teacher_action_dim == 354
    assert mapping.student_action_dim == 339
    assert mapping.n_remaining_muscles == 335
    assert mapping.target_action_dim == 339
    assert len(mapping.teacher_disabled_action_indices) == 19
    assert len(mapping.teacher_remaining_action_indices) == 335
    assert len(mapping.student_prosthesis_action_indices) == 4
    assert mapping.student_prosthesis_action_indices == (335, 336, 337, 338)


def test_obs_dimensions(mapping):
    assert mapping.teacher_obs_dim == 2418
    assert mapping.student_obs_dim == 2327


def test_knee15_action_dimensions(mapping_knee15):
    assert len(mapping_knee15.disabled_muscle_names) == 15
    assert set(mapping_knee15.disabled_muscle_names) == set(KNEE15_DISABLED_MUSCLE_NAMES)
    assert mapping_knee15.teacher_action_dim == 354
    assert mapping_knee15.n_remaining_muscles == 339
    assert mapping_knee15.student_action_dim == 343
    assert mapping_knee15.target_action_dim == 343
    assert mapping_knee15.student_prosthesis_action_indices == (339, 340, 341, 342)


def test_knee15_obs_dimensions(mapping_knee15):
    assert mapping_knee15.teacher_obs_dim == 2418
    assert mapping_knee15.student_obs_dim == 2347


def test_validate_distill_dimensions_passes(mapping):
    validate_distill_dimensions(mapping, obs_dim=mapping.student_obs_dim, n_remaining=mapping.n_remaining_muscles)


def test_spot_check_existing_npz(mapping):
    npz_files = sorted(DATASET_DIR.glob("*.npz"))
    if not npz_files:
        pytest.skip("No teacher rollout npz files on disk")
    spot_check_rollout_npz(npz_files[0], mapping)
    with np.load(npz_files[0], allow_pickle=True) as data:
        assert data["obs_student"].shape[-1] == 2327
        assert data["target_remaining_muscle_action"].shape[-1] == 335
        assert data["target_prosthesis_action"].shape[-1] == 4


def test_mapping_json_roundtrip(mapping, tmp_path):
    path = tmp_path / "mapping.json"
    mapping.save_json(path)
    loaded = type(mapping).load_json(path)
    assert loaded.student_obs_dim == mapping.student_obs_dim
    assert loaded.disabled_muscle_names == mapping.disabled_muscle_names
