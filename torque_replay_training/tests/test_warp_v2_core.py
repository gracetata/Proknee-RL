from __future__ import annotations

import torch

from torque_replay_training.warp_v2.config import (
    ReplayAnnealingConfig,
    RewardConfig,
    load_training_config,
)
from torque_replay_training.warp_v2.curriculum import (
    mix_replay_action,
    replay_beta,
)
from torque_replay_training.warp_v2.observations import (
    ACTOR_OBSERVATION_SIZE,
    ObservationHistory,
    actor_frame,
)
from torque_replay_training.warp_v2.ppo import (
    AsymmetricActorCritic,
    tanh_log_prob,
)
from torque_replay_training.warp_v2.rewards import RewardInputs, compute_reward


def test_formal_config_is_4096_worlds() -> None:
    config = load_training_config(
        "torque_replay_training/configs/flat_walk_warp_v2.yaml"
    )
    assert config.environment.num_envs == 4096
    assert (
        config.environment.num_envs * config.ppo.horizon_length
        == 131_072
    )


def test_replay_beta_has_exact_zero_after_annealing() -> None:
    config = ReplayAnnealingConfig(hold_steps=10, end_steps=30)
    assert replay_beta(0, config) == 1.0
    assert replay_beta(10, config) == 1.0
    assert replay_beta(20, config) == 0.5
    assert replay_beta(30, config) == 0.0
    assert replay_beta(1_000, config) == 0.0


def test_replay_action_only_affects_environment_while_beta_nonzero() -> None:
    policy = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
    replay = torch.tensor([[26.0, 56.0, -8.0, 1.5]])
    limits = torch.tensor([260.0, 280.0, 80.0, 15.0])
    mixed, torque, _saturation = mix_replay_action(
        policy,
        replay,
        limits,
        1.0,
    )
    torch.testing.assert_close(mixed, policy + replay / limits)
    independent, independent_torque, _ = mix_replay_action(
        policy,
        replay,
        limits,
        0.0,
    )
    torch.testing.assert_close(independent, policy)
    torch.testing.assert_close(independent_torque, policy * limits)
    assert not torch.equal(torque, independent_torque)


def test_actor_observation_is_96_and_has_no_replay_torque_input() -> None:
    qpos = torch.zeros(2, 89)
    qpos[:, 3] = 1.0
    qvel = torch.zeros(2, 88)
    frame = actor_frame(
        qpos,
        qvel,
        torch.tensor([80, 83, 84, 85]),
        torch.tensor([79, 82, 83, 84]),
        torch.zeros(2, 3),
        torch.zeros(2),
        torch.zeros(2, 4),
    )
    history = ObservationHistory(2, device=torch.device("cpu"))
    history.reset(torch.ones(2, dtype=torch.bool), frame)
    assert history.observation().shape == (2, ACTOR_OBSERVATION_SIZE)


def test_reward_has_explicit_fall_penalty_and_no_replay_torque_argument() -> None:
    zeros4 = torch.zeros(2, 4)
    inputs = RewardInputs(
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
        root_height=torch.ones(2),
        target_root_height=torch.ones(2),
        root_velocity=torch.zeros(2, 3),
        command_velocity=torch.zeros(2, 3),
        yaw_rate=torch.zeros(2),
        command_yaw_rate=torch.zeros(2),
        foot_contact=torch.ones(2, 2, dtype=torch.bool),
        foot_horizontal_speed=torch.zeros(2, 2),
        prosthesis_torque=zeros4,
        prosthesis_acceleration=zeros4,
        policy_action=zeros4,
        previous_policy_action=zeros4,
        joint_limit_violation=zeros4,
        fell=torch.tensor([False, True]),
    )
    reward, terms = compute_reward(inputs, RewardConfig())
    assert "penalty_fall" in terms
    torch.testing.assert_close(reward[0] - reward[1], torch.tensor(201.0))


def test_tanh_log_probability_is_finite_near_action_bounds() -> None:
    raw = torch.tensor([[10.0, -10.0, 0.0, 1.0]])
    mean = torch.zeros_like(raw)
    log_std = torch.full_like(raw, -3.5)
    assert torch.isfinite(tanh_log_prob(raw, mean, log_std)).all()


def test_asymmetric_policy_starts_at_zero_mean() -> None:
    model = AsymmetricActorCritic(
        96,
        277,
        4,
        (32, 16),
        (64, 16),
        -3.5,
    )
    action = model.act_inference(torch.randn(3, 96))
    assert torch.equal(action, torch.zeros(3, 4))
