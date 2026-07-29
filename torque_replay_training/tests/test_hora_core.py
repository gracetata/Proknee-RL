from __future__ import annotations

import torch

from torque_replay_training.hora.experience import ExperienceBuffer
from torque_replay_training.hora.models import ActorCritic
from torque_replay_training.hora.ppo import policy_kl


def test_actor_mean_starts_at_exact_zero() -> None:
    model = ActorCritic(39, 4, (64, 64), initial_log_std=-3.0)
    observation = torch.randn(8, 39)
    assert torch.equal(model.act_inference(observation), torch.zeros(8, 4))


def test_policy_kl_is_zero_for_identical_distributions() -> None:
    mean = torch.randn(16, 4)
    sigma = torch.full((16, 4), 0.05)
    assert policy_kl(mean, sigma, mean, sigma).item() == 0.0


def test_experience_buffer_gae_is_finite() -> None:
    buffer = ExperienceBuffer(2, 8, 3, 1, torch.device("cpu"))
    for step in range(8):
        buffer.update("rewards", step, torch.ones(2, 1))
        buffer.update("values", step, torch.zeros(2, 1))
    buffer.compute_returns(torch.zeros(2, 1), gamma=0.99, gae_lambda=0.95)
    data = buffer.training_data()
    assert data["returns"].shape == (16, 1)
    assert torch.isfinite(data["returns"]).all()
    assert torch.isfinite(data["advantages"]).all()
