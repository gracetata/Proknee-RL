"""HORA-derived Actor-Critic and running-normalization modules.

Adapted from HaozhiQi/hora at commit 410d958 under the MIT license.
The original HORA model is based on rl_games.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, units: Sequence[int], input_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for output_size in units:
            layers.extend((nn.Linear(input_size, int(output_size)), nn.ELU()))
            input_size = int(output_size)
        self.mlp = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp(value)


class ActorCritic(nn.Module):
    """HORA-style shared-latent Gaussian Actor-Critic."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        units: Sequence[int],
        initial_log_std: float,
    ) -> None:
        super().__init__()
        if not units:
            raise ValueError("at least one hidden layer is required")
        self.actor_mlp = MLP(units, observation_size)
        self.value = nn.Linear(int(units[-1]), 1)
        self.mu = nn.Linear(int(units[-1]), action_size)
        self.log_std = nn.Parameter(
            torch.full((action_size,), float(initial_log_std), dtype=torch.float32)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        # Project adaptation: action zero must exactly preserve the healthy
        # prosthesis baseline before the first optimizer update.
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

    def _actor_critic(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.actor_mlp(observations)
        mean = self.mu(latent)
        value = self.value(latent)
        log_std = mean * 0.0 + self.log_std
        return mean, log_std, value

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_std, value = self._actor_critic(observations)
        sigma = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, sigma)
        action = distribution.sample()
        return {
            "neglogpacs": -distribution.log_prob(action).sum(dim=1),
            "values": value,
            "actions": action,
            "mus": mean,
            "sigmas": sigma,
        }

    @torch.no_grad()
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _log_std, _value = self._actor_critic(observations)
        return mean

    def forward(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mean, log_std, value = self._actor_critic(observations)
        sigma = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, sigma)
        return {
            "prev_neglogp": -distribution.log_prob(previous_actions).sum(dim=1),
            "values": value,
            "entropy": distribution.entropy().sum(dim=-1),
            "mus": mean,
            "sigmas": sigma,
        }


class RunningMeanStd(nn.Module):
    """Numerically stable HORA-style running standardization."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, value: torch.Tensor, unnormalize: bool = False) -> torch.Tensor:
        if self.training:
            batch_mean = value.mean(dim=0).double()
            batch_var = value.var(dim=0, unbiased=False).double()
            batch_count = torch.as_tensor(value.shape[0], device=value.device, dtype=torch.float64)
            delta = batch_mean - self.running_mean
            total = self.count + batch_count
            new_mean = self.running_mean + delta * batch_count / total
            first = self.running_var * self.count
            second = batch_var * batch_count
            moment = first + second + delta.square() * self.count * batch_count / total
            self.running_mean.copy_(new_mean)
            self.running_var.copy_(moment / total)
            self.count.copy_(total)
        mean = self.running_mean.float()
        variance = self.running_var.float()
        if unnormalize:
            return torch.sqrt(variance + self.epsilon) * value.clamp(-5.0, 5.0) + mean
        return ((value - mean) / torch.sqrt(variance + self.epsilon)).clamp(-5.0, 5.0)
