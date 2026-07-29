"""HORA-derived GPU experience buffer and GAE.

Adapted from HaozhiQi/hora at commit 410d958 under the MIT license.
The original HORA implementation is based on rl_games.
"""

from __future__ import annotations

import torch


def _flatten_time_env(value: torch.Tensor) -> torch.Tensor:
    shape = value.size()
    return value.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class ExperienceBuffer:
    def __init__(
        self,
        num_envs: int,
        horizon_length: int,
        observation_size: int,
        action_size: int,
        device: torch.device,
    ) -> None:
        prefix = (horizon_length, num_envs)
        self.horizon_length = horizon_length
        self.storage = {
            "observations": torch.zeros(
                (*prefix, observation_size), dtype=torch.float32, device=device
            ),
            "rewards": torch.zeros((*prefix, 1), dtype=torch.float32, device=device),
            "values": torch.zeros((*prefix, 1), dtype=torch.float32, device=device),
            "neglogpacs": torch.zeros(prefix, dtype=torch.float32, device=device),
            "dones": torch.zeros(prefix, dtype=torch.uint8, device=device),
            "actions": torch.zeros(
                (*prefix, action_size), dtype=torch.float32, device=device
            ),
            "mus": torch.zeros((*prefix, action_size), dtype=torch.float32, device=device),
            "sigmas": torch.zeros(
                (*prefix, action_size), dtype=torch.float32, device=device
            ),
            "returns": torch.zeros((*prefix, 1), dtype=torch.float32, device=device),
        }

    def update(self, name: str, index: int, value: torch.Tensor) -> None:
        self.storage[name][index].copy_(value)

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_gae = torch.zeros_like(last_values)
        advantages = torch.zeros_like(self.storage["rewards"])
        for step in reversed(range(self.horizon_length)):
            next_values = (
                last_values
                if step == self.horizon_length - 1
                else self.storage["values"][step + 1]
            )
            nonterminal = (1.0 - self.storage["dones"][step].float()).unsqueeze(1)
            delta = (
                self.storage["rewards"][step]
                + gamma * next_values * nonterminal
                - self.storage["values"][step]
            )
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[step] = last_gae
        self.storage["returns"].copy_(advantages + self.storage["values"])

    def training_data(self) -> dict[str, torch.Tensor]:
        data = {name: _flatten_time_env(value) for name, value in self.storage.items()}
        advantages = data["returns"] - data["values"]
        data["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ).squeeze(1)
        return data
