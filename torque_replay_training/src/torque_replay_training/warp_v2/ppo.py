"""HORA-derived asymmetric PPO with bounded tanh-Gaussian actions."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from ..hora.models import MLP, RunningMeanStd
from ..hora.ppo import AdaptiveScheduler, policy_kl
from .config import PPOConfig

LOG_PROB_EPSILON = 1e-6


def normalize_snapshot(
    normalizer: RunningMeanStd,
    value: torch.Tensor,
) -> torch.Tensor:
    """Normalize against immutable copies of running statistics."""

    mean = normalizer.running_mean.float().detach().clone()
    variance = normalizer.running_var.float().detach().clone()
    return (
        (value - mean) / torch.sqrt(variance + normalizer.epsilon)
    ).clamp(-5.0, 5.0)


def tanh_log_prob(
    raw_action: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """Log probability of ``tanh(raw_action)`` with Jacobian correction."""

    sigma = log_std.exp()
    distribution = torch.distributions.Normal(mean, sigma)
    bounded = torch.tanh(raw_action)
    correction = torch.log(1.0 - bounded.square() + LOG_PROB_EPSILON)
    return (distribution.log_prob(raw_action) - correction).sum(dim=-1)


class AsymmetricActorCritic(nn.Module):
    def __init__(
        self,
        actor_observation_size: int,
        critic_observation_size: int,
        action_size: int,
        actor_units: Sequence[int],
        critic_units: Sequence[int],
        initial_log_std: float,
    ) -> None:
        super().__init__()
        self.actor_mlp = MLP(actor_units, actor_observation_size)
        self.critic_mlp = MLP(critic_units, critic_observation_size)
        self.mu = nn.Linear(int(actor_units[-1]), action_size)
        self.value = nn.Linear(int(critic_units[-1]), 1)
        self.log_std = nn.Parameter(
            torch.full((action_size,), float(initial_log_std))
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        # Zero mean makes the initial policy explore around replay torque while
        # beta > 0. This is initialization, not supervised behavior cloning.
        nn.init.zeros_(self.mu.weight)

    def distribution_parameters(
        self,
        actor_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mu(self.actor_mlp(actor_observation))
        return mean, mean * 0.0 + self.log_std

    def value_function(self, critic_observation: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic_mlp(critic_observation))

    @torch.no_grad()
    def act(
        self,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mean, log_std = self.distribution_parameters(actor_observation)
        raw_action = mean + log_std.exp() * torch.randn_like(mean)
        return {
            "actions": torch.tanh(raw_action),
            "raw_actions": raw_action,
            "neglogpacs": -tanh_log_prob(raw_action, mean, log_std),
            "values": self.value_function(critic_observation),
            "mus": mean,
            "sigmas": log_std.exp(),
        }

    @torch.no_grad()
    def act_inference(self, actor_observation: torch.Tensor) -> torch.Tensor:
        mean, _log_std = self.distribution_parameters(actor_observation)
        return torch.tanh(mean)

    def evaluate(
        self,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mean, log_std = self.distribution_parameters(actor_observation)
        normal = torch.distributions.Normal(mean, log_std.exp())
        return {
            "neglogpacs": -tanh_log_prob(raw_action, mean, log_std),
            "values": self.value_function(critic_observation),
            # Normal entropy is a stable exploration diagnostic. PPO ratios use
            # the exact transformed log probability above.
            "entropy": normal.entropy().sum(dim=-1),
            "mus": mean,
            "sigmas": log_std.exp(),
        }


def _flatten(value: torch.Tensor) -> torch.Tensor:
    shape = value.shape
    return value.transpose(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class AsymmetricRollout:
    def __init__(
        self,
        num_envs: int,
        horizon: int,
        actor_observation_size: int,
        critic_observation_size: int,
        action_size: int,
        device: torch.device,
    ) -> None:
        prefix = (horizon, num_envs)
        self.horizon = horizon
        self.data = {
            "actor_observations": torch.zeros(
                (*prefix, actor_observation_size), device=device
            ),
            "critic_observations": torch.zeros(
                (*prefix, critic_observation_size), device=device
            ),
            "actions": torch.zeros((*prefix, action_size), device=device),
            "raw_actions": torch.zeros((*prefix, action_size), device=device),
            "neglogpacs": torch.zeros(prefix, device=device),
            "values": torch.zeros((*prefix, 1), device=device),
            "rewards": torch.zeros((*prefix, 1), device=device),
            "dones": torch.zeros(prefix, dtype=torch.bool, device=device),
            "mus": torch.zeros((*prefix, action_size), device=device),
            "sigmas": torch.zeros((*prefix, action_size), device=device),
            "returns": torch.zeros((*prefix, 1), device=device),
        }

    def set(self, name: str, step: int, value: torch.Tensor) -> None:
        self.data[name][step].copy_(value.detach())

    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = torch.zeros_like(last_value)
        for step in reversed(range(self.horizon)):
            next_value = (
                last_value if step == self.horizon - 1 else self.data["values"][step + 1]
            )
            nonterminal = (~self.data["dones"][step]).float().unsqueeze(1)
            delta = (
                self.data["rewards"][step]
                + gamma * next_value * nonterminal
                - self.data["values"][step]
            )
            gae = delta + gamma * gae_lambda * nonterminal * gae
            self.data["returns"][step] = gae + self.data["values"][step]

    def flattened(self) -> dict[str, torch.Tensor]:
        result = {name: _flatten(value) for name, value in self.data.items()}
        advantages = result["returns"] - result["values"]
        result["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ).squeeze(1)
        return result


class WarpHoraPPO:
    """PPO runner whose environment, rollout and optimizer stay on one GPU."""

    def __init__(
        self,
        env: Any,
        output_dir: str | Path,
        config: PPOConfig,
        *,
        run_metadata: dict[str, Any],
    ) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir = self.output_dir / "stage1_nn"
        self.tensorboard_dir = self.output_dir / "stage1_tb"
        self.checkpoint_dir.mkdir()
        self.tensorboard_dir.mkdir()
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.writer = SummaryWriter(str(self.tensorboard_dir))
        self.run_metadata = dict(run_metadata)
        self.model = AsymmetricActorCritic(
            env.actor_observation_size,
            env.critic_observation_size,
            env.action_size,
            config.actor_units,
            config.critic_units,
            config.initial_log_std,
        ).to(self.device)
        self.actor_normalizer = RunningMeanStd(
            (env.actor_observation_size,)
        ).to(self.device)
        self.critic_normalizer = RunningMeanStd(
            (env.critic_observation_size,)
        ).to(self.device)
        self.value_normalizer = RunningMeanStd((1,)).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.scheduler = AdaptiveScheduler(config.target_kl)
        self.learning_rate = config.learning_rate
        self.batch_size = env.num_envs * config.horizon_length
        self.rollout = AsymmetricRollout(
            env.num_envs,
            config.horizon_length,
            env.actor_observation_size,
            env.critic_observation_size,
            env.action_size,
            self.device,
        )
        self.actor_observation: torch.Tensor | None = None
        self.critic_observation: torch.Tensor | None = None
        self.agent_steps = 0
        self.epoch = 0
        self.started = time.time()

    def _normalized(
        self,
        actor_observation: torch.Tensor,
        critic_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.actor_normalizer(actor_observation),
            self.critic_normalizer(critic_observation),
        )

    def _collect(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        if self.actor_observation is None or self.critic_observation is None:
            self.actor_observation, self.critic_observation = self.env.reset()
        self.actor_normalizer.train()
        self.critic_normalizer.train()
        self.value_normalizer.eval()
        info_values: dict[str, list[float]] = {}
        completed_returns: list[torch.Tensor] = []
        completed_falls: list[torch.Tensor] = []
        for step in range(self.config.horizon_length):
            actor_norm, critic_norm = self._normalized(
                self.actor_observation,
                self.critic_observation,
            )
            result = self.model.act(actor_norm, critic_norm)
            result["values"] = self.value_normalizer(
                result["values"],
                unnormalize=True,
            )
            self.rollout.set("actor_observations", step, self.actor_observation)
            self.rollout.set("critic_observations", step, self.critic_observation)
            for name in (
                "actions",
                "raw_actions",
                "neglogpacs",
                "values",
                "mus",
                "sigmas",
            ):
                self.rollout.set(name, step, result[name])
            actor_next, critic_next, reward, done, info = self.env.step(
                result["actions"]
            )
            self.rollout.set("dones", step, done)
            shaped = reward.unsqueeze(1) * self.config.reward_scale
            self.critic_normalizer.eval()
            with torch.no_grad():
                terminal_critic = self.critic_normalizer(
                    info["terminal_critic_observation"]
                )
                terminal_value = self.model.value_function(terminal_critic)
                terminal_value = self.value_normalizer(
                    terminal_value,
                    unnormalize=True,
                )
            self.critic_normalizer.train()
            shaped += (
                self.config.gamma
                * terminal_value
                * info["time_outs"].float().unsqueeze(1)
            )
            self.rollout.set("rewards", step, shaped)
            if info["completed_return"].numel():
                completed_returns.append(info["completed_return"])
                completed_falls.append(info["completed_fell"].float())
            for name, value in info.items():
                if name in {
                    "time_outs",
                    "terminal_critic_observation",
                    "completed_return",
                    "completed_fell",
                }:
                    continue
                if isinstance(value, torch.Tensor) and value.numel():
                    info_values.setdefault(name, []).append(float(value.float().mean()))
            self.actor_observation = actor_next
            self.critic_observation = critic_next
        actor_norm, critic_norm = self._normalized(
            self.actor_observation,
            self.critic_observation,
        )
        with torch.no_grad():
            last_value = self.model.value_function(critic_norm)
            last_value = self.value_normalizer(last_value, unnormalize=True)
        self.rollout.compute_returns(
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
        )
        returns = torch.cat(completed_returns) if completed_returns else torch.empty(0)
        falls = torch.cat(completed_falls) if completed_falls else torch.empty(0)
        metrics = {
            "episodes": float(returns.numel()),
            "mean_episode_return": (
                float(returns.mean()) if returns.numel() else float("nan")
            ),
            "fall_rate": float(falls.mean()) if falls.numel() else float("nan"),
            **{
                f"mean_{name}": float(np.mean(values))
                for name, values in info_values.items()
                if values
            },
        }
        return self.rollout.flattened(), metrics

    def _optimize(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        self.value_normalizer.train()
        # Update value statistics once, then use one immutable snapshot for
        # old values, returns, and every minibatch in this PPO epoch.
        self.value_normalizer(data["returns"])
        self.value_normalizer.eval()
        data["values"] = normalize_snapshot(
            self.value_normalizer,
            data["values"],
        )
        data["returns"] = normalize_snapshot(
            self.value_normalizer,
            data["returns"],
        )
        losses = {
            name: []
            for name in ("actor_loss", "critic_loss", "entropy", "kl")
        }
        count = data["actor_observations"].shape[0]
        self.actor_normalizer.eval()
        self.critic_normalizer.eval()
        for _ in range(self.config.mini_epochs):
            permutation = torch.randperm(count, device=self.device)
            epoch_kl: list[torch.Tensor] = []
            for start in range(0, count, self.config.minibatch_size):
                selected = permutation[start : start + self.config.minibatch_size]
                actor_obs = normalize_snapshot(
                    self.actor_normalizer,
                    data["actor_observations"][selected]
                )
                critic_obs = normalize_snapshot(
                    self.critic_normalizer,
                    data["critic_observations"][selected]
                )
                result = self.model.evaluate(
                    actor_obs,
                    critic_obs,
                    data["raw_actions"][selected],
                )
                ratio = torch.exp(
                    data["neglogpacs"][selected] - result["neglogpacs"]
                )
                advantage = data["advantages"][selected]
                unclipped = -advantage * ratio
                clipped = -advantage * ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                actor_loss = torch.maximum(unclipped, clipped).mean()
                old_value = data["values"][selected]
                returns = data["returns"][selected]
                clipped_value = old_value + (
                    result["values"] - old_value
                ).clamp(
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon,
                )
                critic_loss = torch.maximum(
                    (result["values"] - returns).square(),
                    (clipped_value - returns).square(),
                ).mean()
                entropy = result["entropy"].mean()
                total = (
                    actor_loss
                    + 0.5 * self.config.critic_coefficient * critic_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                with torch.no_grad():
                    kl = policy_kl(
                        result["mus"],
                        result["sigmas"],
                        data["mus"][selected],
                        data["sigmas"][selected],
                    )
                epoch_kl.append(kl)
                losses["actor_loss"].append(float(actor_loss))
                losses["critic_loss"].append(float(critic_loss))
                losses["entropy"].append(float(entropy))
                losses["kl"].append(float(kl))
            mean_kl = float(torch.stack(epoch_kl).mean())
            self.learning_rate = self.scheduler.update(
                self.learning_rate,
                mean_kl,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        return {name: float(np.mean(values)) for name, values in losses.items()}

    def save(self, name: str, metrics: dict[str, Any]) -> Path:
        target = self.checkpoint_dir / f"{name}.pth"
        torch.save(
            {
                "model": self.model.state_dict(),
                "actor_mean_std": self.actor_normalizer.state_dict(),
                "critic_mean_std": self.critic_normalizer.state_dict(),
                "value_mean_std": self.value_normalizer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "agent_steps": self.agent_steps,
                "epoch": self.epoch,
                "ppo_config": asdict(self.config),
                "run_metadata": self.run_metadata,
                "metrics": metrics,
                "replay_beta": float(self.env.beta),
            },
            target,
        )
        return target

    def train(self) -> dict[str, Any]:
        last: dict[str, Any] = {}
        try:
            while self.agent_steps < self.config.max_agent_steps:
                collect_started = time.time()
                data, environment = self._collect()
                collect_seconds = time.time() - collect_started
                optimize_started = time.time()
                optimization = self._optimize(data)
                optimize_seconds = time.time() - optimize_started
                self.agent_steps += self.batch_size
                self.epoch += 1
                self.env.set_agent_steps(self.agent_steps)
                last = {
                    "epoch": self.epoch,
                    "agent_steps": self.agent_steps,
                    "replay_beta": float(self.env.beta),
                    "elapsed_seconds": time.time() - self.started,
                    "collect_seconds": collect_seconds,
                    "optimize_seconds": optimize_seconds,
                    "env_steps_per_second": self.batch_size / collect_seconds,
                    "learning_rate": self.learning_rate,
                    **environment,
                    **optimization,
                }
                if (
                    self.epoch % max(1, self.config.save_frequency) == 0
                    or self.agent_steps >= self.config.max_agent_steps
                ):
                    checkpoint = self.save(
                        f"ep_{self.epoch:06d}_step_{self.agent_steps:012d}",
                        last,
                    )
                    self.save("last", last)
                    last["checkpoint"] = str(checkpoint)
                with self.metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(last, sort_keys=True) + "\n")
                for name, value in last.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        self.writer.add_scalar(name, value, self.agent_steps)
                self.writer.flush()
                print(json.dumps(last, sort_keys=True), flush=True)
            return last
        finally:
            self.writer.close()
