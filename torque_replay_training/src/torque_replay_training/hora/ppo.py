"""HORA-derived PyTorch PPO runner for vectorized torque replay.

Adapted from HaozhiQi/hora at commit 410d958 under the MIT license.
The original HORA PPO is based on rl_games.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from tensorboardX import SummaryWriter
import yaml

from .experience import ExperienceBuffer
from .models import ActorCritic, RunningMeanStd


@dataclass(frozen=True)
class HoraPPOConfig:
    seed: int = 0
    device: str = "cuda:0"
    max_agent_steps: int = 2_000_000
    num_actors: int = 16
    horizon_length: int = 256
    minibatch_size: int = 1024
    mini_epochs: int = 5
    learning_rate: float = 3e-4
    kl_threshold: float = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    e_clip: float = 0.2
    critic_coefficient: float = 2.0
    entropy_coefficient: float = 5e-4
    bounds_loss_coefficient: float = 1e-4
    grad_norm: float = 1.0
    normalize_input: bool = True
    normalize_value: bool = True
    value_bootstrap: bool = True
    reward_scale: float = 0.01
    initial_log_std: float = -3.0
    save_frequency: int = 10


def load_hora_config(
    path: str | Path,
) -> tuple[HoraPPOConfig, dict[str, Any], tuple[int, ...]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ppo = HoraPPOConfig(**dict(raw.get("ppo", {})))
    environment = dict(raw.get("environment", {}))
    units = tuple(int(value) for value in raw.get("network", {}).get("units", (256, 256)))
    if ppo.num_actors <= 0 or ppo.horizon_length <= 0:
        raise ValueError("num_actors and horizon_length must be positive")
    batch_size = ppo.num_actors * ppo.horizon_length
    if batch_size % ppo.minibatch_size:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by "
            f"minibatch_size={ppo.minibatch_size}"
        )
    return ppo, environment, units


class AdaptiveScheduler:
    """HORA/rl_games KL-based learning-rate scheduler."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def update(self, learning_rate: float, kl: float) -> float:
        if kl > 2.0 * self.threshold:
            return max(learning_rate / 1.5, 1e-6)
        if kl < 0.5 * self.threshold:
            return min(learning_rate * 1.5, 1e-2)
        return learning_rate


def policy_kl(
    new_mean: torch.Tensor,
    new_sigma: torch.Tensor,
    old_mean: torch.Tensor,
    old_sigma: torch.Tensor,
) -> torch.Tensor:
    # Algebraically identical to HORA's diagonal-Gaussian KL, but clamp the
    # strictly-positive standard deviations instead of adding epsilon to the
    # ratio.  The latter can produce a small negative "KL" at low sigma.
    epsilon = torch.finfo(new_sigma.dtype).eps
    new_sigma = new_sigma.clamp_min(epsilon)
    old_sigma = old_sigma.clamp_min(epsilon)
    first = torch.log(old_sigma / new_sigma)
    second = (
        new_sigma.square() + (old_mean - new_mean).square()
    ) / (2.0 * old_sigma.square())
    return (first + second - 0.5).sum(dim=-1).clamp_min(0.0).mean()


class HoraPPO:
    """HORA-style PPO operating on a torch vector-environment adapter."""

    def __init__(
        self,
        env: Any,
        output_dir: str | Path,
        config: HoraPPOConfig,
        network_units: tuple[int, ...],
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir = self.output_dir / "stage1_nn"
        self.tensorboard_dir = self.output_dir / "stage1_tb"
        self.checkpoint_dir.mkdir()
        self.tensorboard_dir.mkdir()
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("HORA PPO requested CUDA but PyTorch CUDA is unavailable")
        if env.num_envs != config.num_actors:
            raise ValueError(
                f"environment actors={env.num_envs} != PPO actors={config.num_actors}"
            )
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.model = ActorCritic(
            env.observation_size,
            env.action_size,
            network_units,
            config.initial_log_std,
        ).to(self.device)
        self.observation_normalizer = RunningMeanStd(
            (env.observation_size,)
        ).to(self.device)
        self.value_normalizer = RunningMeanStd((1,)).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.scheduler = AdaptiveScheduler(config.kl_threshold)
        self.learning_rate = config.learning_rate
        self.batch_size = config.num_actors * config.horizon_length
        self.storage = ExperienceBuffer(
            config.num_actors,
            config.horizon_length,
            env.observation_size,
            env.action_size,
            self.device,
        )
        self.writer = SummaryWriter(str(self.tensorboard_dir))
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.run_metadata = dict(run_metadata or {})
        self.current_observation: torch.Tensor | None = None
        self.current_rewards = torch.zeros(
            (config.num_actors, 1), dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            config.num_actors, dtype=torch.float32, device=self.device
        )
        self.agent_steps = 0
        self.epoch = 0
        self.started = time.time()

    def _set_eval(self) -> None:
        self.model.eval()
        self.observation_normalizer.eval()
        self.value_normalizer.eval()

    def _set_train(self) -> None:
        self.model.train()
        self.observation_normalizer.train()
        self.value_normalizer.train()

    def _model_act(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = (
            self.observation_normalizer(observation)
            if self.config.normalize_input
            else observation
        )
        result = self.model.act(normalized)
        if self.config.normalize_value:
            result["values"] = self.value_normalizer(
                result["values"],
                unnormalize=True,
            )
        return result

    def _collect(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        if self.current_observation is None:
            self.current_observation = self.env.reset()
        completed_rewards: list[torch.Tensor] = []
        completed_lengths: list[torch.Tensor] = []
        completed_falls: list[torch.Tensor] = []
        info_accumulator: dict[str, list[float]] = {}
        self._set_eval()
        for step in range(self.config.horizon_length):
            result = self._model_act(self.current_observation)
            self.storage.update("observations", step, self.current_observation)
            for name in ("actions", "neglogpacs", "values", "mus", "sigmas"):
                self.storage.update(name, step, result[name])
            actions = result["actions"].clamp(-1.0, 1.0)
            observation, rewards, dones, info = self.env.step(actions)
            self.storage.update("dones", step, dones)
            shaped = rewards.unsqueeze(1) * self.config.reward_scale
            if self.config.value_bootstrap:
                shaped = shaped + (
                    self.config.gamma
                    * result["values"]
                    * info["time_outs"].unsqueeze(1).float()
                )
            self.storage.update("rewards", step, shaped)
            self.current_rewards += rewards.unsqueeze(1)
            self.current_lengths += 1
            finished = dones.nonzero(as_tuple=False).reshape(-1)
            if finished.numel():
                completed_rewards.append(self.current_rewards[finished].reshape(-1))
                completed_lengths.append(self.current_lengths[finished].reshape(-1))
                completed_falls.append(info["fell"][finished].float().reshape(-1))
            alive = 1.0 - dones.float()
            self.current_rewards *= alive.unsqueeze(1)
            self.current_lengths *= alive
            for name, value in info.items():
                if name in {"time_outs", "fell"}:
                    continue
                if isinstance(value, torch.Tensor):
                    info_accumulator.setdefault(name, []).append(float(value.float().mean()))
                elif isinstance(value, (int, float)):
                    info_accumulator.setdefault(name, []).append(float(value))
            self.current_observation = observation

        final = self._model_act(self.current_observation)
        self.storage.compute_returns(
            final["values"],
            self.config.gamma,
            self.config.gae_lambda,
        )
        data = self.storage.training_data()
        episode_rewards = (
            torch.cat(completed_rewards) if completed_rewards else torch.empty(0)
        )
        episode_lengths = (
            torch.cat(completed_lengths) if completed_lengths else torch.empty(0)
        )
        episode_falls = (
            torch.cat(completed_falls) if completed_falls else torch.empty(0)
        )
        metrics = {
            "episodes": float(episode_rewards.numel()),
            "mean_episode_return": (
                float(episode_rewards.mean()) if episode_rewards.numel() else float("nan")
            ),
            "mean_episode_length": (
                float(episode_lengths.mean()) if episode_lengths.numel() else float("nan")
            ),
            "fall_rate": (
                float(episode_falls.mean()) if episode_falls.numel() else float("nan")
            ),
            **{
                f"mean_{name}": float(np.mean(values))
                for name, values in info_accumulator.items()
                if values
            },
        }
        return data, metrics

    def _optimize(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        if self.config.normalize_value:
            self.value_normalizer.train()
            data["values"] = self.value_normalizer(data["values"])
            data["returns"] = self.value_normalizer(data["returns"])
            self.value_normalizer.eval()
        count = data["observations"].shape[0]
        losses: dict[str, list[float]] = {
            name: []
            for name in (
                "actor_loss",
                "critic_loss",
                "bounds_loss",
                "entropy",
                "kl",
            )
        }
        self._set_train()
        for _epoch in range(self.config.mini_epochs):
            permutation = torch.randperm(count, device=self.device)
            epoch_kls: list[torch.Tensor] = []
            for start in range(0, count, self.config.minibatch_size):
                selected = permutation[start : start + self.config.minibatch_size]
                observations = data["observations"][selected]
                if self.config.normalize_input:
                    observations = self.observation_normalizer(observations)
                result = self.model(observations, data["actions"][selected])
                ratio = torch.exp(
                    data["neglogpacs"][selected] - result["prev_neglogp"]
                )
                advantage = data["advantages"][selected]
                first = advantage * ratio
                second = advantage * ratio.clamp(
                    1.0 - self.config.e_clip,
                    1.0 + self.config.e_clip,
                )
                actor_loss = torch.maximum(-first, -second).mean()
                old_value = data["values"][selected]
                returns = data["returns"][selected]
                clipped_value = old_value + (result["values"] - old_value).clamp(
                    -self.config.e_clip,
                    self.config.e_clip,
                )
                critic_loss = torch.maximum(
                    (result["values"] - returns).square(),
                    (clipped_value - returns).square(),
                ).mean()
                soft_bound = 1.1
                bounds_loss = (
                    (result["mus"] - soft_bound).clamp_min(0.0).square()
                    + (-result["mus"] - soft_bound).clamp_min(0.0).square()
                ).sum(dim=-1).mean()
                entropy = result["entropy"].mean()
                total = (
                    actor_loss
                    + 0.5 * self.config.critic_coefficient * critic_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.bounds_loss_coefficient * bounds_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_norm,
                )
                self.optimizer.step()
                with torch.no_grad():
                    kl = policy_kl(
                        result["mus"],
                        result["sigmas"],
                        data["mus"][selected],
                        data["sigmas"][selected],
                    )
                epoch_kls.append(kl)
                losses["actor_loss"].append(float(actor_loss.detach()))
                losses["critic_loss"].append(float(critic_loss.detach()))
                losses["bounds_loss"].append(float(bounds_loss.detach()))
                losses["entropy"].append(float(entropy.detach()))
                losses["kl"].append(float(kl.detach()))
            mean_kl = float(torch.stack(epoch_kls).mean())
            self.learning_rate = self.scheduler.update(self.learning_rate, mean_kl)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        return {name: float(np.mean(values)) for name, values in losses.items()}

    def save(self, name: str, metrics: dict[str, Any]) -> Path:
        path = self.checkpoint_dir / f"{name}.pth"
        torch.save(
            {
                "model": self.model.state_dict(),
                "running_mean_std": self.observation_normalizer.state_dict(),
                "value_mean_std": self.value_normalizer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "agent_steps": self.agent_steps,
                "epoch": self.epoch,
                "ppo_config": asdict(self.config),
                "run_metadata": self.run_metadata,
                "metrics": metrics,
            },
            path,
        )
        return path

    def train(self) -> dict[str, Any]:
        last_metrics: dict[str, Any] = {}
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
                last_metrics = {
                    "epoch": self.epoch,
                    "agent_steps": self.agent_steps,
                    "elapsed_seconds": time.time() - self.started,
                    "collect_seconds": collect_seconds,
                    "optimize_seconds": optimize_seconds,
                    "env_steps_per_second": self.batch_size / collect_seconds,
                    "learning_rate": self.learning_rate,
                    **environment,
                    **optimization,
                }
                checkpoint_due = (
                    self.epoch % max(1, self.config.save_frequency) == 0
                    or self.agent_steps >= self.config.max_agent_steps
                )
                if checkpoint_due:
                    checkpoint = self.save(
                        f"ep_{self.epoch:06d}_step_{self.agent_steps:012d}",
                        last_metrics,
                    )
                    self.save("last", last_metrics)
                    last_metrics["checkpoint"] = str(checkpoint)
                with self.metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(last_metrics, sort_keys=True) + "\n")
                for name, value in last_metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        self.writer.add_scalar(name, value, self.agent_steps)
                self.writer.flush()
                print(json.dumps(last_metrics, sort_keys=True), flush=True)
            return last_metrics
        finally:
            self.writer.close()
