"""Small self-contained PPO trainer for the four-DOF residual policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

from flax import linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from .replay_env import ReplayConfig, TorqueReplayEnv


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    total_steps: int = 200_000
    rollout_steps: int = 1024
    update_epochs: int = 10
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (128, 128)
    initial_log_std: float = -2.0
    checkpoint_every_updates: int = 10


class ActorCritic(nn.Module):
    action_size: int
    hidden_sizes: Sequence[int]
    initial_log_std: float

    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        actor = observation
        critic = observation
        for width in self.hidden_sizes:
            actor = nn.tanh(nn.Dense(int(width), kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(actor))
            critic = nn.tanh(nn.Dense(int(width), kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(critic))
        # Zero initialization makes the deterministic initial residual exactly zero.
        mean = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="policy_mean",
        )(actor)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="value")(critic)[..., 0]
        log_std = self.param(
            "policy_log_std",
            lambda _key, shape: jnp.full(shape, self.initial_log_std),
            (self.action_size,),
        )
        return mean, log_std, value


def _normal_log_prob(raw_action: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    variance_term = jnp.square((raw_action - mean) / jnp.exp(log_std))
    gaussian = -0.5 * (variance_term + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    squashed = jnp.tanh(raw_action)
    correction = jnp.log(1.0 - jnp.square(squashed) + 1e-6)
    return jnp.sum(gaussian - correction, axis=-1)


def _gaussian_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    accumulator = 0.0
    for index in reversed(range(rewards.size)):
        next_value = last_value if index == rewards.size - 1 else values[index + 1]
        nonterminal = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        accumulator = delta + gamma * gae_lambda * nonterminal * accumulator
        advantages[index] = accumulator
    returns = advantages + values
    return advantages, returns


def load_training_config(path: str | Path) -> tuple[PPOConfig, ReplayConfig]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    ppo_raw = dict(raw.get("ppo", {}))
    replay_raw = dict(raw.get("replay", {}))
    if "hidden_sizes" in ppo_raw:
        ppo_raw["hidden_sizes"] = tuple(int(x) for x in ppo_raw["hidden_sizes"])
    for key in ("residual_limits", "torque_limits"):
        if key in replay_raw:
            replay_raw[key] = tuple(float(x) for x in replay_raw[key])
    return PPOConfig(**ppo_raw), ReplayConfig(**replay_raw)


class ResidualPPOTrainer:
    def __init__(self, env: TorqueReplayEnv, config: PPOConfig, output_dir: str | Path) -> None:
        self.env = env
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = ActorCritic(
            action_size=env.action_size,
            hidden_sizes=config.hidden_sizes,
            initial_log_std=config.initial_log_std,
        )
        self.rng = jax.random.key(config.seed)
        obs, _ = env.reset(seed=config.seed)
        self.current_obs = obs
        self.rng, init_key = jax.random.split(self.rng)
        params = self.model.init(init_key, jnp.asarray(obs)[None, :])["params"]
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
        self._update_jit = jax.jit(self._update_minibatch)

    def _policy(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        self.rng, sample_key = jax.random.split(self.rng)
        mean, log_std, value = self.model.apply({"params": self.state.params}, jnp.asarray(observation)[None, :])
        raw = mean + jnp.exp(log_std) * jax.random.normal(sample_key, mean.shape)
        log_prob = _normal_log_prob(raw, mean, log_std)
        return (
            np.asarray(jnp.tanh(raw)[0], dtype=np.float32),
            np.asarray(raw[0], dtype=np.float32),
            float(np.asarray(log_prob[0])),
            float(np.asarray(value[0])),
        )

    def _value(self, observation: np.ndarray) -> float:
        _mean, _log_std, value = self.model.apply(
            {"params": self.state.params}, jnp.asarray(observation)[None, :]
        )
        return float(np.asarray(value[0]))

    def _collect(self, steps: int) -> dict[str, np.ndarray]:
        rows: dict[str, list[Any]] = {
            name: [] for name in ("obs", "raw_action", "log_prob", "value", "reward", "done")
        }
        episode_returns: list[float] = []
        episode_return = 0.0
        for _ in range(steps):
            action, raw_action, log_prob, value = self._policy(self.current_obs)
            next_obs, reward, terminated, truncated, _info = self.env.step(action)
            done = bool(terminated or truncated)
            rows["obs"].append(self.current_obs.copy())
            rows["raw_action"].append(raw_action)
            rows["log_prob"].append(log_prob)
            rows["value"].append(value)
            rows["reward"].append(reward)
            rows["done"].append(done)
            episode_return += reward
            self.current_obs = next_obs
            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0
                self.current_obs, _ = self.env.reset()
        last_value = self._value(self.current_obs)
        batch = {name: np.asarray(values) for name, values in rows.items()}
        advantages, returns = _gae(
            batch["reward"],
            batch["value"],
            batch["done"],
            last_value,
            self.config.gamma,
            self.config.gae_lambda,
        )
        batch["advantage"] = advantages
        batch["return"] = returns
        batch["episode_returns"] = np.asarray(episode_returns, dtype=np.float32)
        return batch

    def _update_minibatch(self, state, batch):
        config = self.config

        def loss_fn(params):
            mean, log_std, value = self.model.apply({"params": params}, batch["obs"])
            log_prob = _normal_log_prob(batch["raw_action"], mean, log_std)
            ratio = jnp.exp(log_prob - batch["log_prob"])
            unclipped = ratio * batch["advantage"]
            clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * batch[
                "advantage"
            ]
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            value_loss = 0.5 * jnp.mean(jnp.square(value - batch["return"]))
            entropy = jnp.mean(_gaussian_entropy(log_std))
            total = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
            approx_kl = jnp.mean(batch["log_prob"] - log_prob)
            clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_epsilon)
            return total, (policy_loss, value_loss, entropy, approx_kl, clip_fraction)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, (loss, *aux)

    def _optimize(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        count = int(batch["obs"].shape[0])
        advantage = batch["advantage"].astype(np.float32)
        batch["advantage"] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        usable_keys = ("obs", "raw_action", "log_prob", "advantage", "return")
        losses = []
        minibatch = min(self.config.minibatch_size, count)
        for _epoch in range(self.config.update_epochs):
            self.rng, permutation_key = jax.random.split(self.rng)
            indices = np.asarray(jax.random.permutation(permutation_key, count))
            for start in range(0, count, minibatch):
                selected = indices[start : start + minibatch]
                mb = {key: jnp.asarray(batch[key][selected]) for key in usable_keys}
                self.state, values = self._update_jit(self.state, mb)
                losses.append(np.asarray(values, dtype=np.float64))
        mean = np.mean(np.stack(losses), axis=0)
        names = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        return {name: float(value) for name, value in zip(names, mean, strict=True)}

    def save(self, update: int, total_steps: int, metrics: dict[str, Any]) -> Path:
        checkpoint = self.output_dir / f"policy_{total_steps:09d}.msgpack"
        metadata = {
            "step": int(self.state.step),
            "update": int(update),
            "total_steps": int(total_steps),
            "observation_size": int(self.env.observation_size),
            "action_size": int(self.env.action_size),
            "ppo_config": asdict(self.config),
            "replay_config": asdict(self.env.config),
            "dataset_paths": [str(path) for path in self.env.dataset_paths],
        }
        # Flax msgpack uses strict types. Keep the PyTree binary and store all
        # Python tuples/configuration as a JSON string for portable loading.
        payload = {
            "params": self.state.params,
            "metadata_json": json.dumps(metadata, sort_keys=True),
        }
        checkpoint.write_bytes(serialization.msgpack_serialize(payload))
        metrics_path = self.output_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        return checkpoint

    def train(self) -> dict[str, Any]:
        started = time.time()
        total_steps = 0
        update = 0
        last_metrics: dict[str, Any] = {}
        while total_steps < self.config.total_steps:
            steps = min(self.config.rollout_steps, self.config.total_steps - total_steps)
            batch = self._collect(steps)
            optimization = self._optimize(batch)
            total_steps += steps
            update += 1
            returns = batch["episode_returns"]
            last_metrics = {
                "update": update,
                "total_steps": total_steps,
                "mean_step_reward": float(np.mean(batch["reward"])),
                "mean_episode_return": float(np.mean(returns)) if returns.size else None,
                "episodes": int(returns.size),
                "elapsed_seconds": float(time.time() - started),
                **optimization,
            }
            checkpoint_due = update % max(1, self.config.checkpoint_every_updates) == 0
            if checkpoint_due or total_steps >= self.config.total_steps:
                last_metrics["checkpoint"] = str(self.save(update, total_steps, last_metrics))
            print(json.dumps(last_metrics, sort_keys=True), flush=True)
        return last_metrics


def load_policy_checkpoint(path: str | Path) -> tuple[ActorCritic, Any, dict[str, Any]]:
    """Load a policy checkpoint for deterministic evaluation or deployment tests."""

    payload = serialization.msgpack_restore(Path(path).read_bytes())
    metadata = json.loads(payload["metadata_json"])
    ppo = metadata["ppo_config"]
    model = ActorCritic(
        action_size=int(metadata["action_size"]),
        hidden_sizes=tuple(int(width) for width in ppo["hidden_sizes"]),
        initial_log_std=float(ppo["initial_log_std"]),
    )
    return model, payload["params"], metadata
