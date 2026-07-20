"""Policy wrappers used by prosthesis distillation scripts."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.ppo.inference import ObservationHistoryBuffer
from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint


@dataclass
class PolicyRunner:
    agent_conf: object
    agent_state: object
    deterministic: bool = True
    seed: int = 0
    obs_buffer: ObservationHistoryBuffer | None = None

    def __post_init__(self):
        self.rng = jax.random.key(self.seed)
        self.train_state = self.agent_state.train_state
        self._act = jax.jit(self._act_impl)

    @classmethod
    def from_checkpoint(cls, checkpoint: str, env, *, deterministic: bool = True, seed: int = 0):
        config, agent_state, _metadata = load_checkpoint(checkpoint)
        OmegaConf.set_struct(config, False)
        agent_conf = PPOJax.init_agent_conf(env, config)
        agent_state = align_agent_state(agent_state, agent_conf)
        return cls(
            agent_conf=agent_conf,
            agent_state=agent_state,
            deterministic=deterministic,
            seed=seed,
            obs_buffer=build_obs_buffer(config.experiment, env),
        )

    @classmethod
    def from_agent_state(cls, agent_conf, agent_state, env, *, deterministic: bool = True, seed: int = 0):
        return cls(
            agent_conf=agent_conf,
            agent_state=agent_state,
            deterministic=deterministic,
            seed=seed,
            obs_buffer=build_obs_buffer(agent_conf.config.experiment, env),
        )

    def reset_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_buffer is None:
            return np.asarray(obs).reshape(-1)
        return self.obs_buffer.reset(obs)

    def update_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_buffer is None:
            return np.asarray(obs).reshape(-1)
        return self.obs_buffer.step(obs)

    def _act_impl(self, train_state, obs, rng):
        obs = jnp.asarray(obs)
        obs = jnp.atleast_2d(obs)
        y, updates = self.agent_conf.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs,
            mutable=["run_stats"],
        )
        pi, value = y
        action = pi.mean() if self.deterministic else pi.sample(seed=rng)
        return action, value, train_state.replace(run_stats=updates["run_stats"])

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.rng, subkey = jax.random.split(self.rng)
        action, value, train_state = self._act(self.train_state, np.asarray(obs), subkey)
        self.train_state = train_state
        action_np = np.asarray(action)
        if action_np.ndim > 1 and action_np.shape[0] == 1:
            action_np = action_np[0]
        return action_np, np.asarray(value)


def build_obs_buffer(exp_cfg, env) -> ObservationHistoryBuffer | None:
    len_obs_history = int(getattr(exp_cfg, "len_obs_history", 1))
    if len_obs_history <= 1:
        return None
    split_goal = bool(getattr(exp_cfg, "split_goal", False))
    if not split_goal:
        return ObservationHistoryBuffer(len_obs_history)
    if not hasattr(env, "obs_container"):
        raise ValueError("split_goal=True requires env.obs_container")
    goal_indices = np.asarray(env.obs_container.get_obs_ind_by_group("goal"), dtype=int)
    raw_obs_dim = int(env.info.observation_space.shape[0])
    state_mask = np.ones(raw_obs_dim, dtype=bool)
    state_mask[goal_indices] = False
    state_indices = np.arange(raw_obs_dim, dtype=int)[state_mask]
    return ObservationHistoryBuffer(
        len_obs_history,
        split_goal=True,
        state_indices=state_indices,
        goal_indices=goal_indices,
    )
