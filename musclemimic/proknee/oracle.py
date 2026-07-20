"""Frozen MuscleMimic oracle policy used as Stage 0."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.ppo.inference import ObservationHistoryBuffer
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint


@dataclass
class OracleState:
    train_state: object
    rng: jax.Array
    obs_buffer: ObservationHistoryBuffer | None


class FrozenMMOracle:
    """Small inference wrapper around the official `mm-10m-2` policy."""

    def __init__(self, checkpoint_path: str, env, seed: int = 0, deterministic: bool = False):
        config, agent_state, metadata = load_checkpoint(checkpoint_path)
        OmegaConf.set_struct(config, False)
        apply_temporal_params(config)

        self.config = config
        self.metadata = metadata
        self.agent_conf = PPOJax.init_agent_conf(env, config)
        self.agent_state = align_agent_state(agent_state, self.agent_conf)
        train_state = self.agent_state.train_state

        if deterministic:
            train_state = train_state.replace(
                params={**train_state.params, "log_std": np.ones_like(train_state.params["log_std"]) * -np.inf}
            )

        self._sample = jax.jit(self._sample_actions)
        self.state = OracleState(
            train_state=train_state,
            rng=jax.random.key(seed),
            obs_buffer=self._build_obs_buffer(env),
        )

    def _build_obs_buffer(self, env):
        exp_cfg = self.config.experiment
        len_obs_history = int(getattr(exp_cfg, "len_obs_history", 1))
        if len_obs_history <= 1:
            return None

        split_goal = bool(getattr(exp_cfg, "split_goal", False))
        if not split_goal:
            return ObservationHistoryBuffer(len_obs_history)

        if not hasattr(env, "obs_container"):
            raise ValueError("split_goal=True requires env.obs_container")
        goal_indices = np.asarray(env.obs_container.get_obs_ind_by_group("goal"), dtype=int)
        raw_obs_dim = env.info.observation_space.shape[0]
        state_mask = np.ones(raw_obs_dim, dtype=bool)
        state_mask[goal_indices] = False
        state_indices = np.arange(raw_obs_dim, dtype=int)[state_mask]
        return ObservationHistoryBuffer(
            len_obs_history,
            split_goal=True,
            state_indices=state_indices,
            goal_indices=goal_indices,
        )

    def reset(self, obs: np.ndarray) -> np.ndarray:
        if self.state.obs_buffer is None:
            return np.asarray(obs).flatten()
        return self.state.obs_buffer.reset(obs)

    def update_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.state.obs_buffer is None:
            return np.asarray(obs).flatten()
        return self.state.obs_buffer.step(obs)

    def _sample_actions(self, train_state, obs, rng):
        rng, subkey = jax.random.split(rng)
        y, updates = self.agent_conf.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs,
            mutable=["run_stats"],
        )
        pi, _value = y
        train_state = train_state.replace(run_stats=updates["run_stats"])
        action = pi.sample(seed=subkey)
        return action, train_state, rng

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_jax = jnp.asarray(np.atleast_2d(obs))
        action, train_state, rng = self._sample(self.state.train_state, obs_jax, self.state.rng)
        self.state.train_state = train_state
        self.state.rng = rng
        return np.asarray(action)[0]
