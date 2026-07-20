"""Minimal adapters for the official MuscleMimic environment and checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

import musclemimic.environments  # noqa: F401 - register environments
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.ppo.inference import ObservationHistoryBuffer
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint


def build_fullbody_env(
    checkpoint_path: str,
    motion_path: str,
    *,
    fixed_start_step: int = 0,
) -> tuple[Any, Any, Any, dict]:
    """Build the CPU MyoFullBody tracker environment for one motion."""

    config, agent_state, metadata = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)
    apply_temporal_params(config)

    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBody"
    env_params["headless"] = True
    env_params.pop("prosthesis", None)
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update(
        {
            "random_start": False,
            "fixed_start_conf": [0, int(fixed_start_step)],
            "start_from_random_step": False,
        }
    )
    env_params["th_params"] = th_params

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["dataset_group"] = None
    amass["rel_dataset_path"] = [str(motion_path)]
    task_params["amass_dataset_conf"] = amass
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**{**env_params, **task_params})
    return env, config, agent_state, metadata


def _build_obs_buffer(experiment_config, env) -> ObservationHistoryBuffer | None:
    history = int(getattr(experiment_config, "len_obs_history", 1))
    if history <= 1:
        return None
    split_goal = bool(getattr(experiment_config, "split_goal", False))
    if not split_goal:
        return ObservationHistoryBuffer(history)
    goal_indices = np.asarray(env.obs_container.get_obs_ind_by_group("goal"), dtype=np.int32)
    raw_dim = int(env.info.observation_space.shape[0])
    keep = np.ones(raw_dim, dtype=bool)
    keep[goal_indices] = False
    state_indices = np.arange(raw_dim, dtype=np.int32)[keep]
    return ObservationHistoryBuffer(
        history,
        split_goal=True,
        state_indices=state_indices,
        goal_indices=goal_indices,
    )


@dataclass
class OfficialPolicyRunner:
    """Deterministic checkpoint inference without legacy prosthesis code."""

    agent_conf: Any
    train_state: Any
    obs_buffer: ObservationHistoryBuffer | None
    rng: jax.Array
    deterministic: bool = True

    @classmethod
    def create(
        cls,
        env,
        config,
        agent_state,
        *,
        seed: int = 0,
        train_state_seed: int = 0,
        deterministic: bool = True,
    ) -> "OfficialPolicyRunner":
        agent_conf = PPOJax.init_agent_conf(env, config)
        agent_state = align_agent_state(agent_state, agent_conf)
        train_state = agent_state.train_state
        if int(config.experiment.n_seeds) > 1:
            train_state = jax.tree.map(lambda x: x[int(train_state_seed)], train_state)
        result = cls(
            agent_conf=agent_conf,
            train_state=train_state,
            obs_buffer=_build_obs_buffer(config.experiment, env),
            rng=jax.random.key(int(seed)),
            deterministic=bool(deterministic),
        )
        result._act_jit = jax.jit(result._act_impl)
        return result

    def reset_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_buffer is None:
            return np.asarray(obs).reshape(-1)
        return self.obs_buffer.reset(obs)

    def update_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_buffer is None:
            return np.asarray(obs).reshape(-1)
        return self.obs_buffer.step(obs)

    def _act_impl(self, train_state, obs, rng):
        values, updates = self.agent_conf.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            jnp.atleast_2d(jnp.asarray(obs)),
            mutable=["run_stats"],
        )
        distribution, value = values
        action = distribution.mean() if self.deterministic else distribution.sample(seed=rng)
        return action, value, train_state.replace(run_stats=updates["run_stats"])

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.rng, subkey = jax.random.split(self.rng)
        action, value, self.train_state = self._act_jit(self.train_state, np.asarray(obs), subkey)
        action_np = np.asarray(action)
        if action_np.ndim == 2 and action_np.shape[0] == 1:
            action_np = action_np[0]
        return action_np.astype(np.float32), np.asarray(value)

