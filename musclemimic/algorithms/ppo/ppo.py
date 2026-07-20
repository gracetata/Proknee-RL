"""
PPO algorithm class.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from omegaconf import ListConfig, open_dict

from musclemimic.algorithms import ActorCritic, JaxRLAlgorithmBase, SplitActionActorCritic
from musclemimic.algorithms.common.moe_networks import SoftMoEActorCritic
from musclemimic.algorithms.ppo.checkpoint import load_checkpoint_for_resume
from musclemimic.algorithms.ppo.config import PPOAgentConf, PPOAgentState, get_ppo_config
from musclemimic.algorithms.common.env_utils import expand_obs_indices_for_history, wrap_env
from musclemimic.algorithms.ppo.inference import play_policy, play_policy_mujoco
from musclemimic.algorithms.common.optimizer import get_optimizer
from musclemimic.algorithms.ppo.runner import train

if TYPE_CHECKING:
    from musclemimic.utils.metrics import MetricsHandler


logger = logging.getLogger(__name__)


class PPOJax(JaxRLAlgorithmBase):
    """PPO algorithm for JAX-based training."""

    _agent_conf = PPOAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def init_agent_conf(cls, env: Any, config: Any) -> PPOAgentConf:
        """
        Initialize agent configuration.

        Args:
            env: environment instance
            config: hydra config

        Returns:
            PPOAgentConf with network and optimizer
        """
        # Merge ppo_config into experiment for backward compatibility
        exp = get_ppo_config(config.experiment)
        adaptive_cfg = exp.get("adaptive_sampling", {})
        if adaptive_cfg.get("enabled", False) and exp.get("n_trajectories", None) is None:
            base_env = env
            if hasattr(env, "unwrapped"):
                unwrapped_attr = getattr(env, "unwrapped")
                base_env = unwrapped_attr() if callable(unwrapped_attr) else unwrapped_attr
            if hasattr(base_env, "th") and base_env.th is not None:
                with open_dict(exp):
                    exp.n_trajectories = int(base_env.th.n_trajectories)
            else:
                raise ValueError(
                    "adaptive_sampling enabled but env has no trajectory handler to derive n_trajectories"
                )

        # compute derived config values
        with open_dict(exp):
            exp.num_updates = exp.total_timesteps // exp.num_steps // exp.num_envs
            exp.minibatch_size = exp.num_envs * exp.num_steps // exp.num_minibatches
            exp.validation_interval = exp.num_updates // exp.validation.num
            exp.validation.num = int(exp.num_updates // exp.validation_interval)

        # utd override
        with open_dict(exp):
            if "utd" in exp and exp.utd is not None:
                exp.update_epochs = int(exp.utd)

        # compute training metrics
        batch_size = int(exp.num_envs) * int(exp.num_steps)
        gradient_steps = int(exp.update_epochs) * int(exp.num_minibatches)

        # standard utd: gradient_updates / samples_collected
        utd = float(gradient_steps) / float(batch_size)
        # sample reuse: how many times each sample is used (= epochs)
        sample_reuse = float(exp.update_epochs)
        # update frequency
        unrolls_per_1m = 1_000_000.0 / float(batch_size)
        grad_steps_per_1m = unrolls_per_1m * float(gradient_steps)

        with open_dict(exp):
            exp.effective_utd = utd
            exp.sample_reuse = sample_reuse
            exp.unrolls_per_1m = unrolls_per_1m
            exp.updates_per_1m = unrolls_per_1m
            exp.grad_steps_per_1m = grad_steps_per_1m

        # Update config.experiment with merged + derived values for downstream use
        with open_dict(config):
            config.experiment = exp

        msg = (
            f"[ppo] roll out size={batch_size}, minibatch size={int(exp.minibatch_size)}, "
            f"grad_steps={gradient_steps}, utd={utd:.4f}, "
            f"sample_reuse={sample_reuse:.0f}x, unrolls/1M={unrolls_per_1m:.1f}, "
            f"grad_steps/1M={grad_steps_per_1m:.1f}"
        )
        if logger.isEnabledFor(logging.INFO) and (logger.handlers or logging.getLogger().handlers):
            logger.info(msg)
        else:
            print(msg)

        network = cls._create_network(env, config)
        tx = get_optimizer(config.experiment)

        return cls._agent_conf(config, network, tx)

    @classmethod
    def _create_network(cls, env: Any, config: Any) -> ActorCritic | SoftMoEActorCritic:
        """Create actor-critic network."""
        exp = config.experiment

        # parse hidden layers
        actor_hidden = cls._parse_hidden_layers(exp.actor_hidden_layers)
        critic_hidden = cls._parse_hidden_layers(exp.critic_hidden_layers)

        # observation indices for actor/critic
        if hasattr(exp, "actor_obs_group") and exp.actor_obs_group is not None:
            actor_obs_ind = env.obs_container.get_obs_ind_by_group(exp.actor_obs_group)
        else:
            actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])

        if hasattr(exp, "critic_obs_group") and exp.critic_obs_group is not None:
            critic_obs_ind = env.obs_container.get_obs_ind_by_group(exp.critic_obs_group)
        else:
            critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])

        if hasattr(exp, "len_obs_history") and exp.len_obs_history > 1:
            actor_obs_ind = expand_obs_indices_for_history(actor_obs_ind, env, exp)
            critic_obs_ind = expand_obs_indices_for_history(critic_obs_ind, env, exp)

        use_moe = exp.get("use_moe", False)
        use_layernorm = exp.get("use_layernorm", False)
        layernorm_eps = exp.get("layernorm_eps", 1e-5)

        # Residual network options
        use_residual = exp.get("use_residual", False)
        residual_type = exp.get("residual_type", "gated")
        residual_gate_init = exp.get("residual_gate_init", -2.0)

        split_action = exp.get("split_action_actor", {}) or {}
        if split_action.get("enabled", False):
            if use_moe:
                raise ValueError("split_action_actor is not compatible with use_moe=True")
            action_dim = int(env.info.action_space.shape[0])
            prosthesis_action_dim = int(split_action.get("prosthesis_action_dim", 4))
            remaining_action_dim = int(split_action.get("remaining_action_dim", action_dim - prosthesis_action_dim))
            return SplitActionActorCritic(
                action_dim=action_dim,
                remaining_action_dim=remaining_action_dim,
                prosthesis_action_dim=prosthesis_action_dim,
                activation=exp.activation,
                init_std=exp.init_std,
                learnable_std=exp.learnable_std,
                hidden_layer_dims=actor_hidden,
                critic_hidden_layer_dims=critic_hidden if critic_hidden != actor_hidden else None,
                actor_obs_ind=actor_obs_ind,
                critic_obs_ind=critic_obs_ind,
                use_layernorm=use_layernorm,
                layernorm_eps=layernorm_eps,
            )

        if use_moe:
            moe_config = exp.get("moe_config", {})
            return SoftMoEActorCritic(
                action_dim=env.info.action_space.shape[0],
                activation=exp.activation,
                init_std=exp.init_std,
                learnable_std=exp.learnable_std,
                hidden_layer_dims=actor_hidden,  # moe uses same for both
                actor_obs_ind=actor_obs_ind,
                critic_obs_ind=critic_obs_ind,
                num_experts=moe_config.get("num_experts", 8),
                moe_at_layers=moe_config.get("moe_at_layers", (1,)),
                apply_moe_to=moe_config.get("apply_moe_to", "both"),
                temperature=moe_config.get("temperature", 1.0),
                load_balance_loss_weight=moe_config.get("load_balance_loss_weight", 0.01),
                use_layernorm=use_layernorm,
                layernorm_eps=layernorm_eps,
            )
        else:
            return ActorCritic(
                env.info.action_space.shape[0],
                activation=exp.activation,
                init_std=exp.init_std,
                learnable_std=exp.learnable_std,
                hidden_layer_dims=actor_hidden,
                critic_hidden_layer_dims=critic_hidden if critic_hidden != actor_hidden else None,
                actor_obs_ind=actor_obs_ind,
                critic_obs_ind=critic_obs_ind,
                use_layernorm=use_layernorm,
                layernorm_eps=layernorm_eps,
                use_residual=use_residual,
                residual_type=residual_type,
                residual_gate_init=residual_gate_init,
            )

    @classmethod
    def _parse_hidden_layers(cls, layers: Any) -> list[int]:
        """Parse hidden layers from config value."""
        if isinstance(layers, list | ListConfig):
            return list(layers)
        return ast.literal_eval(layers)

    @classmethod
    def _get_optimizer(cls, config: Any):
        """Get optimizer (delegates to optimizer module)."""
        return get_optimizer(config.experiment)

    @classmethod
    def _train_fn(
        cls,
        rng: jax.Array,
        env: Any,
        agent_conf: PPOAgentConf,
        agent_state: PPOAgentState | None = None,
        mh: MetricsHandler | None = None,
        online_logging_callback: Callable[[dict], None] | None = None,
        logging_interval: int = 10,
        resume_info: dict | None = None,
        val_env: Any | None = None,
        apply_resume_resets: bool = True,
    ) -> dict[str, Any]:
        """Run PPO training (delegates to runner module)."""
        return train(
            rng=rng,
            env=env,
            agent_conf=agent_conf,
            agent_state_cls=cls._agent_state,
            agent_state=agent_state,
            mh=mh,
            online_logging_callback=online_logging_callback,
            logging_interval=logging_interval,
            resume_info=resume_info,
            val_env=val_env,
            apply_resume_resets=apply_resume_resets,
        )

    @classmethod
    def _wrap_env(cls, env: Any, config: Any) -> Any:
        """Wrap environment (delegates to env_utils module)."""
        return wrap_env(env, config)

    @classmethod
    def play_policy(
        cls,
        env: Any,
        agent_conf: PPOAgentConf,
        agent_state: PPOAgentState,
        n_envs: int,
        n_steps: int | None = None,
        render: bool = True,
        record: bool = False,
        rng: jax.Array | None = None,
        deterministic: bool = False,
        use_mujoco: bool = False,
        wrap_env: bool = True,
        train_state_seed: int | None = None,
        sequential_mjx: bool = False,
    ) -> None:
        """Run policy for visualization (delegates to inference module)."""
        play_policy(
            env,
            agent_conf,
            agent_state,
            n_envs,
            n_steps,
            render,
            record,
            rng,
            deterministic,
            use_mujoco,
            wrap_env,
            train_state_seed,
            sequential_mjx,
        )

    @classmethod
    def play_policy_mujoco(
        cls,
        env: Any,
        agent_conf: PPOAgentConf,
        agent_state: PPOAgentState,
        n_steps: int | None = None,
        render: bool = True,
        record: bool = False,
        rng: jax.Array | None = None,
        deterministic: bool = False,
        train_state_seed: int | None = None,
    ) -> None:
        """Run policy with mujoco backend (delegates to inference module)."""
        play_policy_mujoco(
            env,
            agent_conf,
            agent_state,
            n_steps,
            render,
            record,
            rng,
            deterministic,
            train_state_seed,
        )

    @classmethod
    def build_resume_train_fn(
        cls,
        env: Any,
        agent_conf: PPOAgentConf,
        mh: MetricsHandler | None = None,
        online_logging_callback: Callable[[dict], None] | None = None,
        logging_interval: int = 10,
    ) -> Callable[[jax.Array, PPOAgentState], dict[str, Any]]:
        """Return a train fn that resumes from a provided agent_state."""
        return lambda rng_key, agent_state: cls._train_fn(
            rng_key,
            env,
            agent_conf,
            agent_state,
            mh=mh,
            online_logging_callback=online_logging_callback,
            logging_interval=logging_interval,
        )

    @classmethod
    def build_resume_train_fn_from_path(
        cls,
        env: Any,
        agent_conf: PPOAgentConf,
        checkpoint_path: str,
        mh: MetricsHandler | None = None,
        online_logging_callback: Callable[[dict], None] | None = None,
        logging_interval: int = 10,
        val_env: Any | None = None,
        apply_resume_resets: bool = True,
    ) -> Callable[[jax.Array], dict[str, Any]]:
        """Load checkpoint and return a train fn that only needs RNG."""
        loaded_state, resume_info = load_checkpoint_for_resume(checkpoint_path, agent_conf)

        return lambda rng_key: cls._train_fn(
            rng_key,
            env,
            agent_conf,
            loaded_state,
            mh=mh,
            online_logging_callback=online_logging_callback,
            logging_interval=logging_interval,
            resume_info=resume_info,
            val_env=val_env,
            apply_resume_resets=apply_resume_resets,
        )
