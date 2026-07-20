import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):

    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            if self.use_layernorm:
                x = nn.LayerNorm(epsilon=self.layernorm_eps)(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x


class ResidualFCNet(nn.Module):
    """
    Residual MLP with gated skip connections and projection shortcuts.

    Pairs consecutive hidden layers into residual blocks:
    - Block structure: Dense(d0)->LN->act -> Dense(d1)->LN -> skip+gate*h -> act
    - When input dim != d1, uses a projection Dense on the skip path
    - Gated residual (default): gate = sigmoid(g), g initialized to -2 (~0.12)
    - Second Dense in each block uses small init (0.01) for near-identity at init

    Args:
        hidden_layer_dims: Sequence of hidden layer dimensions (paired into blocks)
        output_dim: Output dimension
        activation: Activation function name (e.g., "silu", "tanh")
        use_layernorm: Whether to apply LayerNorm after each Dense
        layernorm_eps: LayerNorm epsilon
        residual_type: "gated" (learnable gate) or "fixed" (constant scale)
        residual_scale: Scale factor for fixed residual (default 0.1)
        residual_gate_init: Initial value for gated residual (default -2.0, ~0.12 after sigmoid)
        residual_second_gain: Orthogonal init gain for 2nd Dense in block (default 0.01)
        proj_gain: Orthogonal init gain for projection shortcuts (default 1.0)
    """
    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "silu"
    use_layernorm: bool = True
    layernorm_eps: float = 1e-5
    residual_type: str = "gated"
    residual_scale: float = 0.1
    residual_gate_init: float = -2.0
    residual_second_gain: float = 0.01
    proj_gain: float = 1.0

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):
        act = self.activation_fn
        dims = list(self.hidden_layer_dims)
        n = len(dims)

        # Helper: Dense -> (LN) -> act
        def dense_ln_act(x_in, dim, gain, name_prefix):
            y = nn.Dense(
                dim,
                kernel_init=orthogonal(gain),
                bias_init=constant(0.0),
                name=f"{name_prefix}_dense",
            )(x_in)
            if self.use_layernorm:
                y = nn.LayerNorm(epsilon=self.layernorm_eps, name=f"{name_prefix}_ln")(y)
            return act(y)

        # Helper: Dense -> (LN) without activation
        def dense_ln(x_in, dim, gain, name_prefix):
            y = nn.Dense(
                dim,
                kernel_init=orthogonal(gain),
                bias_init=constant(0.0),
                name=f"{name_prefix}_dense",
            )(x_in)
            if self.use_layernorm:
                y = nn.LayerNorm(epsilon=self.layernorm_eps, name=f"{name_prefix}_ln")(y)
            return y

        i = 0
        block_idx = 0
        while i + 1 < n:
            d0, d1 = dims[i], dims[i + 1]

            # Main branch: Dense(d0)->LN->act -> Dense(d1)->LN
            h = dense_ln_act(x, d0, np.sqrt(2.0), f"block{block_idx}_layer0")
            h = dense_ln(h, d1, self.residual_second_gain, f"block{block_idx}_layer1")

            # Skip/projection path
            if x.shape[-1] != d1:
                skip = nn.Dense(
                    d1,
                    kernel_init=orthogonal(self.proj_gain),
                    bias_init=constant(0.0),
                    name=f"block{block_idx}_proj",
                )(x)
            else:
                skip = x

            # Residual weight (gated or fixed)
            if self.residual_type == "gated":
                g = self.param(f"res_gate_{block_idx}", constant(self.residual_gate_init), (1,))
                w = nn.sigmoid(g)[0]
            else:
                w = self.residual_scale

            # Combine and activate
            x = act(skip + w * h)
            i += 2
            block_idx += 1

        # Tail layer if odd number of hidden dims
        if i < n:
            x = dense_ln_act(x, dims[i], np.sqrt(2.0), "tail")

        # Output head (same as FullyConnectedNet: gain=0.01)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="output",
        )(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    critic_hidden_layer_dims: Sequence[int] | None = None
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5
    # Residual network options
    use_residual: bool = False
    residual_type: str = "gated"
    residual_gate_init: float = -2.0

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        if self.use_residual:
            actor_mean = ResidualFCNet(
                hidden_layer_dims=self.hidden_layer_dims,
                output_dim=self.action_dim,
                activation=self.activation,
                use_layernorm=self.use_layernorm,
                layernorm_eps=self.layernorm_eps,
                residual_type=self.residual_type,
                residual_gate_init=self.residual_gate_init,
                name="actor",
            )(actor_x)
        else:
            actor_mean = FullyConnectedNet(
                hidden_layer_dims=self.hidden_layer_dims,
                output_dim=self.action_dim,
                activation=self.activation,
                output_activation=None,
                use_running_mean_stand=False,
                squeeze_output=False,
                use_layernorm=self.use_layernorm,
                layernorm_eps=self.layernorm_eps,
                name="actor",
            )(actor_x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic_dims = self.critic_hidden_layer_dims if self.critic_hidden_layer_dims is not None else self.hidden_layer_dims
        if self.use_residual:
            critic = ResidualFCNet(
                hidden_layer_dims=critic_dims,
                output_dim=1,
                activation=self.activation,
                use_layernorm=self.use_layernorm,
                layernorm_eps=self.layernorm_eps,
                residual_type=self.residual_type,
                residual_gate_init=self.residual_gate_init,
                name="critic",
            )(critic_x)
        else:
            critic = FullyConnectedNet(
                hidden_layer_dims=critic_dims,
                output_dim=1,
                activation=self.activation,
                output_activation=None,
                use_running_mean_stand=False,
                squeeze_output=False,
                use_layernorm=self.use_layernorm,
                layernorm_eps=self.layernorm_eps,
                name="critic",
            )(critic_x)

        return pi, jnp.squeeze(critic, axis=-1)


class SplitActionActorCritic(nn.Module):
    """Actor-critic with explicit remaining-muscle and prosthesis action heads."""

    action_dim: int
    remaining_action_dim: int
    prosthesis_action_dim: int = 4
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    critic_hidden_layer_dims: Sequence[int] | None = None
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5

    def setup(self):
        if int(self.remaining_action_dim) + int(self.prosthesis_action_dim) != int(self.action_dim):
            raise ValueError(
                "SplitActionActorCritic dimensions mismatch: "
                f"remaining={self.remaining_action_dim} prosthesis={self.prosthesis_action_dim} action_dim={self.action_dim}"
            )

    @nn.compact
    def __call__(self, x):
        x = RunningMeanStd()(x)

        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        remaining_mean = FullyConnectedNet(
            hidden_layer_dims=self.hidden_layer_dims,
            output_dim=self.remaining_action_dim,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=False,
            squeeze_output=False,
            use_layernorm=self.use_layernorm,
            layernorm_eps=self.layernorm_eps,
            name="actor_remaining_muscles",
        )(actor_x)
        prosthesis_mean = FullyConnectedNet(
            hidden_layer_dims=self.hidden_layer_dims,
            output_dim=self.prosthesis_action_dim,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=False,
            squeeze_output=False,
            use_layernorm=self.use_layernorm,
            layernorm_eps=self.layernorm_eps,
            name="actor_prosthesis",
        )(actor_x)
        actor_mean = jnp.concatenate([remaining_mean, prosthesis_mean], axis=-1)
        actor_logtstd = self.param(
            "log_std",
            nn.initializers.constant(jnp.log(self.init_std)),
            (self.action_dim,),
        )
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic_dims = self.critic_hidden_layer_dims if self.critic_hidden_layer_dims is not None else self.hidden_layer_dims
        critic = FullyConnectedNet(
            hidden_layer_dims=critic_dims,
            output_dim=1,
            activation=self.activation,
            output_activation=None,
            use_running_mean_stand=False,
            squeeze_output=False,
            use_layernorm=self.use_layernorm,
            layernorm_eps=self.layernorm_eps,
            name="critic",
        )(critic_x)
        return pi, jnp.squeeze(critic, axis=-1)


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and variance for input normalization."""

    @nn.compact
    def __call__(self, x):

        x = jnp.atleast_2d(x)

        # Initialize running mean, variance, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
        batch_count = x.shape[0]

        # Update counts
        updated_count = count.value + batch_count

        # Numerically stable mean and variance update
        delta = batch_mean - mean.value
        new_mean = mean.value + delta * batch_count / updated_count

        # Compute the new variance using Welford's method
        m_a = var.value * count.value
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
        new_var = M2 / updated_count

        # Normalize input
        normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

        # Update state variables
        mean.value = new_mean
        var.value = new_var
        count.value = updated_count

        return jnp.squeeze(normalized_x)
