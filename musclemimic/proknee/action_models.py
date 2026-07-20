"""Policies for closed-loop prosthesis action training."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .models import PrivilegedMLP, ProKneeActor


class CoupledActionPolicy(nn.Module):
    """Privileged prosthesis action policy with optional body residual adapter."""

    prosthesis_action_dim: int = 4
    body_residual_dim: int = 0
    latent_dim: int = 32
    action_scale: tuple[float, ...] = (110.0, 65.0, 45.0, 22.0)
    init_log_std: float = -2.5

    @nn.compact
    def __call__(self, obs, priv_info, *, body_residual: bool = False):
        latent = PrivilegedMLP(latent_dim=self.latent_dim, name="priv_mlp")(priv_info)
        raw_torque = ProKneeActor(action_dim=self.prosthesis_action_dim, name="prosthesis_actor")(obs, latent)
        scale = jnp.asarray(self.action_scale, dtype=raw_torque.dtype)
        prosthesis_mean = nn.tanh(raw_torque / jnp.maximum(scale, 1e-6))
        prosthesis_log_std = self.param(
            "prosthesis_log_std",
            nn.initializers.constant(self.init_log_std),
            (self.prosthesis_action_dim,),
        )
        if body_residual and self.body_residual_dim > 0:
            body_mean = nn.tanh(ProKneeActor(action_dim=self.body_residual_dim, name="body_residual_actor")(obs, latent))
            body_log_std = self.param(
                "body_log_std",
                nn.initializers.constant(self.init_log_std),
                (self.body_residual_dim,),
            )
        else:
            body_mean = jnp.zeros((obs.shape[0], 0), dtype=obs.dtype)
            body_log_std = jnp.zeros((0,), dtype=obs.dtype)
        return {
            "prosthesis_mean": prosthesis_mean,
            "prosthesis_log_std": prosthesis_log_std,
            "body_mean": body_mean,
            "body_log_std": body_log_std,
            "latent": latent,
        }

