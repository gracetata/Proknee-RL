"""JAX/Flax Teacher-Student models for MuscleMimic-ProKnee."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class PrivilegedMLP(nn.Module):
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 128)

    @nn.compact
    def __call__(self, priv_info):
        x = priv_info
        for width in self.hidden_dims:
            x = nn.elu(nn.Dense(width)(x))
        return nn.tanh(nn.Dense(self.latent_dim)(x))


class ProprioAdaptTConv(nn.Module):
    latent_dim: int = 32
    channels: int = 32

    @nn.compact
    def __call__(self, proprio_hist):
        # proprio_hist: [B, T, D]
        x = nn.elu(nn.Dense(self.channels)(proprio_hist))
        x = nn.elu(nn.Conv(self.channels, kernel_size=(9,), strides=(2,), padding="SAME")(x))
        x = nn.elu(nn.Conv(self.channels, kernel_size=(5,), strides=(1,), padding="SAME")(x))
        x = nn.elu(nn.Conv(self.channels, kernel_size=(3,), strides=(1,), padding="SAME")(x))
        x = x.reshape((x.shape[0], -1))
        return nn.tanh(nn.Dense(self.latent_dim)(x))


class ProKneeActor(nn.Module):
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 128, 64)

    @nn.compact
    def __call__(self, obs, latent):
        x = jnp.concatenate([obs, latent], axis=-1)
        for width in self.hidden_dims:
            x = nn.elu(nn.Dense(width)(x))
        return nn.Dense(self.action_dim)(x)


class MuscleProKneeTeacher(nn.Module):
    action_dim: int
    latent_dim: int = 32

    @nn.compact
    def __call__(self, obs, priv_info, return_torque: bool = False):
        latent = PrivilegedMLP(latent_dim=self.latent_dim, name="priv_mlp")(priv_info)
        action = ProKneeActor(action_dim=self.action_dim, name="actor")(obs, latent)
        if return_torque:
            torque = ProKneeActor(action_dim=self.action_dim, name="torque_actor")(obs, latent)
            return action, torque, latent
        return action, latent


class MuscleProKneeStudent(nn.Module):
    action_dim: int
    latent_dim: int = 32

    @nn.compact
    def __call__(self, obs, priv_info=None, proprio_hist=None, mode: str = "student"):
        if mode == "teacher":
            if priv_info is None:
                raise ValueError("priv_info is required in teacher mode")
            latent = PrivilegedMLP(latent_dim=self.latent_dim, name="priv_mlp")(priv_info)
        else:
            if proprio_hist is None:
                raise ValueError("proprio_hist is required in student mode")
            latent = ProprioAdaptTConv(latent_dim=self.latent_dim, name="adapt_tconv")(proprio_hist)
        action = ProKneeActor(action_dim=self.action_dim, name="actor")(obs, latent)
        return action, latent
