from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from torque_replay_training.ppo import ActorCritic, _gae, _normal_log_prob


def test_initial_policy_mean_is_zero() -> None:
    model = ActorCritic(action_size=4, hidden_sizes=(16,), initial_log_std=-2.0)
    obs = jnp.ones((3, 11))
    params = model.init(jax.random.key(0), obs)
    mean, log_std, value = model.apply(params, obs)
    np.testing.assert_array_equal(np.asarray(mean), np.zeros((3, 4)))
    assert log_std.shape == (4,)
    assert value.shape == (3,)


def test_log_prob_and_gae_are_finite() -> None:
    raw = jnp.zeros((2, 4))
    log_prob = _normal_log_prob(raw, jnp.zeros_like(raw), jnp.full((4,), -2.0))
    assert np.all(np.isfinite(np.asarray(log_prob)))
    advantages, returns = _gae(
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([0.2, 0.3], dtype=np.float32),
        np.asarray([False, True]),
        0.0,
        0.99,
        0.95,
    )
    assert np.all(np.isfinite(advantages))
    assert np.all(np.isfinite(returns))
