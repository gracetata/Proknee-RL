"""Custom observations for prosthesis control."""

from __future__ import annotations

import numpy as np

from loco_mujoco.core.observations import StatefulObservation


class ProsthesisPrevTau(StatefulObservation):
    """Previous executed prosthesis torque, exposed as a 4D observation."""

    dim = 4

    def _init_from_mj(self, env, model, data, current_obs_size):
        del model, data
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.data_type_ind = np.arange(self.dim, dtype=int)
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim, dtype=int)
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend):
        del env, key, model, data
        return backend.zeros(self.dim)

    def reset_state(self, env, model, data, carry, backend):
        del env, model, backend
        return data, carry

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        del model, data
        if backend == np and hasattr(env, "_last_prosthesis_tau"):
            tau = backend.asarray(env._last_prosthesis_tau)
        else:
            action = carry.last_action
            start = env.prosthesis_action_slice.start
            end = env.prosthesis_action_slice.stop
            raw = backend.clip(action[start:end], -1.0, 1.0)
            tau = raw * backend.asarray(env.prosthesis_torque_limits)
        return tau, carry


ProsthesisPrevTau.register()
