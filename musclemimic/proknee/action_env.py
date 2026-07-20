"""Closed-loop action environment for human-prosthesis coupling experiments.

This module is intentionally separate from ``hybrid_env.py`` so the current
qpos+torque-residual Stage1 workflow remains the default rollback path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .hybrid_env import HybridStep, MuscleProKneeHybridEnv


@dataclass
class CoupledActionStep(HybridStep):
    """One closed-loop action step plus scalar reward terms."""

    reward_terms: dict


def _root_up_z(qpos: np.ndarray) -> float:
    quat = np.asarray(qpos[3:7], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-8:
        quat = quat / norm
    _qw, qx, qy, _qz = quat
    return float(1.0 - 2.0 * (qx * qx + qy * qy))


class MuscleProKneeCoupledActionEnv(MuscleProKneeHybridEnv):
    """Execute prosthesis motor actions in the real replacement environment.

    The frozen full-body MuscleMimic oracle still drives the non-prosthesis body.
    The prosthesis controller directly injects bounded torques on the four
    controlled prosthesis DOFs.  Optionally, a small residual can be added to
    non-boundary muscle controls for later co-adaptation experiments.
    """

    def __init__(
        self,
        *args,
        action_mode: str = "torque",
        action_torque_limit: float | tuple[float, ...] = (110.0, 65.0, 45.0, 22.0),
        action_torque_slew_limit: float | tuple[float, ...] | None = (10.0, 5.0, 3.0, 1.5),
        body_residual_scale: float = 0.0,
        **kwargs,
    ):
        if action_mode not in {"torque", "impedance_residual"}:
            raise ValueError(f"Unknown action_mode={action_mode!r}")
        super().__init__(*args, target_mode="qpos", pd_override=True, **kwargs)
        self.action_mode = action_mode
        self.action_torque_limit = self._vector(action_torque_limit)
        self.action_torque_slew_limit = None if action_torque_slew_limit is None else self._vector(
            action_torque_slew_limit
        )
        self.body_residual_scale = float(body_residual_scale)
        self._last_action_torque = np.zeros(self.action_dim, dtype=np.float64)
        self._non_boundary_action_indices = self._resolve_non_boundary_action_indices()

    def _vector(self, value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(self.action_dim, float(arr), dtype=np.float64)
        arr = arr.reshape(-1)
        if arr.size != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} values, got {arr.size}")
        return arr

    @property
    def body_residual_dim(self) -> int:
        return int(self._non_boundary_action_indices.size)

    def reset(self) -> CoupledActionStep:
        self._last_action_torque[:] = 0.0
        step = super().reset()
        return CoupledActionStep(**step.__dict__, reward_terms={})

    def _resolve_non_boundary_action_indices(self) -> np.ndarray:
        all_indices = np.asarray(self.env._action_indices, dtype=np.int32)
        boundary = set(int(x) for x in self._prosthesis_boundary_actuator_indices)
        keep = [int(aid) for aid in all_indices if int(aid) not in boundary]
        return np.asarray(keep, dtype=np.int32)

    def _action_to_torque(self, prosthesis_action: np.ndarray) -> np.ndarray:
        action = np.asarray(prosthesis_action, dtype=np.float64).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected prosthesis action dim {self.action_dim}, got {action.size}")
        # Policies emit normalized actions; the environment owns physical limits.
        desired = np.clip(action, -1.0, 1.0) * self.action_torque_limit
        if self.action_torque_slew_limit is not None:
            delta = np.clip(
                desired - self._last_action_torque,
                -self.action_torque_slew_limit,
                self.action_torque_slew_limit,
            )
            desired = self._last_action_torque + delta
        self._last_action_torque = desired.copy()
        return desired

    def _left_foot_z(self) -> float:
        for site_name, site_id in self.audit.sites:
            if "toe" in site_name.lower() or "toes" in site_name.lower():
                return float(self.env.data.site_xpos[int(site_id), 2])
        return float("nan")

    def _step_with_action_torque(
        self,
        oracle_action: np.ndarray,
        prosthesis_torque: np.ndarray,
        body_residual: np.ndarray | None = None,
    ):
        cur_info = self.env._info.copy()
        carry = self.env._additional_carry.replace(last_action=oracle_action)
        processed_action, carry = self.env._preprocess_action(oracle_action, self.env._model, self.env._data, carry)
        self.env._model, self.env._data, carry = self.env._simulation_pre_step(
            self.env._model, self.env._data, carry
        )

        torque_sum = np.zeros(self.action_dim, dtype=np.float64)
        for _ in range(self.env._n_intermediate_steps):
            self.env._data.qfrc_applied[:] = 0.0
            ctrl_action, carry = self.env._compute_action(processed_action, self.env._model, self.env._data, carry)
            ctrl = np.asarray(ctrl_action, dtype=np.float64).reshape(-1).copy()
            self.env._data.ctrl[self.env._action_indices] = ctrl
            if body_residual is not None and self.body_residual_scale != 0.0:
                residual = np.asarray(body_residual, dtype=np.float64).reshape(-1)
                if residual.size != self.body_residual_dim:
                    raise ValueError(f"Expected body residual dim {self.body_residual_dim}, got {residual.size}")
                self.env._data.ctrl[self._non_boundary_action_indices] += self.body_residual_scale * np.tanh(residual)
            if self._prosthesis_boundary_actuator_indices.size:
                self.env._data.ctrl[self._prosthesis_boundary_actuator_indices] *= float(
                    np.clip(self._prosthesis_muscle_scale, 0.0, 1.0)
                )
            self.env._data.qfrc_applied[self.audit.qvel_indices] = prosthesis_torque
            torque_sum += prosthesis_torque
            mujoco.mj_step(self.env._model, self.env._data, self.env._n_substeps)
            self.env._data.qfrc_applied[:] = 0.0

        self.env._data, carry = self.env._simulation_post_step(self.env._model, self.env._data, carry)
        cur_obs, carry = self.env._create_observation(self.env._model, self.env._data, carry)
        cur_obs, self.env._data, cur_info, carry = self.env._step_finalize(
            cur_obs, self.env._model, self.env._data, cur_info, carry
        )
        cur_info = self.env._update_info_dictionary(cur_info, cur_obs, self.env._data, carry)
        absorbing, carry = self.env._is_absorbing(cur_obs, cur_info, self.env._data, carry)
        reward, carry = self.env._reward(
            self.env._obs, oracle_action, cur_obs, absorbing, cur_info, self.env._model, self.env._data, carry
        )
        done = self.env._is_done(cur_obs, absorbing, cur_info, self.env._data, carry)
        carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
        self.env._obs = cur_obs
        self.env._additional_carry = carry
        return np.asarray(cur_obs), reward, absorbing, done, cur_info, torque_sum / max(self.env._n_intermediate_steps, 1)

    def step_action(
        self,
        prosthesis_action: np.ndarray,
        body_residual: np.ndarray | None = None,
    ) -> CoupledActionStep:
        if self._oracle_obs is None:
            self.reset()

        qpos_before = np.asarray(self.env.data.qpos, dtype=np.float32).copy()
        qvel_before = np.asarray(self.env.data.qvel, dtype=np.float32).copy()
        obs_before = self._last_obs.copy() if self._last_obs is not None else np.zeros(self.spec.obs_dim, dtype=np.float32)
        hist_before = self.history.as_array()
        placeholder_target = qpos_before[self.audit.qpos_indices].astype(np.float32)
        priv_before = self._build_priv_info(placeholder_target)

        oracle_action = self.oracle.act(self._oracle_obs)
        qpos_oracle_after, qvel_oracle_after, torque_oracle_after = self._oracle_step_prosthesis_labels(oracle_action)
        prosthesis_torque = self._action_to_torque(prosthesis_action)
        raw_obs, _env_reward, _absorbing, done, info, applied_mean_torque = self._step_with_action_torque(
            oracle_action,
            prosthesis_torque,
            body_residual=body_residual,
        )

        qpos_after = np.asarray(self.env.data.qpos, dtype=np.float32).copy()
        qvel_after = np.asarray(self.env.data.qvel, dtype=np.float32).copy()
        oracle_target = qpos_oracle_after[self.audit.qpos_indices].astype(np.float32)
        self._oracle_obs = self.oracle.update_obs(raw_obs)
        obs = self._build_obs()
        self.history.append(obs)
        self._previous_speed = float(np.linalg.norm(np.asarray(self.env.data.qvel[:2], dtype=np.float64)))
        self._last_obs = obs

        reward, terms = self._reward_terms(
            qpos_after,
            qvel_after,
            qpos_oracle_after,
            qvel_oracle_after,
            torque_oracle_after,
            applied_mean_torque,
            prosthesis_torque,
            body_residual,
            done,
        )
        info = dict(info)
        info.update(
            {
                "qpos_before": qpos_before,
                "qvel_before": qvel_before,
                "qpos_after": qpos_after,
                "qvel_after": qvel_after,
                "prosthesis_target_qpos": oracle_target,
                "prosthesis_executed_qpos": qpos_after[self.audit.qpos_indices].astype(np.float32),
                "prosthesis_qvel": qvel_after[self.audit.qvel_indices].astype(np.float32),
                "prosthesis_oracle_torque_target": torque_oracle_after.astype(np.float32),
                "prosthesis_applied_torque_mean": applied_mean_torque.astype(np.float32),
                "prosthesis_action_torque": prosthesis_torque.astype(np.float32),
                "prosthesis_tracking_mae": float(
                    np.mean(np.abs(qpos_after[self.audit.qpos_indices] - oracle_target))
                ),
                "prosthesis_root_height": float(qpos_after[2]),
                "prosthesis_root_up_z": _root_up_z(qpos_after),
                "prosthesis_ncon": int(self.env.data.ncon),
                "prosthesis_toe_z": self._left_foot_z(),
            }
        )
        return CoupledActionStep(
            obs_before,
            priv_before,
            hist_before,
            oracle_target,
            torque_oracle_after.astype(np.float32),
            float(reward),
            bool(done),
            info,
            terms,
        )

    def _build_obs(self) -> np.ndarray:
        return super()._build_obs()

    def _build_priv_info(self, target: np.ndarray) -> np.ndarray:
        return super()._build_priv_info(target)

    def _reward_terms(
        self,
        qpos_after: np.ndarray,
        qvel_after: np.ndarray,
        qpos_oracle_after: np.ndarray,
        qvel_oracle_after: np.ndarray,
        torque_oracle: np.ndarray,
        applied_torque: np.ndarray,
        action_torque: np.ndarray,
        body_residual: np.ndarray | None,
        done: bool,
    ) -> tuple[float, dict]:
        prosthesis_q = qpos_after[self.audit.qpos_indices]
        oracle_q = qpos_oracle_after[self.audit.qpos_indices]
        prosthesis_qd = qvel_after[self.audit.qvel_indices]
        oracle_qd = qvel_oracle_after[self.audit.qvel_indices]
        q_err = float(np.mean(np.square(prosthesis_q - oracle_q)))
        qd_err = float(np.mean(np.square(prosthesis_qd - oracle_qd)))
        torque_err = float(np.mean(np.square((applied_torque - torque_oracle) / np.maximum(self.action_torque_limit, 1.0))))
        root_height_short = max(0.85 - float(qpos_after[2]), 0.0)
        root_tilt = max(1.0 - _root_up_z(qpos_after), 0.0)
        action_energy = float(np.mean(np.square(action_torque / np.maximum(self.action_torque_limit, 1.0))))
        residual_energy = 0.0 if body_residual is None else float(np.mean(np.square(np.tanh(body_residual))))
        contact_short = max(1.0 - float(self.env.data.ncon), 0.0)
        done_penalty = 1.0 if done else 0.0
        terms = {
            "q_err": q_err,
            "qvel_err": qd_err,
            "torque_err": torque_err,
            "root_height_short": root_height_short,
            "root_tilt": root_tilt,
            "action_energy": action_energy,
            "body_residual_energy": residual_energy,
            "contact_short": contact_short,
            "done_penalty": done_penalty,
        }
        reward = (
            1.0
            - 15.0 * q_err
            - 0.2 * qd_err
            - 2.0 * torque_err
            - 8.0 * root_height_short
            - 4.0 * root_tilt
            - 0.05 * action_energy
            - 0.1 * residual_energy
            - 0.5 * contact_short
            - 5.0 * done_penalty
        )
        return float(reward), terms

