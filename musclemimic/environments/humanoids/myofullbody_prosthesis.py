"""MyoFullBody environment with control-level left prosthesis replacement."""

from __future__ import annotations

from collections.abc import Mapping
import jax.numpy as jnp
import mujoco
import numpy as np
from loco_mujoco.core import ObservationType
from mujoco import MjSpec

from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody, MyoFullBody
from musclemimic.prosthesis.constants import (
    DEFAULT_DISABLED_MUSCLE_NAMES,
    DEFAULT_PROSTHESIS_DISTILL_MASK_PRESET,
    DEFAULT_PROSTHESIS_TORQUE_LIMITS,
    PROSTHESIS_JOINT_NAMES,
    PROSTHESIS_MOTOR_NAMES,
    disabled_muscle_names_for_preset,
)
from musclemimic.prosthesis.mapping import (
    ProsthesisMapping,
    actuator_names_for_action_spec_from_spec,
    build_prosthesis_mapping,
    contact_load,
    prosthesis_body_descendants,
)
from musclemimic.prosthesis.observations import ProsthesisPrevTau  # noqa: F401 - registers observation


def _cfg_get(config: dict, key: str, default=None):
    if config is None:
        return default
    return config.get(key, default)


class MyoFullBodyProsthesisMixin:
    """Shared prosthesis action/diagnostic logic for CPU and MJX envs."""

    def __init__(
        self,
        *args,
        prosthesis: dict | None = None,
        prosthesis_control_mode: str | None = None,
        prosthesis_controller=None,
        **kwargs,
    ):
        prosthesis = dict(prosthesis or {})
        if prosthesis_control_mode is not None:
            prosthesis["control_mode"] = prosthesis_control_mode

        self.prosthesis_enabled = bool(_cfg_get(prosthesis, "enabled", True))
        self.prosthesis_side = str(_cfg_get(prosthesis, "side", "left"))
        self.prosthesis_control_mode = str(_cfg_get(prosthesis, "control_mode", "train_policy"))
        self.prosthesis_action_type = str(_cfg_get(prosthesis, "action_type", "torque"))
        if self.prosthesis_action_type not in {"torque", "pd_residual_torque"}:
            raise ValueError(f"Unknown prosthesis action_type={self.prosthesis_action_type!r}")
        self.prosthesis_joint_names = tuple(_cfg_get(prosthesis, "joints", PROSTHESIS_JOINT_NAMES))
        if self.prosthesis_joint_names != PROSTHESIS_JOINT_NAMES:
            raise ValueError(f"First version expects joints {PROSTHESIS_JOINT_NAMES}, got {self.prosthesis_joint_names}")

        torque_limits_cfg = _cfg_get(prosthesis, "torque_limits", DEFAULT_PROSTHESIS_TORQUE_LIMITS)
        if isinstance(torque_limits_cfg, Mapping) or hasattr(torque_limits_cfg, "keys"):
            torque_limits_cfg = (
                torque_limits_cfg.get("knee", DEFAULT_PROSTHESIS_TORQUE_LIMITS[0]),
                torque_limits_cfg.get("ankle", DEFAULT_PROSTHESIS_TORQUE_LIMITS[1]),
                torque_limits_cfg.get("subtalar", DEFAULT_PROSTHESIS_TORQUE_LIMITS[2]),
                torque_limits_cfg.get("mtp", DEFAULT_PROSTHESIS_TORQUE_LIMITS[3]),
            )
        self.prosthesis_torque_limits = np.asarray(torque_limits_cfg, dtype=np.float32)
        if self.prosthesis_torque_limits.size != len(PROSTHESIS_JOINT_NAMES):
            raise ValueError("prosthesis torque_limits must have four values")

        residual_pd_cfg = dict(_cfg_get(prosthesis, "residual_pd", {}) or {})
        self.prosthesis_residual_pd_kp = np.asarray(
            residual_pd_cfg.get("kp", (240.0, 180.0, 60.0, 40.0)),
            dtype=np.float32,
        )
        self.prosthesis_residual_pd_kd = np.asarray(
            residual_pd_cfg.get("kd", (24.0, 18.0, 6.0, 4.0)),
            dtype=np.float32,
        )
        if self.prosthesis_residual_pd_kp.size != len(PROSTHESIS_JOINT_NAMES):
            raise ValueError("prosthesis residual_pd.kp must have four values")
        if self.prosthesis_residual_pd_kd.size != len(PROSTHESIS_JOINT_NAMES):
            raise ValueError("prosthesis residual_pd.kd must have four values")

        exec_cfg = dict(_cfg_get(prosthesis, "execution", {}) or {})
        slew_cfg = exec_cfg.get("torque_slew_limit")
        if slew_cfg is None:
            self.prosthesis_torque_slew_limit = None
        elif isinstance(slew_cfg, (int, float)):
            self.prosthesis_torque_slew_limit = np.full(len(PROSTHESIS_JOINT_NAMES), float(slew_cfg), dtype=np.float32)
        else:
            self.prosthesis_torque_slew_limit = np.asarray(slew_cfg, dtype=np.float32)
            if self.prosthesis_torque_slew_limit.size != len(PROSTHESIS_JOINT_NAMES):
                raise ValueError("prosthesis execution.torque_slew_limit must have four values or be scalar")
        lowpass = exec_cfg.get("tau_lowpass_alpha")
        self.prosthesis_tau_lowpass_alpha = None if lowpass is None else float(lowpass)

        disable_cfg = dict(_cfg_get(prosthesis, "disable_muscles", {}) or {})
        self._disable_muscle_mode = str(disable_cfg.get("mode", "hard_zero_force"))
        self._disable_muscle_preset = str(disable_cfg.get("preset", DEFAULT_PROSTHESIS_DISTILL_MASK_PRESET))
        self._hard_zero_disabled_force = self._disable_muscle_mode in {
            "hard_zero_force",
            "zero_force",
            "name_matching_hard_zero_force",
            "auto_hard_zero_force",
        }
        self._disable_include = tuple(disable_cfg.get("include", ()))
        self._disable_exclude = tuple(disable_cfg.get("exclude", ()))
        self._disabled_muscle_names_for_spec = disabled_muscle_names_for_preset(
            self._disable_muscle_preset,
            include=self._disable_include,
            exclude=self._disable_exclude,
        )

        perturb_cfg = dict(_cfg_get(prosthesis, "tau_perturbation", {}) or {})
        self._tau_perturb_enabled = bool(perturb_cfg.get("enabled", False))
        self._tau_perturb_scale = float(perturb_cfg.get("scale", 1.0))
        self._tau_perturb_noise_std = float(perturb_cfg.get("noise_std", 0.0))

        penalty_cfg = dict(_cfg_get(prosthesis, "penalties", {}) or {})
        self._prosthesis_torque_penalty_coeff = float(penalty_cfg.get("torque", 0.0))
        self._prosthesis_tau_rate_penalty_coeff = float(penalty_cfg.get("torque_rate", 0.0))
        self._prosthesis_joint_limit_penalty_coeff = float(penalty_cfg.get("joint_limit", 0.0))
        self._prosthesis_foot_slip_penalty_coeff = float(penalty_cfg.get("foot_slip", 0.0))

        self.prosthesis_controller = prosthesis_controller
        self._last_prosthesis_tau = np.zeros(len(PROSTHESIS_JOINT_NAMES), dtype=np.float32)
        self._prosthesis_mapping: ProsthesisMapping | None = None
        self._prosthesis_body_ids: set[int] = set()

        if self.prosthesis_control_mode not in {"train_policy", "eval_policy", "eval_external_controller"}:
            raise ValueError(f"Unknown prosthesis_control_mode={self.prosthesis_control_mode!r}")
        if getattr(self, "mjx_enabled", False) and self.prosthesis_control_mode == "eval_external_controller":
            raise ValueError("eval_external_controller is CPU MuJoCo only; use MyoFullBodyProsthesisEnv")

        super().__init__(*args, **kwargs)

        self._prosthesis_mapping = build_prosthesis_mapping(
            self.model,
            disabled_mode=self._disable_muscle_mode,
            disabled_preset=self._disable_muscle_preset,
            disabled_include=self._disable_include,
            disabled_exclude=self._disable_exclude,
            torque_limits=tuple(float(x) for x in self.prosthesis_torque_limits),
        )
        self._prosthesis_body_ids = prosthesis_body_descendants(
            self.model, self._prosthesis_mapping.prosthesis_joint_ids
        )
        self.prosthesis_action_slice = slice(self.action_dim - len(PROSTHESIS_JOINT_NAMES), self.action_dim)
        self._prosthesis_action_positions = np.asarray(
            [
                int(np.where(np.asarray(self._action_indices) == aid)[0][0])
                for aid in self._prosthesis_mapping.prosthesis_actuator_ids
            ],
            dtype=np.int32,
        )
        if self.prosthesis_controller is not None:
            self.prosthesis_controller.reset(self)

    @property
    def prosthesis_mapping(self) -> ProsthesisMapping:
        assert self._prosthesis_mapping is not None
        return self._prosthesis_mapping

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        spec = super()._apply_spec_changes(spec)
        if not self.prosthesis_enabled:
            return spec
        if self._hard_zero_disabled_force:
            self._zero_disabled_actuators_in_spec(spec)
        existing = {actuator.name for actuator in spec.actuators}
        for name, joint_name, limit in zip(PROSTHESIS_MOTOR_NAMES, PROSTHESIS_JOINT_NAMES, self.prosthesis_torque_limits):
            if name in existing:
                continue
            actuator = spec.add_actuator()
            actuator.name = name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.target = joint_name
            actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            actuator.gainprm[0] = 1.0
            actuator.ctrllimited = True
            actuator.ctrlrange = [-float(limit), float(limit)]
        return spec

    def _zero_disabled_actuators_in_spec(self, spec: MjSpec) -> None:
        disabled = set(self._disabled_muscle_names_for_spec)
        for actuator in spec.actuators:
            if actuator.name not in disabled:
                continue
            # Keep the XML object and tendon transmission, but remove any active
            # or passive force generation from this actuator.
            actuator.ctrllimited = True
            actuator.ctrlrange = [-1e-9, 1e-9]
            actuator.forcelimited = True
            actuator.forcerange = [-1e-9, 1e-9]
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            actuator.gainprm[0] = 0.0
            for i in range(len(actuator.biasprm)):
                actuator.biasprm[i] = 0.0

    def _get_action_specification(self, spec: MjSpec) -> list[str]:
        if not self.prosthesis_enabled:
            return super()._get_action_specification(spec)
        return actuator_names_for_action_spec_from_spec(
            spec,
            disabled_names=self._disabled_muscle_names_for_spec,
        )

    def _get_observation_specification(self, spec: MjSpec):
        if not self.prosthesis_enabled:
            return super()._get_observation_specification(spec)

        obs_spec = []
        j_names = [j.name for j in spec.joints if j.name != self.root_free_joint_xml_name]
        if self._enable_joint_pos_observations:
            obs_spec.append(ObservationType.FreeJointPosNoXY("q_free_joint", self.root_free_joint_xml_name))
            obs_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))
        if self._enable_joint_vel_observations:
            obs_spec.append(ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name))
            obs_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        disabled = set(self._disabled_muscle_names_for_spec)
        prosthesis_motors = set(PROSTHESIS_MOTOR_NAMES)
        for actuator in spec.actuators:
            actuator_name = actuator.name
            if actuator_name in disabled or actuator_name in prosthesis_motors:
                continue
            if actuator.dyntype != mujoco.mjtDyn.mjDYN_MUSCLE:
                continue
            if self._enable_muscle_length_observations:
                obs_spec.append(ObservationType.ActuatorLength(f"muscle_length_{actuator_name.lower()}", xml_name=actuator_name))
            if self._enable_muscle_velocity_observations:
                obs_spec.append(ObservationType.ActuatorVelocity(f"muscle_velocity_{actuator_name.lower()}", xml_name=actuator_name))
            if self._enable_muscle_force_observations:
                obs_spec.append(ObservationType.ActuatorForce(f"muscle_force_{actuator_name.lower()}", xml_name=actuator_name))
            if self._enable_muscle_excitation_observations:
                obs_spec.append(ObservationType.ActuatorExcitation(f"muscle_excitation_{actuator_name.lower()}", xml_name=actuator_name))
            if self._enable_muscle_activation_observations:
                obs_spec.append(ObservationType.ActuatorActivation(f"muscle_activation_{actuator_name.lower()}", xml_name=actuator_name))

        if self._enable_touch_sensor_observations:
            for sensor_name in ("r_foot", "r_toes", "l_foot", "l_toes"):
                obs_spec.append(ObservationType.TouchSensor(f"touch_{sensor_name}", xml_name=sensor_name))

        obs_spec.append(ProsthesisPrevTau("prev_prosthesis_tau", group="prosthesis"))
        return obs_spec

    def _zero_disabled_muscles_np(self, data):
        ids = np.asarray(self.prosthesis_mapping.disabled_muscle_ids, dtype=np.int32)
        if ids.size:
            data.ctrl[ids] = 0.0
            actadr = np.asarray(self.model.actuator_actadr[ids], dtype=np.int32)
            for adr in actadr:
                if int(adr) >= 0:
                    data.act[int(adr)] = 0.0
        return data

    def _zero_disabled_muscles_jax(self, data):
        ids = jnp.asarray(self.prosthesis_mapping.disabled_muscle_ids, dtype=jnp.int32)
        if ids.size == 0:
            return data
        ctrl = data.ctrl.at[ids].set(0.0)
        actadr = jnp.asarray(np.asarray(self.model.actuator_actadr[list(self.prosthesis_mapping.disabled_muscle_ids)], dtype=np.int32))
        valid = actadr >= 0
        safe_actadr = jnp.where(valid, actadr, 0)
        act = data.act.at[safe_actadr].set(jnp.where(valid, 0.0, data.act[safe_actadr]))
        return data.replace(ctrl=ctrl, act=act)

    def _reference_prosthesis_state_np(self, data, carry):
        q_idx = np.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=np.int32)
        qd_idx = np.asarray(self.prosthesis_mapping.prosthesis_qvel_indices, dtype=np.int32)
        ref_q = np.asarray(data.qpos[q_idx], dtype=np.float64)
        ref_qd = np.asarray(data.qvel[qd_idx], dtype=np.float64)
        if getattr(self, "th", None) is not None and carry is not None:
            traj_state = carry.traj_state
            if hasattr(traj_state, "traj_no") and hasattr(traj_state, "subtraj_step_no"):
                traj_data = self.th.get_current_traj_data(carry, np)
                ref_q = np.asarray(traj_data.qpos[q_idx], dtype=np.float64)
                ref_qd = np.asarray(traj_data.qvel[qd_idx], dtype=np.float64)
        return ref_q, ref_qd

    def _reference_prosthesis_state_jax(self, data, carry):
        q_idx = jnp.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=jnp.int32)
        qd_idx = jnp.asarray(self.prosthesis_mapping.prosthesis_qvel_indices, dtype=jnp.int32)
        ref_q = data.qpos[q_idx]
        ref_qd = data.qvel[qd_idx]
        if getattr(self, "th", None) is not None and carry is not None:
            traj_data = self.th.get_current_traj_data(carry, jnp)
            ref_q = traj_data.qpos[q_idx]
            ref_qd = traj_data.qvel[qd_idx]
        return ref_q, ref_qd

    def _pd_residual_tau_np(self, residual_tau, data, carry):
        q_idx = np.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=np.int32)
        qd_idx = np.asarray(self.prosthesis_mapping.prosthesis_qvel_indices, dtype=np.int32)
        q = np.asarray(data.qpos[q_idx], dtype=np.float64)
        qd = np.asarray(data.qvel[qd_idx], dtype=np.float64)
        ref_q, ref_qd = self._reference_prosthesis_state_np(data, carry)
        pd_tau = self.prosthesis_residual_pd_kp * (ref_q - q) + self.prosthesis_residual_pd_kd * (ref_qd - qd)
        return self._finalize_prosthesis_tau_np(np.clip(pd_tau + residual_tau, -self.prosthesis_torque_limits, self.prosthesis_torque_limits))

    def _finalize_prosthesis_tau_np(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64)
        prev = np.asarray(self._last_prosthesis_tau, dtype=np.float64)
        if self.prosthesis_torque_slew_limit is not None:
            delta = np.clip(tau - prev, -self.prosthesis_torque_slew_limit, self.prosthesis_torque_slew_limit)
            tau = prev + delta
        if self.prosthesis_tau_lowpass_alpha is not None:
            alpha = float(self.prosthesis_tau_lowpass_alpha)
            tau = alpha * tau + (1.0 - alpha) * prev
        limits = np.asarray(self.prosthesis_torque_limits, dtype=np.float64)
        return np.clip(tau, -limits, limits)

    def _pd_residual_tau_jax(self, residual_tau, data, carry):
        q_idx = jnp.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=jnp.int32)
        qd_idx = jnp.asarray(self.prosthesis_mapping.prosthesis_qvel_indices, dtype=jnp.int32)
        q = data.qpos[q_idx]
        qd = data.qvel[qd_idx]
        ref_q, ref_qd = self._reference_prosthesis_state_jax(data, carry)
        pd_tau = jnp.asarray(self.prosthesis_residual_pd_kp) * (ref_q - q) + jnp.asarray(self.prosthesis_residual_pd_kd) * (ref_qd - qd)
        limits = jnp.asarray(self.prosthesis_torque_limits)
        return jnp.clip(pd_tau + residual_tau, -limits, limits)

    def _reset_carry(self, model, data, carry):
        data, carry = super()._reset_carry(model, data, carry)
        data = self._zero_disabled_muscles_np(data)
        self._last_prosthesis_tau[:] = 0.0
        if self.prosthesis_controller is not None:
            self.prosthesis_controller.reset(self)
        return data, carry

    def _mjx_reset_carry(self, model, data, carry):
        data, carry = super()._mjx_reset_carry(model, data, carry)
        data = self._zero_disabled_muscles_jax(data)
        return data, carry

    def _simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._simulation_pre_step(model, data, carry)
        data = self._zero_disabled_muscles_np(data)
        return model, data, carry

    def _mjx_simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        data = self._zero_disabled_muscles_jax(data)
        return model, data, carry

    def _compute_action(self, action, model, data, carry):
        ctrl_action, carry = super()._compute_action(action, model, data, carry)
        ctrl_action = np.asarray(ctrl_action, dtype=np.float64).copy()
        tau = ctrl_action[self._prosthesis_action_positions].astype(np.float64)
        if self.prosthesis_control_mode == "eval_external_controller":
            if self.prosthesis_controller is None:
                raise ValueError("eval_external_controller requires prosthesis_controller")
            tau = np.asarray(self.prosthesis_controller.update(self.get_prosthesis_obs()), dtype=np.float64).reshape(-1)
            if tau.size != len(PROSTHESIS_JOINT_NAMES):
                raise ValueError(f"External controller must return 4 torques, got {tau.size}")
            tau = np.clip(tau, -self.prosthesis_torque_limits, self.prosthesis_torque_limits)
            ctrl_action[self._prosthesis_action_positions] = tau
        elif self.prosthesis_action_type == "pd_residual_torque":
            tau = self._pd_residual_tau_np(tau, data, carry)
            ctrl_action[self._prosthesis_action_positions] = tau
        elif self.prosthesis_action_type == "torque":
            tau = self._finalize_prosthesis_tau_np(
                np.clip(tau, -self.prosthesis_torque_limits, self.prosthesis_torque_limits)
            )
            ctrl_action[self._prosthesis_action_positions] = tau
        elif self._tau_perturb_enabled:
            tau = self._tau_perturb_scale * tau
            if self._tau_perturb_noise_std > 0.0:
                tau = tau + np.random.normal(0.0, self._tau_perturb_noise_std, size=tau.shape)
            tau = np.clip(tau, -self.prosthesis_torque_limits, self.prosthesis_torque_limits)
            ctrl_action[self._prosthesis_action_positions] = tau
        self._last_prosthesis_tau = tau.astype(np.float32)
        return ctrl_action, carry

    def _mjx_compute_action(self, action, model, data, carry):
        ctrl_action, carry = super()._mjx_compute_action(action, model, data, carry)
        pos = jnp.asarray(self._prosthesis_action_positions, dtype=jnp.int32)
        if self.prosthesis_action_type == "pd_residual_torque":
            tau = self._pd_residual_tau_jax(ctrl_action[pos], data, carry)
            ctrl_action = ctrl_action.at[pos].set(tau)
        elif self._tau_perturb_enabled:
            limits = jnp.asarray(self.prosthesis_torque_limits)
            tau = jnp.clip(self._tau_perturb_scale * ctrl_action[pos], -limits, limits)
            ctrl_action = ctrl_action.at[pos].set(tau)
        return ctrl_action, carry

    def _update_info_dictionary(self, info, obs, data, carry):
        info = super()._update_info_dictionary(info, obs, data, carry)
        ids = np.asarray(self.prosthesis_mapping.disabled_muscle_ids, dtype=np.int32)
        info.update(
            {
                "prosthesis_tau": self._last_prosthesis_tau.copy(),
                "prev_prosthesis_tau": self._last_prosthesis_tau.copy(),
                "disabled_muscle_ctrl_norm": float(np.linalg.norm(data.ctrl[ids])) if ids.size else 0.0,
                "disabled_muscle_force_norm": float(np.linalg.norm(data.actuator_force[ids])) if ids.size else 0.0,
                "prosthesis_q": np.asarray(
                    data.qpos[np.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=np.int32)],
                    dtype=np.float32,
                ),
                "prosthesis_qd": np.asarray(
                    data.qvel[np.asarray(self.prosthesis_mapping.prosthesis_qvel_indices, dtype=np.int32)],
                    dtype=np.float32,
                ),
            }
        )
        return info

    def _mjx_reset_info_dictionary(self, obs, data, key):
        info = super()._mjx_reset_info_dictionary(obs, data, key)
        info.update(
            {
                "prosthesis_tau": jnp.zeros((len(PROSTHESIS_JOINT_NAMES),), dtype=jnp.float32),
                "prev_prosthesis_tau": jnp.zeros((len(PROSTHESIS_JOINT_NAMES),), dtype=jnp.float32),
                "disabled_muscle_ctrl_norm": jnp.asarray(0.0, dtype=jnp.float32),
                "disabled_muscle_force_norm": jnp.asarray(0.0, dtype=jnp.float32),
            }
        )
        return info

    def _mjx_update_info_dictionary(self, info, obs, data, carry):
        info = super()._mjx_update_info_dictionary(info, obs, data, carry)
        raw = jnp.clip(carry.last_action[self.prosthesis_action_slice], -1.0, 1.0)
        tau = raw * jnp.asarray(self.prosthesis_torque_limits)
        ids = jnp.asarray(self.prosthesis_mapping.disabled_muscle_ids, dtype=jnp.int32)
        info.update(
            {
                "prosthesis_tau": tau,
                "prev_prosthesis_tau": tau,
                "disabled_muscle_ctrl_norm": jnp.linalg.norm(data.ctrl[ids]) if ids.size else jnp.asarray(0.0),
                "disabled_muscle_force_norm": jnp.linalg.norm(data.actuator_force[ids]) if ids.size else jnp.asarray(0.0),
            }
        )
        return info

    def _prosthesis_penalty_np(self, action, data) -> tuple[float, dict]:
        tau = np.asarray(self._last_prosthesis_tau, dtype=np.float64)
        prev_tau = np.clip(np.asarray(action[self.prosthesis_action_slice], dtype=np.float64), -1.0, 1.0) * self.prosthesis_torque_limits
        torque_pen = -float(np.mean((tau / self.prosthesis_torque_limits) ** 2))
        rate_pen = -float(np.mean(((tau - prev_tau) / self.prosthesis_torque_limits) ** 2))
        qpos_idx = np.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=np.int32)
        q = np.asarray(data.qpos[qpos_idx], dtype=np.float64)
        low_high = np.asarray([self.model.jnt_range[jid] for jid in self.prosthesis_mapping.prosthesis_joint_ids], dtype=np.float64)
        violation = np.maximum(low_high[:, 0] - q, 0.0) + np.maximum(q - low_high[:, 1], 0.0)
        limit_pen = -float(np.sum(violation**2))
        total = (
            self._prosthesis_torque_penalty_coeff * torque_pen
            + self._prosthesis_tau_rate_penalty_coeff * rate_pen
            + self._prosthesis_joint_limit_penalty_coeff * limit_pen
        )
        return total, {
            "penalty_prosthesis_tau": self._prosthesis_torque_penalty_coeff * torque_pen,
            "penalty_prosthesis_tau_rate": self._prosthesis_tau_rate_penalty_coeff * rate_pen,
            "penalty_prosthesis_joint_limit": self._prosthesis_joint_limit_penalty_coeff * limit_pen,
        }

    def _prosthesis_penalty_jax(self, action, data):
        raw = jnp.clip(action[self.prosthesis_action_slice], -1.0, 1.0)
        limits = jnp.asarray(self.prosthesis_torque_limits)
        tau = raw * limits
        torque_pen = -jnp.mean((tau / limits) ** 2)
        rate_pen = -jnp.asarray(0.0)
        q = data.qpos[jnp.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=jnp.int32)]
        joint_ranges = jnp.asarray(np.asarray([self.model.jnt_range[jid] for jid in self.prosthesis_mapping.prosthesis_joint_ids]))
        violation = jnp.maximum(joint_ranges[:, 0] - q, 0.0) + jnp.maximum(q - joint_ranges[:, 1], 0.0)
        limit_pen = -jnp.sum(violation**2)
        total = (
            self._prosthesis_torque_penalty_coeff * torque_pen
            + self._prosthesis_tau_rate_penalty_coeff * rate_pen
            + self._prosthesis_joint_limit_penalty_coeff * limit_pen
        )
        return total, {
            "penalty_prosthesis_tau": self._prosthesis_torque_penalty_coeff * torque_pen,
            "penalty_prosthesis_tau_rate": self._prosthesis_tau_rate_penalty_coeff * rate_pen,
            "penalty_prosthesis_joint_limit": self._prosthesis_joint_limit_penalty_coeff * limit_pen,
        }

    def _reward(self, obs, action, next_obs, absorbing, info, model, data, carry):
        reward, carry = super()._reward(obs, action, next_obs, absorbing, info, model, data, carry)
        penalty, penalty_info = self._prosthesis_penalty_np(action, data)
        reward = max(float(reward) + float(penalty), 0.0)
        info.update(penalty_info)
        info["reward_total"] = reward
        return reward, carry

    def _mjx_reward(self, obs, action, next_obs, absorbing, info, model, data, carry):
        reward, carry, reward_info = super()._mjx_reward(obs, action, next_obs, absorbing, info, model, data, carry)
        penalty, penalty_info = self._prosthesis_penalty_jax(action, data)
        reward = jnp.maximum(reward + penalty, 0.0)
        reward_info.update(penalty_info)
        reward_info["reward_total"] = reward
        return reward, carry, reward_info

    def get_prosthesis_obs(self) -> dict:
        q_idx = np.asarray(self.prosthesis_mapping.prosthesis_qpos_indices, dtype=np.int32)
        qd_idx = np.asarray(self.prosthesis_mapping.prosthesis_qvel_indices, dtype=np.int32)
        ref_q = np.asarray(self.data.qpos[q_idx], dtype=np.float32)
        ref_qd = np.asarray(self.data.qvel[qd_idx], dtype=np.float32)
        phase = 0.0
        if getattr(self, "th", None) is not None and self._additional_carry is not None:
            traj_state = self._additional_carry.traj_state
            if hasattr(traj_state, "traj_no") and hasattr(traj_state, "subtraj_step_no"):
                traj_data = self.th.get_current_traj_data(self._additional_carry, np)
                ref_q = np.asarray(traj_data.qpos[q_idx], dtype=np.float32)
                ref_qd = np.asarray(traj_data.qvel[qd_idx], dtype=np.float32)
                traj_len = max(int(self.th.len_trajectory(traj_state.traj_no)), 1)
                phase = float(traj_state.subtraj_step_no) / float(traj_len)
        foot_contact, grf = contact_load(self.model, self.data, self._prosthesis_body_ids)
        return {
            "q": np.asarray(self.data.qpos[q_idx], dtype=np.float32),
            "qd": np.asarray(self.data.qvel[qd_idx], dtype=np.float32),
            "pelvis_pos": np.asarray(self.data.qpos[:3], dtype=np.float32),
            "pelvis_vel": np.asarray(self.data.qvel[:3], dtype=np.float32),
            "pelvis_quat": np.asarray(self.data.qpos[3:7], dtype=np.float32),
            "pelvis_angvel": np.asarray(self.data.qvel[3:6], dtype=np.float32),
            "thigh_angle": np.asarray([self.data.qpos[q_idx[0]]], dtype=np.float32),
            "thigh_velocity": np.asarray([self.data.qvel[qd_idx[0]]], dtype=np.float32),
            "foot_contact": foot_contact,
            "grf": grf,
            "ref_q": ref_q,
            "ref_qd": ref_qd,
            "phase": phase,
            "prev_tau": self._last_prosthesis_tau.copy(),
        }


class MyoFullBodyProsthesisEnv(MyoFullBodyProsthesisMixin, MyoFullBody):
    """CPU MuJoCo MyoFullBody prosthesis environment."""


class MjxMyoFullBodyProsthesisEnv(MyoFullBodyProsthesisMixin, MjxMyoFullBody):
    """MJX MyoFullBody prosthesis environment for PPO training."""

    mjx_enabled = True
