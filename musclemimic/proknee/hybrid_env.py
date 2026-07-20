"""Hybrid environment scaffold for MuscleMimic-ProKnee Stage 1/2."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint

from musclemimic.prosthesis.constants import disabled_muscle_names_for_preset

from .constants import LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES, LeftLegAudit, audit_myofullbody_left_leg
from .observation import ProprioHistory, build_global_observation, build_observation_spec, build_priv_info
from .oracle import FrozenMMOracle


@dataclass
class HybridStep:
    obs: np.ndarray
    priv_info: np.ndarray
    proprio_hist: np.ndarray
    oracle_target: np.ndarray
    oracle_torque_target: np.ndarray
    reward: float
    done: bool
    info: dict


def _optional_vector(value, n: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size != n:
        raise ValueError(f"Expected scalar or {n} values, got {arr.size}")
    return arr


def make_musclemimic_env(
    checkpoint_path: str,
    dataset_group: str = "KIT_KINESIS_TRAINING_MOTIONS",
    rel_dataset_path: list[str] | None = None,
    use_mujoco: bool = True,
):
    """Create a MyoFullBody env using the checkpoint's training config."""

    config, _agent_state, _metadata = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)
    apply_temporal_params(config)
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)

    if use_mujoco and "Mjx" in env_params.get("env_name", ""):
        env_params["env_name"] = env_params["env_name"].replace("Mjx", "")
    env_params["headless"] = True

    amass_conf = config.experiment.task_factory.params.amass_dataset_conf
    amass_conf.dataset_group = dataset_group
    if rel_dataset_path is not None:
        amass_conf.dataset_group = None
        amass_conf.rel_dataset_path = rel_dataset_path

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


class MuscleProKneeHybridEnv:
    """Stage-1/2 data interface around a frozen MuscleMimic oracle.

    **Prosthesis replacement (DAgger + PD):** when ``apply_teacher_action``,
    ``pd_override``, and ``target_mode=="qpos"``, each env step runs:
    1) backup sim → oracle ``env.step`` → read next-step expert prosthesis
       ``qpos``/``qvel`` for labels → restore sim;
    2) execute Stage1 target via :meth:`_step_prosthesis_replacement_pd` (PD
       torques on the 4 prosthesis DOFs, oracle muscles elsewhere, boundary
       muscles scaled by :attr:`prosthesis_muscle_scale`).
    """

    def __init__(
        self,
        checkpoint_path: str,
        dataset_group: str = "KIT_KINESIS_TRAINING_MOTIONS",
        rel_dataset_path: list[str] | None = None,
        history_len: int = 30,
        deterministic_oracle: bool = False,
        apply_teacher_action: bool = False,
        target_mode: str = "residual",
        pd_override: bool = False,
        pd_kp: tuple[float, ...] = (300.0, 200.0, 200.0, 120.0),
        pd_kd: tuple[float, ...] = (30.0, 20.0, 20.0, 12.0),
        pd_torque_limit: float = 120.0,
        pd_torque_slew_limit: float | tuple[float, ...] | None = None,
        oracle_torque_ff_scale: float = 0.0,
        oracle_torque_ff_limit: float | tuple[float, ...] | None = None,
        torque_feedforward_target_mode: str = "oracle",
        pd_gain_scale: float = 1.0,
        max_prosthesis_qpos_step: float | None = None,
        prosthesis_muscle_scale: float = 1.0,
        conflict_muscle_scale: float | None = None,
        mask_preset: str = "strict19",
        obs_mode: str = "easy",
    ):
        if target_mode not in {"residual", "qpos"}:
            raise ValueError(f"Unknown target_mode={target_mode!r}; expected 'residual' or 'qpos'")
        if obs_mode not in {"easy", "pose_only"}:
            raise ValueError(f"Unknown obs_mode={obs_mode!r}; expected 'easy' or 'pose_only'")
        if torque_feedforward_target_mode not in {"oracle", "pd_residual"}:
            raise ValueError(
                "Unknown torque_feedforward_target_mode="
                f"{torque_feedforward_target_mode!r}; expected 'oracle' or 'pd_residual'"
            )
        self.env = make_musclemimic_env(
            checkpoint_path,
            dataset_group=dataset_group,
            rel_dataset_path=rel_dataset_path,
            use_mujoco=True,
        )
        self.audit: LeftLegAudit = audit_myofullbody_left_leg(self.env.model)
        self.oracle = FrozenMMOracle(checkpoint_path, self.env, deterministic=deterministic_oracle)
        self.apply_teacher_action = bool(apply_teacher_action)
        self.target_mode = target_mode
        self.pd_override = bool(pd_override)
        self.pd_kp = np.asarray(pd_kp, dtype=np.float64)
        self.pd_kd = np.asarray(pd_kd, dtype=np.float64)
        self.pd_torque_limit = _optional_vector(pd_torque_limit, len(self.audit.joints))
        self.pd_torque_slew_limit = _optional_vector(pd_torque_slew_limit, len(self.audit.joints))
        self.oracle_torque_ff_scale = float(oracle_torque_ff_scale)
        self.oracle_torque_ff_limit = _optional_vector(oracle_torque_ff_limit, len(self.audit.joints))
        self.torque_feedforward_target_mode = torque_feedforward_target_mode
        self.pd_gain_scale = float(np.clip(pd_gain_scale, 1e-6, 10.0))
        self.max_prosthesis_qpos_step = _optional_vector(max_prosthesis_qpos_step, len(self.audit.joints))
        self._last_commanded_prosthesis_qpos: np.ndarray | None = None
        self._last_prosthesis_pd_torque = np.zeros(len(self.audit.joints), dtype=np.float64)
        # Curriculum: scale actuators that cross the prosthesis boundary (see constants).
        # ``conflict_muscle_scale`` is deprecated; when set it overrides ``prosthesis_muscle_scale``.
        if conflict_muscle_scale is not None:
            self._prosthesis_muscle_scale = float(conflict_muscle_scale)
        else:
            self._prosthesis_muscle_scale = float(prosthesis_muscle_scale)
        self.mask_preset = str(mask_preset).strip().lower()
        self.obs_mode = str(obs_mode).strip().lower()
        self._disabled_muscle_names = self._resolve_disabled_muscle_names()
        self._prosthesis_boundary_actuator_indices = self._resolve_prosthesis_boundary_actuators()

        self.spec = build_observation_spec(
            self.env, self.audit, history_len=history_len, obs_mode=self.obs_mode
        )
        self.history = ProprioHistory(history_len=history_len, dim=self.spec.proprio_dim)
        self._previous_speed: float | None = None
        self._oracle_obs: np.ndarray | None = None
        self._last_obs: np.ndarray | None = None

    @property
    def prosthesis_muscle_scale(self) -> float:
        """Scale factor in [0, 1] on boundary muscles during prosthesis-replacement PD steps."""

        return self._prosthesis_muscle_scale

    @property
    def conflict_muscle_scale(self) -> float:
        """Deprecated alias for :attr:`prosthesis_muscle_scale`."""

        return self._prosthesis_muscle_scale

    @property
    def conflict_actuator_indices(self) -> np.ndarray:
        """Indices of actuators scaled by :attr:`prosthesis_muscle_scale` (backward-compatible name)."""

        return self._prosthesis_boundary_actuator_indices

    @property
    def action_dim(self) -> int:
        return len(self.audit.joints)

    def reset(self) -> HybridStep:
        raw_obs = self.env.reset()
        self._oracle_obs = self.oracle.reset(raw_obs)
        self.history.reset()
        self._previous_speed = None
        if self.target_mode == "qpos":
            target = np.asarray(self.env.data.qpos[self.audit.qpos_indices], dtype=np.float32)
            self._last_commanded_prosthesis_qpos = target.copy()
        else:
            target = np.zeros(self.action_dim, dtype=np.float32)
            self._last_commanded_prosthesis_qpos = None
        self._last_prosthesis_pd_torque[:] = 0.0
        obs = self._build_obs()
        hist = self.history.append(obs)
        priv = self._build_priv_info(target)
        self._last_obs = obs
        torque_target = np.zeros(self.action_dim, dtype=np.float32)
        return HybridStep(obs, priv, hist, target, torque_target, 0.0, False, {})

    def _apply_joint_residual(self, residual: np.ndarray, qpos_before: np.ndarray) -> None:
        residual = np.asarray(residual, dtype=np.float32).reshape(-1)
        if residual.shape[0] != self.action_dim:
            raise ValueError(f"Expected teacher action dim {self.action_dim}, got {residual.shape[0]}")
        target = qpos_before[self.audit.qpos_indices] + residual
        for value, joint in zip(target, self.audit.joints, strict=True):
            lo, hi = joint.joint_range
            self.env.data.qpos[joint.qposadr] = float(np.clip(value, lo, hi))
        mujoco.mj_forward(self.env.model, self.env.data)

    @property
    def disabled_muscle_names(self) -> tuple[str, ...]:
        """Muscles scaled by :attr:`prosthesis_muscle_scale` during prosthesis replacement."""

        return self._disabled_muscle_names

    def _resolve_disabled_muscle_names(self) -> tuple[str, ...]:
        if self.mask_preset == "legacy_boundary":
            return LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES
        return disabled_muscle_names_for_preset(self.mask_preset)

    def _build_obs(self) -> np.ndarray:
        return build_global_observation(self.env, self.audit, self._previous_speed, obs_mode=self.obs_mode)

    def _build_priv_info(self, target: np.ndarray) -> np.ndarray:
        return build_priv_info(self.env, self.audit, target, self._previous_speed, obs_mode=self.obs_mode)

    def _resolve_prosthesis_boundary_actuators(self) -> np.ndarray:
        ids: list[int] = []
        for name in self._disabled_muscle_names:
            aid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                ids.append(int(aid))
        return np.asarray(sorted(set(ids)), dtype=np.int32)

    def _backup_sim_state(self) -> dict:
        data = self.env.data
        return {
            "time": float(data.time),
            "qpos": np.asarray(data.qpos).copy(),
            "qvel": np.asarray(data.qvel).copy(),
            "act": np.asarray(data.act).copy(),
            "ctrl": np.asarray(data.ctrl).copy(),
            "qfrc_applied": np.asarray(data.qfrc_applied).copy(),
            "qacc_warmstart": np.asarray(data.qacc_warmstart).copy(),
            "carry": self.env._additional_carry,
            "obs": np.asarray(self.env._obs).copy(),
            "info": dict(self.env._info),
        }

    def _restore_sim_state(self, state: dict) -> None:
        data = self.env.data
        data.time = state["time"]
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
        data.act[:] = state["act"]
        data.ctrl[:] = state["ctrl"]
        data.qfrc_applied[:] = state["qfrc_applied"]
        data.qacc_warmstart[:] = state["qacc_warmstart"]
        self.env._additional_carry = state["carry"]
        self.env._obs = state["obs"]
        self.env._info = state["info"]
        mujoco.mj_forward(self.env.model, self.env.data)

    def _step_prosthesis_replacement_pd(
        self,
        oracle_action: np.ndarray,
        target_qpos: np.ndarray,
        feedforward_torque: np.ndarray | None = None,
    ):
        """One env step: oracle muscle control + PD torques on the 4 prosthesis DOFs (motor layer)."""

        target_qpos = np.asarray(target_qpos, dtype=np.float64).reshape(-1)
        cur_info = self.env._info.copy()
        carry = self.env._additional_carry.replace(last_action=oracle_action)
        processed_action, carry = self.env._preprocess_action(oracle_action, self.env._model, self.env._data, carry)
        self.env._model, self.env._data, carry = self.env._simulation_pre_step(
            self.env._model, self.env._data, carry
        )

        torque_abs_sum = 0.0
        torque_abs_max = 0.0
        torque_count = 0
        torque_sum = np.zeros(len(self.audit.joints), dtype=np.float64)
        ff_torque = np.zeros(len(self.audit.joints), dtype=np.float64)
        if feedforward_torque is not None and self.oracle_torque_ff_scale != 0.0:
            ff_torque = np.asarray(feedforward_torque, dtype=np.float64).reshape(-1)
            if ff_torque.size != len(self.audit.joints):
                raise ValueError(f"Expected feedforward torque dim {len(self.audit.joints)}, got {ff_torque.size}")
            ff_torque = self.oracle_torque_ff_scale * ff_torque
            if self.oracle_torque_ff_limit is not None:
                ff_torque = np.clip(ff_torque, -self.oracle_torque_ff_limit, self.oracle_torque_ff_limit)
        for _ in range(self.env._n_intermediate_steps):
            self.env._data.qfrc_applied[:] = 0.0
            ctrl_action, carry = self.env._compute_action(processed_action, self.env._model, self.env._data, carry)
            self.env._data.ctrl[self.env._action_indices] = np.asarray(ctrl_action).reshape(-1)
            if self._prosthesis_boundary_actuator_indices.size:
                scale = float(np.clip(self._prosthesis_muscle_scale, 0.0, 1.0))
                self.env._data.ctrl[self._prosthesis_boundary_actuator_indices] *= scale

            q = np.asarray(self.env._data.qpos[self.audit.qpos_indices], dtype=np.float64)
            qd = np.asarray(self.env._data.qvel[self.audit.qvel_indices], dtype=np.float64)
            kp = self.pd_kp * self.pd_gain_scale
            kd = self.pd_kd * self.pd_gain_scale
            torque = ff_torque + kp * (target_qpos - q) - kd * qd
            torque = np.clip(torque, -self.pd_torque_limit, self.pd_torque_limit)
            if self.pd_torque_slew_limit is not None:
                delta = torque - self._last_prosthesis_pd_torque
                delta = np.clip(delta, -self.pd_torque_slew_limit, self.pd_torque_slew_limit)
                torque = self._last_prosthesis_pd_torque + delta
            self._last_prosthesis_pd_torque = torque.copy()
            torque_abs = np.abs(torque)
            torque_abs_sum += float(np.sum(torque_abs))
            torque_abs_max = max(torque_abs_max, float(np.max(torque_abs)))
            torque_count += int(torque_abs.size)
            torque_sum += torque
            self.env._data.qfrc_applied[self.audit.qvel_indices] = torque
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
        executed_qpos = np.asarray(self.env._data.qpos[self.audit.qpos_indices], dtype=np.float32)
        executed_qvel = np.asarray(self.env._data.qvel[self.audit.qvel_indices], dtype=np.float32)
        root_height = float(self.env._data.qpos[2])
        root_quat = np.asarray(self.env._data.qpos[3:7], dtype=np.float64)
        quat_norm = float(np.linalg.norm(root_quat))
        if quat_norm > 1e-8:
            root_quat = root_quat / quat_norm
        qw, qx, qy, _qz = root_quat
        # World-up component of the pelvis local z-axis. 1 is upright, 0 is sideways.
        root_up_z = float(1.0 - 2.0 * (qx * qx + qy * qy))
        toe_z = None
        for site_name, site_id in self.audit.sites:
            if "toe" in site_name or "toes" in site_name:
                toe_z = float(self.env._data.site_xpos[site_id, 2])
                break
        target_qpos32 = target_qpos.astype(np.float32)
        tracking_error = target_qpos32 - executed_qpos
        cur_info = dict(cur_info)
        cur_info["prosthesis_target_qpos"] = target_qpos32
        cur_info["prosthesis_executed_qpos"] = executed_qpos
        cur_info["prosthesis_qvel"] = executed_qvel
        cur_info["prosthesis_oracle_ff_torque"] = ff_torque.astype(np.float32)
        cur_info["prosthesis_applied_torque"] = self._last_prosthesis_pd_torque.astype(np.float32)
        cur_info["prosthesis_applied_torque_mean"] = (torque_sum / max(self.env._n_intermediate_steps, 1)).astype(
            np.float32
        )
        cur_info["prosthesis_tracking_error"] = tracking_error.astype(np.float32)
        cur_info["prosthesis_tracking_mae"] = float(np.mean(np.abs(tracking_error)))
        cur_info["prosthesis_tracking_mse"] = float(np.mean(np.square(tracking_error)))
        cur_info["prosthesis_qvel_ma"] = float(np.mean(np.abs(executed_qvel)))
        cur_info["prosthesis_root_height"] = root_height
        cur_info["prosthesis_root_up_z"] = root_up_z
        cur_info["prosthesis_ncon"] = int(self.env._data.ncon)
        if toe_z is not None:
            cur_info["prosthesis_toe_z"] = toe_z
        cur_info["prosthesis_pd_torque_abs_mean"] = float(torque_abs_sum / max(torque_count, 1))
        cur_info["prosthesis_pd_torque_abs_max"] = float(torque_abs_max)
        cur_info["prosthesis_muscle_scale"] = float(np.clip(self._prosthesis_muscle_scale, 0.0, 1.0))
        return np.asarray(cur_obs), reward, absorbing, done, cur_info

    # Backward-compatible name
    _step_with_stage1_pd = _step_prosthesis_replacement_pd

    def _oracle_equivalent_feedforward_target(
        self,
        oracle_torque: np.ndarray,
        target_qpos: np.ndarray,
    ) -> np.ndarray:
        """Feedforward label that makes PD+FF approximate the oracle muscle torque."""

        oracle_torque = np.asarray(oracle_torque, dtype=np.float64).reshape(-1)
        if self.torque_feedforward_target_mode == "oracle":
            return oracle_torque.astype(np.float32)
        q = np.asarray(self.env.data.qpos[self.audit.qpos_indices], dtype=np.float64)
        qd = np.asarray(self.env.data.qvel[self.audit.qvel_indices], dtype=np.float64)
        kp = self.pd_kp * self.pd_gain_scale
        kd = self.pd_kd * self.pd_gain_scale
        pd_term = kp * (np.asarray(target_qpos, dtype=np.float64).reshape(-1) - q) - kd * qd
        return (oracle_torque - pd_term).astype(np.float32)

    def _oracle_step_prosthesis_labels(
        self, oracle_action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance physics with oracle only; return labels plus oracle generalized forces on prosthesis DOFs."""

        sim_state = self._backup_sim_state()
        oracle_obs_state = self._oracle_obs
        _raw, _r, _a, _d, _i = self.env.step(oracle_action)
        qpos_after = np.asarray(self.env.data.qpos, dtype=np.float32).copy()
        qvel_after = np.asarray(self.env.data.qvel, dtype=np.float32).copy()
        torque_after = np.asarray(self.env.data.qfrc_actuator[self.audit.qvel_indices], dtype=np.float32).copy()
        self._restore_sim_state(sim_state)
        self._oracle_obs = oracle_obs_state
        return qpos_after, qvel_after, torque_after

    def _prosthesis_replacement_dagger_step(
        self,
        teacher_action: np.ndarray,
        oracle_action: np.ndarray,
        teacher_torque: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, dict, np.ndarray, np.ndarray, np.ndarray]:
        """Oracle labels on ghost step, then physical step with Stage1 PD (DAgger prosthesis replacement)."""

        qpos_oracle_after, qvel_oracle_after, torque_oracle_after = self._oracle_step_prosthesis_labels(oracle_action)
        target = np.asarray(teacher_action, dtype=np.float32)
        target = np.asarray(
            [
                np.clip(value, joint.joint_range[0], joint.joint_range[1])
                for value, joint in zip(target, self.audit.joints, strict=True)
            ],
            dtype=np.float32,
        )
        if self.max_prosthesis_qpos_step is not None and self._last_commanded_prosthesis_qpos is not None:
            lim = self.max_prosthesis_qpos_step.astype(np.float64)
            delta = target.astype(np.float64) - self._last_commanded_prosthesis_qpos.astype(np.float64)
            delta = np.clip(delta, -lim, lim)
            target = (self._last_commanded_prosthesis_qpos.astype(np.float64) + delta).astype(np.float32)
            target = np.asarray(
                [
                    np.clip(value, joint.joint_range[0], joint.joint_range[1])
                    for value, joint in zip(target, self.audit.joints, strict=True)
                ],
                dtype=np.float32,
            )
        self._last_commanded_prosthesis_qpos = target.copy()
        oracle_ff_target = self._oracle_equivalent_feedforward_target(torque_oracle_after, target)
        torque_ff = oracle_ff_target if teacher_torque is None else np.asarray(teacher_torque, dtype=np.float32)
        raw_obs, reward, _absorbing, done, info = self._step_prosthesis_replacement_pd(
            oracle_action, target, torque_ff
        )
        info["prosthesis_oracle_ff_target"] = oracle_ff_target.astype(np.float32)
        return raw_obs, reward, done, info, qpos_oracle_after, qvel_oracle_after, torque_oracle_after

    def step(self, teacher_action: np.ndarray | None = None, teacher_torque: np.ndarray | None = None) -> HybridStep:
        if self._oracle_obs is None:
            self.reset()

        qpos_before = np.asarray(self.env.data.qpos, dtype=np.float32).copy()
        qvel_before = np.asarray(self.env.data.qvel, dtype=np.float32).copy()
        obs_before = self._build_obs()
        hist_before = self.history.as_array()
        if self.target_mode == "qpos":
            placeholder_target = qpos_before[self.audit.qpos_indices].astype(np.float32)
        else:
            placeholder_target = np.zeros(self.action_dim, dtype=np.float32)
        priv_before = self._build_priv_info(placeholder_target)

        oracle_action = self.oracle.act(self._oracle_obs)
        if teacher_action is not None and self.apply_teacher_action and self.pd_override and self.target_mode == "qpos":
            raw_obs, reward, done, info, qpos_after, qvel_after, oracle_torque_target = (
                self._prosthesis_replacement_dagger_step(teacher_action, oracle_action, teacher_torque)
            )
        else:
            raw_obs, reward, _absorbing, done, info = self.env.step(oracle_action)
            if teacher_action is not None and self.apply_teacher_action:
                self._apply_joint_residual(teacher_action, qpos_before)
            qpos_after = np.asarray(self.env.data.qpos, dtype=np.float32).copy()
            qvel_after = np.asarray(self.env.data.qvel, dtype=np.float32).copy()
            oracle_torque_target = np.asarray(
                self.env.data.qfrc_actuator[self.audit.qvel_indices], dtype=np.float32
            ).copy()
            if self.target_mode == "qpos":
                self._last_commanded_prosthesis_qpos = qpos_after[self.audit.qpos_indices].astype(np.float32).copy()

        if self.target_mode == "qpos":
            oracle_target = qpos_after[self.audit.qpos_indices].astype(np.float32)
        else:
            oracle_target = (qpos_after[self.audit.qpos_indices] - qpos_before[self.audit.qpos_indices]).astype(np.float32)

        self._oracle_obs = self.oracle.update_obs(raw_obs)
        obs = self._build_obs()
        self.history.append(obs)
        self._previous_speed = float(np.linalg.norm(np.asarray(self.env.data.qvel[:2], dtype=np.float64)))
        self._last_obs = obs
        info = dict(info)
        info["qpos_before"] = qpos_before
        info["qvel_before"] = qvel_before
        info["qpos_after"] = qpos_after
        info["qvel_after"] = qvel_after
        info["prosthesis_oracle_torque_target"] = oracle_torque_target.astype(np.float32)
        return HybridStep(
            obs_before,
            priv_before,
            hist_before,
            oracle_target,
            oracle_torque_target.astype(np.float32),
            float(reward),
            bool(done),
            info,
        )
