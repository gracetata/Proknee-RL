"""Evaluation adapters for different controllers and environments."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf

import musclemimic.environments  # noqa: F401
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.obs_mask import MaskedObservationEnvView, MaskedObsSpec, apply_obs_mask, build_masked_obs_spec
from musclemimic.distill.policy import PolicyRunner
from musclemimic.evaluation.contact_extractor import FootContactConfig, FootContactExtractor
from musclemimic.evaluation.types import EvalConfig, StepResult
from musclemimic.proknee.hybrid_env import MuscleProKneeHybridEnv
from musclemimic.proknee.models import MuscleProKneeStudent, MuscleProKneeTeacher
from musclemimic.prosthesis.controllers import FSMImpedanceProsthesisController, ReferencePDProsthesisController
from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint

from fullbody._eval_terminal import apply_eval_terminal_defaults


def _find_mask_spec(checkpoint_path: str, env) -> MaskedObsSpec:
    checkpoint = Path(checkpoint_path)
    candidates = [
        checkpoint / "masked_obs_spec.json",
        checkpoint.parent / "masked_obs_spec.json",
        checkpoint.parent.parent / "masked_obs_spec.json",
        checkpoint.parent / "distilled_metadata.json",
    ]
    try:
        resolved = checkpoint.resolve()
        candidates.extend(
            [
                resolved / "masked_obs_spec.json",
                resolved.parent / "masked_obs_spec.json",
                resolved.parent.parent / "masked_obs_spec.json",
                resolved.parent / "distilled_metadata.json",
            ]
        )
    except OSError:
        pass
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "masked_obs_spec.json":
            return MaskedObsSpec.load_json(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_payload = payload.get("masked_obs_spec")
        if spec_payload:
            return MaskedObsSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec_payload.items()})
    return build_masked_obs_spec(env)


def _numpy_backend():
    return np


def _get_reference(env, carry=None) -> dict[str, Any]:
    th = getattr(env, "th", None)
    if th is None:
        return {}
    if carry is None:
        carry = getattr(env, "_additional_carry", None) or getattr(env, "carry", None)
    if carry is None:
        traj_data = th.get_current_traj_data(None, _numpy_backend())
    else:
        traj_data = th.get_current_traj_data(carry, _numpy_backend())
    site_pos = np.asarray(traj_data.site_xpos, dtype=np.float32) if hasattr(traj_data, "site_xpos") else None
    qpos = np.asarray(traj_data.qpos, dtype=np.float64).reshape(-1)
    root_pos = qpos[:3].copy()
    return {"qpos": qpos, "qvel": np.asarray(traj_data.qvel, dtype=np.float64).reshape(-1), "site_pos": site_pos, "root_pos": root_pos}


class BaseEvalAdapter(ABC):
    def __init__(self, eval_config: EvalConfig, motion_path: str):
        self.eval_config = eval_config
        self.motion_path = motion_path
        self.env = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._contact: FootContactExtractor | None = None
        self._free_qpos_idx = None
        self.save_video = bool(eval_config.save_video)
        self.dt = 0.01

    def _init_contact(self):
        if self.model is None:
            return
        fc = FootContactConfig.from_dict(self.eval_config.eval_force)
        self._contact = FootContactExtractor(self.model, fc, dt=float(self.dt))

    @abstractmethod
    def reset(self, motion_path: str) -> np.ndarray: ...

    @abstractmethod
    def step(self) -> StepResult: ...

    def get_traj_length(self) -> int:
        if self.env is None or getattr(self.env, "th", None) is None:
            return 1000
        return int(self.env.th.len_trajectory(0))

    def get_body_mass(self) -> float:
        if self.model is None:
            return 0.0
        return float(np.sum(self.model.body_mass))

    def get_root_pos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:3], dtype=np.float64).copy()

    def get_root_quat(self) -> np.ndarray:
        return np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()

    def get_qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos, dtype=np.float64).copy()

    def get_qvel(self) -> np.ndarray:
        return np.asarray(self.data.qvel, dtype=np.float64).copy()

    def get_site_pos(self) -> np.ndarray | None:
        if self.env is None:
            return None
        sites = getattr(self.env, "sites_for_mimic", []) or []
        if not sites:
            return None
        out = []
        for name in sites:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                out.append(np.asarray(self.data.site_xpos[sid], dtype=np.float32))
        return np.stack(out, axis=0) if out else None

    def get_contact_forces(self) -> dict[str, Any]:
        if self._contact is None or self.data is None:
            return {}
        return self._contact.extract(self.data)

    def get_joint_torques(self) -> dict[str, np.ndarray]:
        if self.data is None:
            return {"joint_torque": np.zeros(0), "prosthesis_tau": np.zeros(0)}
        tau = np.asarray(self.data.qfrc_actuator, dtype=np.float64).copy()
        prosthesis_tau = np.zeros(0, dtype=np.float64)
        if hasattr(self.env, "get_prosthesis_obs"):
            po = self.env.get_prosthesis_obs()
            prosthesis_tau = np.asarray(po.get("prev_tau", po.get("prosthesis_tau", np.zeros(4))), dtype=np.float64)
        elif hasattr(self.env, "_last_prosthesis_tau"):
            prosthesis_tau = np.asarray(self.env._last_prosthesis_tau, dtype=np.float64)
        return {"joint_torque": tau, "prosthesis_tau": prosthesis_tau.reshape(-1)}

    def render(self, record: bool = False) -> None:
        if self.env is not None:
            self.env.render(record=record)

    def close(self) -> None:
        if self.env is not None and hasattr(self.env, "stop"):
            try:
                self.env.stop()
            except Exception:
                pass

    def meta_for_metrics(self) -> dict[str, Any]:
        bw = self.get_body_mass() * 9.81
        site_names = list(getattr(self.env, "sites_for_mimic", []) or [])
        limits = None
        if hasattr(self.env, "model"):
            try:
                from musclemimic.distill.mapping import build_distill_mapping
                from musclemimic.distill.config import make_bare_env

                # optional prosthesis limits not always available
            except Exception:
                pass
        return {
            "body_weight_N": bw,
            "site_names": site_names,
            "torque_limits": limits,
        }


class OfficialMuscleMimicAdapter(BaseEvalAdapter):
    def __init__(self, eval_config: EvalConfig, motion_path: str):
        super().__init__(eval_config, motion_path)
        self.config, agent_state, _ = load_checkpoint(eval_config.checkpoint_path)
        OmegaConf.set_struct(self.config, False)
        self._build_env(motion_path)
        self.env = self._env
        self.model = self.env.model
        self.data = self.env.data
        self.dt = float(self.env.dt)
        self._init_contact()
        self._policy = PolicyRunner.from_agent_state(
            PPOJax.init_agent_conf(self.env, self.config),
            align_agent_state(agent_state, PPOJax.init_agent_conf(self.env, self.config)),
            self.env,
            deterministic=True,
            seed=eval_config.eval_seed,
        )
        self._obs_policy = None

    def _build_env(self, motion_path: str):
        cfg = deepcopy(self.config)
        env_params = OmegaConf.to_container(cfg.experiment.env_params, resolve=True)
        task_params = OmegaConf.to_container(cfg.experiment.task_factory.params, resolve=True)
        env_params["env_name"] = "MyoFullBody" if self.eval_config.use_mujoco else env_params.get("env_name")
        if self.eval_config.use_mujoco and "Mjx" in str(env_params.get("env_name", "")):
            env_params["env_name"] = str(env_params["env_name"]).replace("Mjx", "")
        env_params.pop("prosthesis", None)
        env_params["headless"] = True
        if self.eval_config.save_video:
            fps = int(round(1.0 / max(float(env_params.get("dt", 0.01)), 1e-6)))
            env_params["recorder_params"] = {
                "path": str(self.eval_config.output_dir / "per_motion"),
                "tag": motion_path.replace("/", "_"),
                "video_name": "rollout",
                "fps": fps,
                "compress": True,
            }
        goal_params = env_params.get("goal_params", {})
        goal_params["visualize_goal"] = self.eval_config.show_ghost
        env_params["goal_params"] = goal_params
        if self.eval_config.no_termination:
            env_params["terminal_state_type"] = "NoTerminalStateHandler"
        else:
            apply_eval_terminal_defaults(env_params, cfg, strict_termination=False)
        amass = dict(task_params.get("amass_dataset_conf", {}) or {})
        amass["rel_dataset_path"] = [motion_path]
        amass["dataset_group"] = None
        task_params["amass_dataset_conf"] = amass
        th_params = dict(env_params.get("th_params", {}) or {})
        th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
        env_params["th_params"] = th_params
        factory = TaskFactory.get_factory_cls(cfg.experiment.task_factory.name)
        self._env = factory.make(**{**env_params, **task_params})

    def reset(self, motion_path: str) -> np.ndarray:
        if self._contact is not None:
            self._contact.reset_filters()
        obs = self.env.reset()
        self._obs_policy = self._policy.reset_obs(obs)
        return np.asarray(obs).reshape(-1)

    def step(self) -> StepResult:
        action, _ = self._policy.act(self._obs_policy)
        obs, reward, absorbing, done, info = self.env.step(action)
        self._obs_policy = self._policy.update_obs(obs)
        ref = _get_reference(self.env)
        return StepResult(
            obs=obs,
            reward=float(np.asarray(reward).item()),
            absorbing=bool(absorbing),
            done=bool(done),
            info=dict(info),
            reference=ref,
        )


class MaskedFullMusclePolicyAdapter(OfficialMuscleMimicAdapter):
    """Evaluate full-muscle distilled policies that consume masked observations."""

    def __init__(self, eval_config: EvalConfig, motion_path: str):
        BaseEvalAdapter.__init__(self, eval_config, motion_path)
        self.config, agent_state, _ = load_checkpoint(eval_config.checkpoint_path)
        OmegaConf.set_struct(self.config, False)
        self._build_env(motion_path)
        self.env = self._env
        self.model = self.env.model
        self.data = self.env.data
        self.dt = float(self.env.dt)
        self._init_contact()
        self._mask_spec = _find_mask_spec(eval_config.checkpoint_path, self.env)
        self._env_view = MaskedObservationEnvView(self.env, self._mask_spec)
        agent_conf = PPOJax.init_agent_conf(self._env_view, self.config)
        self._policy = PolicyRunner.from_agent_state(
            agent_conf,
            align_agent_state(agent_state, agent_conf),
            self._env_view,
            deterministic=True,
            seed=eval_config.eval_seed,
        )
        self._obs_policy = None

    def reset(self, motion_path: str) -> np.ndarray:
        if self._contact is not None:
            self._contact.reset_filters()
        obs = self.env.reset()
        self._obs_policy = self._policy.reset_obs(apply_obs_mask(obs, self._mask_spec))
        return np.asarray(obs).reshape(-1)

    def step(self) -> StepResult:
        action, _ = self._policy.act(self._obs_policy)
        obs, reward, absorbing, done, info = self.env.step(action)
        self._obs_policy = self._policy.update_obs(apply_obs_mask(obs, self._mask_spec))
        ref = _get_reference(self.env)
        return StepResult(
            obs=obs,
            reward=float(np.asarray(reward).item()),
            absorbing=bool(absorbing),
            done=bool(done),
            info=dict(info),
            reference=ref,
        )

    def meta_for_metrics(self) -> dict[str, Any]:
        meta = super().meta_for_metrics()
        meta["masked_obs_dim"] = self._mask_spec.masked_obs_dim
        meta["raw_obs_dim"] = self._mask_spec.raw_obs_dim
        return meta


class ProsthesisPolicyAdapter(BaseEvalAdapter):
    def __init__(self, eval_config: EvalConfig, motion_path: str, *, prosthesis_controller=None):
        super().__init__(eval_config, motion_path)
        self.config, agent_state, _ = load_checkpoint(eval_config.checkpoint_path)
        OmegaConf.set_struct(self.config, False)
        self._prosthesis_controller = prosthesis_controller
        self._build_env(motion_path)
        self.env = self._env
        self.model = self.env.model
        self.data = self.env.data
        self.dt = float(self.env.dt)
        self._init_contact()
        agent_conf = PPOJax.init_agent_conf(self.env, self.config)
        self._policy = PolicyRunner.from_agent_state(
            agent_conf, align_agent_state(agent_state, agent_conf), self.env, deterministic=True, seed=eval_config.eval_seed
        )
        self._obs_policy = None

    def _build_env(self, motion_path: str):
        cfg = deepcopy(self.config)
        env_params = OmegaConf.to_container(cfg.experiment.env_params, resolve=True)
        task_params = OmegaConf.to_container(cfg.experiment.task_factory.params, resolve=True)
        env_params["env_name"] = "MyoFullBodyProsthesisEnv"
        env_params["headless"] = True
        prosthesis = dict(env_params.get("prosthesis", {}) or {})
        prosthesis["enabled"] = True
        prosthesis["control_mode"] = self.eval_config.prosthesis_control_mode
        env_params["prosthesis"] = prosthesis
        goal_params = env_params.get("goal_params", {})
        goal_params["visualize_goal"] = self.eval_config.show_ghost
        env_params["goal_params"] = goal_params
        if self.eval_config.no_termination:
            env_params["terminal_state_type"] = "NoTerminalStateHandler"
        else:
            apply_eval_terminal_defaults(env_params, cfg, strict_termination=False)
        if self.eval_config.save_video:
            env_params["recorder_params"] = {
                "path": str(self.eval_config.output_dir / "per_motion"),
                "tag": motion_path.replace("/", "_"),
                "video_name": "rollout",
                "fps": int(round(1.0 / 0.01)),
                "compress": True,
            }
        amass = dict(task_params.get("amass_dataset_conf", {}) or {})
        amass["rel_dataset_path"] = [motion_path]
        amass["dataset_group"] = None
        task_params["amass_dataset_conf"] = amass
        th_params = dict(env_params.get("th_params", {}) or {})
        th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
        env_params["th_params"] = th_params
        factory = TaskFactory.get_factory_cls(cfg.experiment.task_factory.name)
        merged = {**env_params, **task_params}
        if self._prosthesis_controller is not None:
            merged["prosthesis_controller"] = self._prosthesis_controller
        self._env = factory.make(**merged)

    def reset(self, motion_path: str) -> np.ndarray:
        if self._contact is not None:
            self._contact.reset_filters()
        obs = self.env.reset()
        self._obs_policy = self._policy.reset_obs(obs)
        return np.asarray(obs).reshape(-1)

    def step(self) -> StepResult:
        action, _ = self._policy.act(self._obs_policy)
        obs, reward, absorbing, done, info = self.env.step(action)
        self._obs_policy = self._policy.update_obs(obs)
        ref = _get_reference(self.env)
        return StepResult(
            obs=obs,
            reward=float(np.asarray(reward).item()),
            absorbing=bool(absorbing),
            done=bool(done),
            info=dict(info),
            reference=ref,
        )

    def meta_for_metrics(self) -> dict[str, Any]:
        meta = super().meta_for_metrics()
        try:
            from musclemimic.distill.mapping import build_distill_mapping
            from musclemimic.distill.config import make_bare_env

            teacher = make_bare_env(self.config, env_name="MyoFullBody", use_mujoco=True)
            student = make_bare_env(self.config, env_name="MyoFullBodyProsthesisEnv", use_mujoco=True)
            mapping = build_distill_mapping(teacher, student)
            meta["torque_limits"] = mapping.prosthesis_torque_limits
        except Exception:
            pass
        return meta


class ProsthesisExternalControllerAdapter(ProsthesisPolicyAdapter):
    def __init__(self, eval_config: EvalConfig, motion_path: str):
        if eval_config.prosthesis_controller == "reference_pd":
            controller = ReferencePDProsthesisController()
        elif eval_config.prosthesis_controller == "fsm_impedance":
            controller = FSMImpedanceProsthesisController()
        else:
            raise ValueError(f"Unknown prosthesis_controller={eval_config.prosthesis_controller}")
        super().__init__(eval_config, motion_path, prosthesis_controller=controller)


class ProKneeHybridAdapter(BaseEvalAdapter):
    def __init__(self, eval_config: EvalConfig, motion_path: str):
        super().__init__(eval_config, motion_path)
        if not eval_config.proknee_checkpoint:
            raise ValueError("proknee_checkpoint required for env_type=proknee_hybrid")
        with open(eval_config.proknee_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        self._ckpt = ckpt
        target_mode = str(ckpt.get("target_mode", "qpos"))
        self._hybrid = MuscleProKneeHybridEnv(
            eval_config.checkpoint_path,
            rel_dataset_path=[motion_path],
            history_len=int(ckpt.get("history_len", 30)),
            apply_teacher_action=True,
            target_mode=target_mode,
            pd_override=True,
            deterministic_oracle=True,
        )
        self.env = self._hybrid.env
        self.model = self.env.model
        self.data = self.env.data
        self.dt = float(self.env.dt)
        self._init_contact()
        params = ckpt["params"]
        action_dim = int(ckpt.get("action_dim", self._hybrid.action_dim))
        if target_mode == "qpos":
            teacher = MuscleProKneeTeacher(action_dim=action_dim)

            @jax.jit
            def predict(p, obs, priv, hist):
                pred, _ = teacher.apply(p, obs, priv)
                return pred

        else:
            student = MuscleProKneeStudent(action_dim=action_dim)

            @jax.jit
            def predict(p, obs, priv, hist):
                pred, _ = student.apply(p, obs, priv, hist, mode="student")
                return pred

        self._predict = predict
        self._params = params
        self._state = None

    def reset(self, motion_path: str) -> np.ndarray:
        if self._contact is not None:
            self._contact.reset_filters()
        step = self._hybrid.reset()
        self._state = step
        return np.asarray(step.obs).reshape(-1)

    def step(self) -> StepResult:
        assert self._state is not None
        obs = jnp.asarray(self._state.obs[None, :], dtype=jnp.float32)
        priv = jnp.asarray(self._state.priv_info[None, :], dtype=jnp.float32)
        hist = jnp.asarray(self._state.proprio_hist[None, :, :], dtype=jnp.float32)
        pred = np.asarray(self._predict(self._params, obs, priv, hist)[0], dtype=np.float32).reshape(-1)
        step = self._hybrid.step(teacher_action=pred)
        self._state = step
        ref = _get_reference(self.env)
        info = dict(step.info)
        return StepResult(
            obs=step.obs,
            reward=float(step.reward),
            absorbing=bool(info.get("terminated", False)),
            done=bool(step.done),
            info=info,
            reference=ref,
        )

    def get_joint_torques(self) -> dict[str, np.ndarray]:
        tau = np.asarray(getattr(self._hybrid, "_last_prosthesis_pd_torque", np.zeros(4)), dtype=np.float64)
        base = super().get_joint_torques()
        base["prosthesis_tau"] = tau.reshape(-1)
        return base


def build_adapter(eval_config: EvalConfig, motion_path: str) -> BaseEvalAdapter:
    env_type = eval_config.env_type
    if env_type == "official_fullbody":
        return OfficialMuscleMimicAdapter(eval_config, motion_path)
    if env_type == "masked_full_muscle":
        return MaskedFullMusclePolicyAdapter(eval_config, motion_path)
    if env_type == "prosthesis":
        return ProsthesisPolicyAdapter(eval_config, motion_path)
    if env_type == "prosthesis_external":
        from dataclasses import replace

        cfg = replace(eval_config, prosthesis_control_mode="eval_external_controller")
        return ProsthesisExternalControllerAdapter(cfg, motion_path)
    if env_type == "proknee_hybrid":
        return ProKneeHybridAdapter(eval_config, motion_path)
    raise ValueError(f"Unknown env_type={env_type}")


def resolve_n_steps(n_steps: str | int, traj_len: int) -> int:
    if str(n_steps).lower() == "auto":
        return int(traj_len)
    return int(n_steps)
