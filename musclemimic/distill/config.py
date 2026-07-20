"""Hydra/config helpers for prosthesis distillation scripts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import musclemimic.environments  # noqa: F401 - registers environments
from loco_mujoco.task_factories import RLFactory, TaskFactory


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_fullbody_config(config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(repo_root() / "fullbody")):
        return compose(config_name=config_name)


def parse_optional_csv(value: str | None) -> list[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_optional_vec4(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None or str(value).strip() == "":
        return None
    vals = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if len(vals) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got {value!r}")
    return vals


def apply_prosthesis_overrides(
    config,
    *,
    disable_mode: str | None = None,
    disable_preset: str | None = None,
    disable_include: list[str] | tuple[str, ...] | None = None,
    disable_exclude: list[str] | tuple[str, ...] | None = None,
    torque_limits: tuple[float, float, float, float] | None = None,
    action_type: str | None = None,
    residual_pd_kp: tuple[float, float, float, float] | None = None,
    residual_pd_kd: tuple[float, float, float, float] | None = None,
    torque_slew_limit: float | tuple[float, float, float, float] | None = None,
    tau_lowpass_alpha: float | None = None,
):
    OmegaConf.set_struct(config, False)
    env_params = config.experiment.env_params
    OmegaConf.set_struct(env_params, False)
    prosthesis = env_params.setdefault("prosthesis", {})
    if not isinstance(prosthesis, dict):
        prosthesis = env_params.prosthesis
    OmegaConf.set_struct(prosthesis, False)
    if action_type:
        prosthesis["action_type"] = str(action_type)
    if torque_limits is not None:
        prosthesis["torque_limits"] = {
            "knee": float(torque_limits[0]),
            "ankle": float(torque_limits[1]),
            "subtalar": float(torque_limits[2]),
            "mtp": float(torque_limits[3]),
        }
    disable_cfg = prosthesis.setdefault("disable_muscles", {})
    if disable_mode:
        disable_cfg["mode"] = str(disable_mode)
    if disable_preset:
        disable_cfg["preset"] = str(disable_preset)
    if disable_include is not None:
        disable_cfg["include"] = list(disable_include)
    if disable_exclude is not None:
        disable_cfg["exclude"] = list(disable_exclude)
    prosthesis_plain = dict(OmegaConf.to_container(prosthesis, resolve=True) or {})
    if residual_pd_kp is not None or residual_pd_kd is not None:
        residual = dict(prosthesis_plain.get("residual_pd") or {})
        if residual_pd_kp is not None:
            residual["kp"] = [float(x) for x in residual_pd_kp]
        if residual_pd_kd is not None:
            residual["kd"] = [float(x) for x in residual_pd_kd]
        prosthesis_plain["residual_pd"] = residual
    if torque_slew_limit is not None or tau_lowpass_alpha is not None:
        execution = dict(prosthesis_plain.get("execution") or {})
        if torque_slew_limit is not None:
            if isinstance(torque_slew_limit, tuple):
                execution["torque_slew_limit"] = [float(x) for x in torque_slew_limit]
            else:
                execution["torque_slew_limit"] = float(torque_slew_limit)
        if tau_lowpass_alpha is not None:
            execution["tau_lowpass_alpha"] = float(tau_lowpass_alpha)
        prosthesis_plain["execution"] = execution
    env_params.prosthesis = OmegaConf.create(prosthesis_plain)
    return config


def as_plain_dict(value: Any) -> dict:
    return OmegaConf.to_container(value, resolve=True)


def make_env(config, *, env_name: str | None = None, motion_paths: list[str] | None = None, motion_group: str | None = None, use_mujoco: bool = True, fixed_start: bool = True):
    cfg = deepcopy(config)
    OmegaConf.set_struct(cfg, False)
    env_params = as_plain_dict(cfg.experiment.env_params)
    task_params = as_plain_dict(cfg.experiment.task_factory.params)
    if env_name is not None:
        env_params["env_name"] = env_name
    if use_mujoco and str(env_params.get("env_name", "")).startswith("Mjx"):
        env_params["env_name"] = str(env_params["env_name"]).replace("Mjx", "", 1)
    env_params["headless"] = True
    if env_params["env_name"] in {"MyoFullBody", "MjxMyoFullBody"}:
        env_params.pop("prosthesis", None)

    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    if motion_paths:
        amass["rel_dataset_path"] = list(motion_paths)
        amass["dataset_group"] = None
    elif motion_group:
        amass["dataset_group"] = motion_group
        amass.pop("rel_dataset_path", None)
    if amass:
        task_params["amass_dataset_conf"] = amass

    th_params = dict(env_params.get("th_params", {}) or {})
    if fixed_start:
        th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params

    factory = TaskFactory.get_factory_cls(cfg.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


def make_bare_env(config, *, env_name: str | None = None, use_mujoco: bool = True):
    env_params = as_plain_dict(config.experiment.env_params)
    if env_name is not None:
        env_params["env_name"] = env_name
    if use_mujoco and str(env_params.get("env_name", "")).startswith("Mjx"):
        env_params["env_name"] = str(env_params["env_name"]).replace("Mjx", "", 1)
    if env_params["env_name"] in {"MyoFullBody", "MjxMyoFullBody"}:
        env_params.pop("prosthesis", None)
    env_params["headless"] = True
    name = env_params.pop("env_name")
    return RLFactory.make(name, **env_params)


def motion_list_from_group(dataset_group: str) -> list[str]:
    from loco_mujoco.task_factories.dataset_confs import expand_amass_dataset_group_spec, get_amass_dataset_groups

    groups = get_amass_dataset_groups()
    motions: list[str] = []
    for name in expand_amass_dataset_group_spec(dataset_group):
        if name not in groups:
            raise KeyError(f"Unknown dataset_group={name!r}. Available groups: {sorted(groups)}")
        motions.extend(str(x) for x in groups[name])
    return motions


def filter_motions(motions: list[str], motion_filter: str | None) -> list[str]:
    if not motion_filter or motion_filter.lower() == "none":
        return motions
    token = motion_filter.lower()
    return [m for m in motions if token in m.lower()]
