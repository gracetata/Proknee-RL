#!/usr/bin/env python
"""Print and save MyoFullBody prosthesis name/id mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import musclemimic.environments  # noqa: F401 - registers envs
from loco_mujoco.task_factories import RLFactory

from musclemimic.prosthesis.constants import PROSTHESIS_JOINT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=None, help="Hydra config from fullbody/, e.g. conf_fullbody_prosthesis_gmr_resnet")
    parser.add_argument("--env-name", default="MyoFullBodyProsthesisEnv")
    parser.add_argument("--teacher-env-name", default="MyoFullBody")
    parser.add_argument("--output-dir", default="outputs/prosthesis_distill_debug")
    parser.add_argument(
        "--disable-mode",
        default="hard_zero_force",
        choices=[
            "mask_ctrl",
            "default",
            "hard_zero_force",
            "zero_force",
            "name_matching",
            "name_matching_hard_zero_force",
            "auto",
            "auto_hard_zero_force",
        ],
    )
    parser.add_argument("--disable-include", default="")
    parser.add_argument("--disable-exclude", default="")
    return parser.parse_args()


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_name: str | None):
    if config_name is None:
        return None
    with initialize_config_dir(version_base=None, config_dir=str(repo_root() / "fullbody")):
        return compose(config_name=config_name)


def make_env_from_config(config, env_name: str):
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = env_name
    env_params["headless"] = True
    if env_name in {"MyoFullBody", "MjxMyoFullBody"}:
        env_params.pop("prosthesis", None)
    env_name = env_params.pop("env_name")
    return RLFactory.make(env_name, **env_params)


def make_env_from_args(args):
    return RLFactory.make(
        args.env_name,
        disable_fingers=True,
        prosthesis={
            "enabled": True,
            "control_mode": "train_policy",
            "disable_muscles": {
                "mode": args.disable_mode,
                "include": split_csv(args.disable_include),
                "exclude": split_csv(args.disable_exclude),
            },
        },
    )


def actuator_payload(model, names):
    payload = {}
    for aid, name in enumerate(names):
        if not name:
            continue
        payload[name] = {
            "id": int(aid),
            "dyntype": int(model.actuator_dyntype[aid]),
            "trntype": int(model.actuator_trntype[aid]),
            "ctrlrange": [float(x) for x in model.actuator_ctrlrange[aid]],
            "forcerange": [float(x) for x in model.actuator_forcerange[aid]],
        }
    return payload


def validate_action_layout(env, mapping):
    action_names = [env.model.actuator(int(aid)).name for aid in getattr(env, "_action_indices", [])]
    expected = list(mapping["remaining_muscle_names"]) + list(mapping["prosthesis_motor_names"])
    if action_names != expected:
        for idx, (got, exp) in enumerate(zip(action_names, expected, strict=False)):
            if got != exp:
                raise ValueError(f"Action layout mismatch at {idx}: got={got!r}, expected={exp!r}")
        raise ValueError(f"Action layout length mismatch: got={len(action_names)}, expected={len(expected)}")
    action_dim = int(env.info.action_space.shape[0])
    expected_dim = len(mapping["remaining_muscle_ids"]) + len(mapping["prosthesis_joint_names"])
    if action_dim != expected_dim:
        raise ValueError(f"action_dim mismatch: action_dim={action_dim}, expected={expected_dim}")
    return action_names


def validate_teacher_mapping(teacher_env, mapping):
    teacher_actions = [
        teacher_env.model.actuator(int(aid)).name
        for aid in getattr(teacher_env, "_action_indices", range(teacher_env.model.nu))
    ]
    missing_remaining = [name for name in mapping["remaining_muscle_names"] if name not in teacher_actions]
    missing_disabled = [name for name in mapping["disabled_muscle_names"] if name not in teacher_actions]
    if missing_remaining or missing_disabled:
        raise ValueError(
            "Teacher fullbody action mapping incomplete: "
            f"missing_remaining={missing_remaining[:10]}, missing_disabled={missing_disabled[:10]}"
        )
    qpos_indices = []
    qvel_indices = []
    for name in PROSTHESIS_JOINT_NAMES:
        jid = mujoco.mj_name2id(teacher_env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Teacher model missing prosthesis joint: {name}")
        qpos_indices.append(int(teacher_env.model.jnt_qposadr[jid]))
        qvel_indices.append(int(teacher_env.model.jnt_dofadr[jid]))

    return {
        "env_name": teacher_env.__class__.__name__,
        "action_dim": int(teacher_env.info.action_space.shape[0]),
        "obs_dim": int(teacher_env.info.observation_space.shape[0]),
        "remaining_action_indices": [int(teacher_actions.index(name)) for name in mapping["remaining_muscle_names"]],
        "disabled_action_indices": [int(teacher_actions.index(name)) for name in mapping["disabled_muscle_names"]],
        "prosthesis_dof_ids": qvel_indices,
        "prosthesis_qpos_indices": qpos_indices,
        "prosthesis_qvel_indices": qvel_indices,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config_name)
    if config is None:
        env = make_env_from_args(args)
        teacher_env = RLFactory.make(args.teacher_env_name, disable_fingers=True)
    else:
        env = make_env_from_config(config, args.env_name)
        teacher_env = make_env_from_config(config, args.teacher_env_name)

    mapping = env.prosthesis_mapping.to_json_dict()
    model = env.model
    action_names = validate_action_layout(env, mapping)
    teacher_payload = validate_teacher_mapping(teacher_env, mapping)

    payload = {
        **mapping,
        "env_name": env.__class__.__name__,
        "action_dim": int(env.info.action_space.shape[0]),
        "obs_dim": int(env.info.observation_space.shape[0]),
        "teacher": teacher_payload,
        "action_names": action_names,
        "n_all_muscles": int(sum(int(model.actuator_dyntype[i]) == int(mujoco.mjtDyn.mjDYN_MUSCLE) for i in range(model.nu))),
        "n_disabled_muscles": len(mapping["disabled_muscle_ids"]),
        "n_remaining_muscles": len(mapping["remaining_muscle_ids"]),
        "target_action_dim": len(mapping["remaining_muscle_ids"]) + len(mapping["prosthesis_joint_names"]),
        "actuator_types": actuator_payload(model, mapping["all_actuator_names"]),
    }
    out_path = out_dir / "mapping.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved mapping: {out_path}")
    print(f"env={payload['env_name']} action_dim={payload['action_dim']} obs_dim={payload['obs_dim']}")
    print(
        "muscles: "
        f"all={payload['n_all_muscles']} disabled={payload['n_disabled_muscles']} "
        f"remaining={payload['n_remaining_muscles']}"
    )
    print("prosthesis joints:")
    for name, jid, qpos, qvel, dof, aid in zip(
        mapping["prosthesis_joint_names"],
        mapping["prosthesis_joint_ids"],
        mapping["prosthesis_qpos_indices"],
        mapping["prosthesis_qvel_indices"],
        mapping["prosthesis_dof_ids"],
        mapping["prosthesis_actuator_ids"],
        strict=True,
    ):
        actuator_name = model.actuator(aid).name
        print(f"  {name}: joint={jid} qpos={qpos} qvel={qvel} dof={dof} actuator={aid}:{actuator_name}")
    print("disabled muscles:")
    for name in mapping["disabled_muscle_names"]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
