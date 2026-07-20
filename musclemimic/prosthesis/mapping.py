"""Name/id mapping helpers for the MyoFullBody prosthesis environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import mujoco
import numpy as np

from .constants import (
    DEFAULT_DISABLED_MUSCLE_NAMES,
    DEFAULT_PROSTHESIS_DISTILL_MASK_PRESET,
    DEFAULT_PROSTHESIS_TORQUE_LIMITS,
    LEFT_DISTAL_PATH_KEYWORDS,
    LEFT_LEG_NAME_KEYWORDS,
    PROSTHESIS_JOINT_NAMES,
    PROSTHESIS_MOTOR_NAMES,
    disabled_muscle_names_for_preset,
)


@dataclass(frozen=True)
class ProsthesisMapping:
    all_joint_names: tuple[str, ...]
    all_actuator_names: tuple[str, ...]
    all_tendon_names: tuple[str, ...]
    disabled_muscle_names: tuple[str, ...]
    remaining_muscle_names: tuple[str, ...]
    prosthesis_joint_names: tuple[str, ...]
    prosthesis_motor_names: tuple[str, ...]
    disabled_muscle_ids: tuple[int, ...]
    remaining_muscle_ids: tuple[int, ...]
    prosthesis_joint_ids: tuple[int, ...]
    prosthesis_qpos_indices: tuple[int, ...]
    prosthesis_qvel_indices: tuple[int, ...]
    prosthesis_dof_ids: tuple[int, ...]
    prosthesis_actuator_ids: tuple[int, ...]
    prosthesis_torque_limits: tuple[float, ...]

    def to_json_dict(self) -> dict:
        return asdict(self)


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> tuple[str, ...]:
    names: list[str] = []
    for idx in range(count):
        name = mujoco.mj_id2name(model, obj_type, idx)
        names.append("" if name is None else str(name))
    return tuple(names)


def _is_muscle_actuator(model: mujoco.MjModel, actuator_id: int) -> bool:
    return int(model.actuator_dyntype[actuator_id]) == int(mujoco.mjtDyn.mjDYN_MUSCLE)


def _normalize_name_list(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    return tuple(str(x) for x in value if str(x))


def default_disabled_muscle_names(
    model: mujoco.MjModel,
    *,
    mode: str = "default",
    preset: str | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return disabled muscle names using a preset and/or heuristics."""

    selection_mode = {
        "mask_ctrl": "default",
        "hard_zero_force": "default",
        "zero_force": "default",
        "name_matching_hard_zero_force": "name_matching",
        "auto_hard_zero_force": "auto",
    }.get(mode, mode)

    include_names = _normalize_name_list(include)
    exclude_names = _normalize_name_list(exclude)
    preset_name = str(preset or DEFAULT_PROSTHESIS_DISTILL_MASK_PRESET)

    if selection_mode in {"default", "mask_ctrl", "hard_zero_force", "zero_force"}:
        candidates = set(
            disabled_muscle_names_for_preset(
                preset_name,
                include=include_names,
                exclude=exclude_names,
            )
        )
    else:
        candidates = set(DEFAULT_DISABLED_MUSCLE_NAMES)
        if selection_mode in {"name_matching", "auto"}:
            for aid in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                if not name or not _is_muscle_actuator(model, aid):
                    continue
                lower = name.lower()
                leftish = any(token in lower for token in LEFT_LEG_NAME_KEYWORDS)
                distal = any(token in lower for token in LEFT_DISTAL_PATH_KEYWORDS)
                if leftish and distal:
                    candidates.add(name)
        candidates |= set(include_names)
        candidates -= set(exclude_names)

    if selection_mode not in {"default", "name_matching", "auto", "mask_ctrl", "hard_zero_force", "zero_force"}:
        raise ValueError(f"Unsupported disable_muscles mode: {mode}")

    return tuple(sorted(name for name in candidates if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0))


def build_prosthesis_mapping(
    model: mujoco.MjModel,
    *,
    disabled_mode: str = "default",
    disabled_preset: str | None = None,
    disabled_include: Iterable[str] | None = None,
    disabled_exclude: Iterable[str] | None = None,
    torque_limits: Iterable[float] = DEFAULT_PROSTHESIS_TORQUE_LIMITS,
) -> ProsthesisMapping:
    disabled_names = default_disabled_muscle_names(
        model,
        mode=disabled_mode,
        preset=disabled_preset,
        include=disabled_include,
        exclude=disabled_exclude,
    )
    disabled_set = set(disabled_names)

    all_actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    all_joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    all_tendon_names = _names(model, mujoco.mjtObj.mjOBJ_TENDON, model.ntendon)

    remaining_muscle_names = tuple(
        name
        for aid, name in enumerate(all_actuator_names)
        if name and _is_muscle_actuator(model, aid) and name not in disabled_set
    )

    prosthesis_joint_ids: list[int] = []
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    for name in PROSTHESIS_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Missing prosthesis joint: {name}")
        prosthesis_joint_ids.append(int(jid))
        qpos_indices.append(int(model.jnt_qposadr[jid]))
        qvel_indices.append(int(model.jnt_dofadr[jid]))

    prosthesis_actuator_ids: list[int] = []
    for name in PROSTHESIS_MOTOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"Missing prosthesis actuator: {name}")
        prosthesis_actuator_ids.append(int(aid))

    limits = tuple(float(x) for x in torque_limits)
    if len(limits) != len(PROSTHESIS_JOINT_NAMES):
        raise ValueError(f"Expected {len(PROSTHESIS_JOINT_NAMES)} torque limits, got {len(limits)}")

    return ProsthesisMapping(
        all_joint_names=all_joint_names,
        all_actuator_names=all_actuator_names,
        all_tendon_names=all_tendon_names,
        disabled_muscle_names=disabled_names,
        remaining_muscle_names=remaining_muscle_names,
        prosthesis_joint_names=PROSTHESIS_JOINT_NAMES,
        prosthesis_motor_names=PROSTHESIS_MOTOR_NAMES,
        disabled_muscle_ids=tuple(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)) for n in disabled_names),
        remaining_muscle_ids=tuple(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)) for n in remaining_muscle_names),
        prosthesis_joint_ids=tuple(prosthesis_joint_ids),
        prosthesis_qpos_indices=tuple(qpos_indices),
        prosthesis_qvel_indices=tuple(qvel_indices),
        prosthesis_dof_ids=tuple(qvel_indices),
        prosthesis_actuator_ids=tuple(prosthesis_actuator_ids),
        prosthesis_torque_limits=limits,
    )


def actuator_names_for_action_spec(model: mujoco.MjModel, mapping: ProsthesisMapping) -> list[str]:
    """Action spec = remaining muscles + four prosthesis motors."""

    return list(mapping.remaining_muscle_names) + list(mapping.prosthesis_motor_names)


def actuator_names_for_action_spec_from_spec(
    spec,
    *,
    disabled_names: Iterable[str],
) -> list[str]:
    disabled = set(disabled_names)
    names: list[str] = []
    for actuator in spec.actuators:
        if actuator.name in disabled:
            continue
        if actuator.name in PROSTHESIS_MOTOR_NAMES:
            names.append(actuator.name)
            continue
        if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
            names.append(actuator.name)
    return names


def prosthesis_body_descendants(model: mujoco.MjModel, joint_ids: Iterable[int]) -> set[int]:
    roots = {int(model.jnt_bodyid[int(jid)]) for jid in joint_ids}
    descendants: set[int] = set()
    for body_id in range(model.nbody):
        cur = int(body_id)
        while cur >= 0:
            if cur in roots:
                descendants.add(int(body_id))
                break
            cur = int(model.body_parentid[cur])
            if cur == 0 and cur not in roots:
                break
    return descendants


def contact_load(model: mujoco.MjModel, data, body_ids: set[int]) -> tuple[bool, float]:
    total_normal_force = 0.0
    force6 = np.zeros(6, dtype=np.float64)
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        if body1 not in body_ids and body2 not in body_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_id, force6)
        total_normal_force += max(float(force6[0]), 0.0)
    return total_normal_force > 1e-6, float(total_normal_force)
