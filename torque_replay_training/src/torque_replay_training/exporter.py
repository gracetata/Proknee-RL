"""Full-body tracker rollout with per-physics-step generalized-force capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .constants import PROSTHESIS_JOINT_NAMES
from .control import root_up_z
from .schema import SCHEMA_VERSION, TorqueReplayDataset
from .upstream import OfficialPolicyRunner, build_fullbody_env


@dataclass
class StepTrace:
    observation: np.ndarray
    reward: float
    absorbing: bool
    done: bool
    info: dict
    reference_qpos: np.ndarray
    reference_qvel: np.ndarray
    qacc: np.ndarray
    actuator_ctrl: np.ndarray
    actuator_force: np.ndarray
    qfrc_actuator: np.ndarray
    qfrc_passive: np.ndarray
    qfrc_constraint: np.ndarray
    contact_ncon: np.ndarray


def _object_names(model, object_type, count: int) -> list[str]:
    return [str(mujoco.mj_id2name(model, object_type, idx) or "") for idx in range(int(count))]


def _joint_metadata(model) -> tuple[list[dict], list[int], list[int], list[int]]:
    rows: list[dict] = []
    prosthesis_dofs: list[int] = []
    prosthesis_qpos: list[int] = []
    root_dofs: list[int] = []
    target = set(PROSTHESIS_JOINT_NAMES)
    for joint_id in range(model.njnt):
        name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "")
        joint_type = int(model.jnt_type[joint_id])
        qposadr = int(model.jnt_qposadr[joint_id])
        dofadr = int(model.jnt_dofadr[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            nq, nv = 7, 6
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            nq, nv = 4, 3
        else:
            nq, nv = 1, 1
        rows.append(
            {
                "name": name,
                "joint_id": int(joint_id),
                "joint_type": joint_type,
                "qposadr": qposadr,
                "dofadr": dofadr,
                "nq": nq,
                "nv": nv,
            }
        )
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            root_dofs.extend(range(dofadr, dofadr + nv))
        if name in target:
            if nv != 1 or nq != 1:
                raise ValueError(f"prosthesis joint {name} must be scalar, got nq={nq}, nv={nv}")
            prosthesis_dofs.append(dofadr)
            prosthesis_qpos.append(qposadr)
    missing = [name for name in PROSTHESIS_JOINT_NAMES if name not in {r["name"] for r in rows}]
    if missing:
        raise KeyError(f"missing prosthesis joints: {missing}")
    ordered_dofs = []
    ordered_qpos = []
    by_name = {row["name"]: row for row in rows}
    for name in PROSTHESIS_JOINT_NAMES:
        ordered_dofs.append(int(by_name[name]["dofadr"]))
        ordered_qpos.append(int(by_name[name]["qposadr"]))
    return rows, ordered_dofs, ordered_qpos, root_dofs


def traced_env_step(env, action: np.ndarray) -> StepTrace:
    """Mirror LocoMuJoCo.step while splitting every internal physics substep."""

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    current_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=action)
    reference = env.th.get_current_traj_data(carry, np)
    processed_action, carry = env._preprocess_action(action, env._model, env._data, carry)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    qacc_rows = []
    ctrl_rows = []
    force_rows = []
    actuator_rows = []
    passive_rows = []
    constraint_rows = []
    ncon_rows = []
    for _intermediate in range(int(env._n_intermediate_steps)):
        ctrl_action, carry = env._compute_action(processed_action, env._model, env._data, carry)
        env._data.ctrl[env._action_indices] = np.asarray(ctrl_action).reshape(-1)
        for _substep in range(int(env._n_substeps)):
            # Forward computes actuator, passive, constraint, and acceleration at
            # the exact pre-integration state for this physics substep.
            mujoco.mj_forward(env._model, env._data)
            qacc_rows.append(np.asarray(env._data.qacc, dtype=np.float64).copy())
            ctrl_rows.append(np.asarray(env._data.ctrl, dtype=np.float64).copy())
            force_rows.append(np.asarray(env._data.actuator_force, dtype=np.float64).copy())
            actuator_rows.append(np.asarray(env._data.qfrc_actuator, dtype=np.float64).copy())
            passive_rows.append(np.asarray(env._data.qfrc_passive, dtype=np.float64).copy())
            constraint_rows.append(np.asarray(env._data.qfrc_constraint, dtype=np.float64).copy())
            ncon_rows.append(int(env._data.ncon))
            mujoco.mj_step(env._model, env._data, 1)

    env._data, carry = env._simulation_post_step(env._model, env._data, carry)
    observation, carry = env._create_observation(env._model, env._data, carry)
    observation, env._data, current_info, carry = env._step_finalize(
        observation, env._model, env._data, current_info, carry
    )
    current_info = env._update_info_dictionary(current_info, observation, env._data, carry)
    absorbing, carry = env._is_absorbing(observation, current_info, env._data, carry)
    reward, carry = env._reward(
        env._obs,
        action,
        observation,
        absorbing,
        current_info,
        env._model,
        env._data,
        carry,
    )
    done = env._is_done(observation, absorbing, current_info, env._data, carry)
    carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
    env._obs = observation
    env._additional_carry = carry
    return StepTrace(
        observation=np.asarray(observation),
        reward=float(np.asarray(reward).item()),
        absorbing=bool(np.asarray(absorbing).item()),
        done=bool(np.asarray(done).item()),
        info=dict(current_info),
        reference_qpos=np.asarray(reference.qpos, dtype=np.float64).copy(),
        reference_qvel=np.asarray(reference.qvel, dtype=np.float64).copy(),
        qacc=np.stack(qacc_rows),
        actuator_ctrl=np.stack(ctrl_rows),
        actuator_force=np.stack(force_rows),
        qfrc_actuator=np.stack(actuator_rows),
        qfrc_passive=np.stack(passive_rows),
        qfrc_constraint=np.stack(constraint_rows),
        contact_ncon=np.asarray(ncon_rows, dtype=np.int32),
    )


def export_fullbody_rollout(
    *,
    checkpoint_path: str,
    motion_path: str,
    output_path: str | Path,
    n_steps: int = 0,
    seed: int = 0,
    train_state_seed: int = 0,
    deterministic: bool = True,
    require_complete: bool = True,
) -> TorqueReplayDataset:
    """Run the official full-body tracker and save a replay-ready dataset."""

    env, config, agent_state, checkpoint_metadata = build_fullbody_env(checkpoint_path, motion_path)
    try:
        policy = OfficialPolicyRunner.create(
            env,
            config,
            agent_state,
            seed=seed,
            train_state_seed=train_state_seed,
            deterministic=deterministic,
        )
        obs = env.reset()
        policy_obs = policy.reset_obs(obs)
        trajectory_length = int(env.th.len_trajectory(0))
        target_steps = trajectory_length if int(n_steps) <= 0 else min(int(n_steps), trajectory_length)

        rollout_qpos = [np.asarray(env.data.qpos, dtype=np.float64).copy()]
        rollout_qvel = [np.asarray(env.data.qvel, dtype=np.float64).copy()]
        references_qpos = []
        references_qvel = []
        actions = []
        rewards = []
        absorbings = []
        dones = []
        traces: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "qacc",
                "actuator_ctrl",
                "actuator_force",
                "qfrc_actuator",
                "qfrc_passive",
                "qfrc_constraint",
                "contact_ncon",
            )
        }
        early_done_step: int | None = None
        for step in range(target_steps):
            action, _value = policy.act(policy_obs)
            trace = traced_env_step(env, action)
            actions.append(action.copy())
            references_qpos.append(trace.reference_qpos)
            references_qvel.append(trace.reference_qvel)
            rollout_qpos.append(np.asarray(env.data.qpos, dtype=np.float64).copy())
            rollout_qvel.append(np.asarray(env.data.qvel, dtype=np.float64).copy())
            rewards.append(trace.reward)
            absorbings.append(trace.absorbing)
            dones.append(trace.done)
            for name in traces:
                traces[name].append(np.asarray(getattr(trace, name)).copy())
            policy_obs = policy.update_obs(trace.observation)
            if trace.done and step + 1 < target_steps:
                early_done_step = step
                break

        joint_rows, prosthesis_dofs, prosthesis_qpos, root_dofs = _joint_metadata(env.model)
        actual_steps = len(actions)
        completed_requested = actual_steps == target_steps
        completed_reference = actual_steps == trajectory_length
        finite = all(
            np.all(np.isfinite(np.asarray(values)))
            for values in (rollout_qpos, rollout_qvel, actions, traces["qfrc_actuator"])
        )
        qualified = completed_reference and early_done_step is None and finite and not any(absorbings[:-1])
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata": str(checkpoint_metadata),
            "seed": int(seed),
            "train_state_seed": int(train_state_seed),
            "deterministic": bool(deterministic),
            "trajectory_length": trajectory_length,
            "requested_steps": target_steps,
            "actual_steps": actual_steps,
            "completed_requested": completed_requested,
            "completed_reference": completed_reference,
            "early_done_step": early_done_step,
            "qualified_full_motion": qualified,
            "finite": finite,
            "root_height_min": float(np.min(np.asarray(rollout_qpos)[:, 2])),
            "root_up_min": float(min(root_up_z(q) for q in rollout_qpos)),
            "nq": int(env.model.nq),
            "nv": int(env.model.nv),
            "nu": int(env.model.nu),
            "joint_metadata": joint_rows,
            "joint_names": _object_names(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.model.njnt),
            "actuator_names": _object_names(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, env.model.nu),
            "prosthesis_joint_names": list(PROSTHESIS_JOINT_NAMES),
            "prosthesis_dof_indices": prosthesis_dofs,
            "prosthesis_qpos_indices": prosthesis_qpos,
            "root_dof_indices": root_dofs,
        }
        dataset = TorqueReplayDataset(
            motion_path=str(motion_path),
            dt_control=float(env.dt),
            dt_physics=float(env.model.opt.timestep),
            reference_qpos=np.stack(references_qpos),
            reference_qvel=np.stack(references_qvel),
            rollout_qpos=np.stack(rollout_qpos),
            rollout_qvel=np.stack(rollout_qvel),
            rollout_qacc=np.stack(traces["qacc"]),
            policy_action=np.stack(actions),
            actuator_ctrl=np.stack(traces["actuator_ctrl"]),
            actuator_force=np.stack(traces["actuator_force"]),
            qfrc_actuator=np.stack(traces["qfrc_actuator"]),
            qfrc_passive=np.stack(traces["qfrc_passive"]),
            qfrc_constraint=np.stack(traces["qfrc_constraint"]),
            contact_ncon=np.stack(traces["contact_ncon"]),
            reward=np.asarray(rewards, dtype=np.float32),
            absorbing=np.asarray(absorbings, dtype=np.bool_),
            done=np.asarray(dones, dtype=np.bool_),
            metadata=metadata,
        )
        if require_complete and not qualified:
            target = Path(output_path)
            rejected = target.with_name(f"{target.stem}.rejected{target.suffix or '.npz'}")
            dataset.save(rejected)
            raise RuntimeError(
                "full-motion rollout did not qualify: "
                f"actual={actual_steps}/{trajectory_length}, early_done={early_done_step}, finite={finite}; "
                f"audit data saved to {rejected}"
            )
        dataset.save(output_path)
        return dataset
    finally:
        env.stop()
