"""Teacher rollout collection for prosthesis distillation."""

from __future__ import annotations

import os
from pathlib import Path
import re

import mujoco
import numpy as np

from musclemimic.distill.mapping import DistillMapping, validate_qfrc_tau, rollout_npz_invalid_reason, rollout_npz_is_valid
from musclemimic.distill.policy import PolicyRunner


def safe_motion_filename(motion_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", motion_path).strip("_") + ".npz"


def actuator_ids_for_names(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids = []
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(name))
        if aid < 0:
            raise KeyError(f"Model missing actuator {name!r}")
        ids.append(int(aid))
    return np.asarray(ids, dtype=np.int32)


def contact_load_by_bodies(model: mujoco.MjModel, data, body_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    left = 0.0
    right = 0.0
    prosthesis_set = set(int(x) for x in body_ids)
    force6 = np.zeros(6, dtype=np.float64)
    for cid in range(int(data.ncon)):
        contact = data.contact[cid]
        b1 = int(model.geom_bodyid[int(contact.geom1)])
        b2 = int(model.geom_bodyid[int(contact.geom2)])
        mujoco.mj_contactForce(model, data, cid, force6)
        normal = max(float(force6[0]), 0.0)
        if b1 in prosthesis_set or b2 in prosthesis_set:
            left += normal
        else:
            right += normal
    return np.asarray([left, right], dtype=np.float32), np.asarray([left > 1e-6, right > 1e-6], dtype=np.float32)


def sync_student_to_teacher_state(teacher_env, student_env) -> None:
    """Align prosthesis env kinematics to the teacher/reference state without stepping it."""
    student_env.data.qpos[:] = np.asarray(teacher_env.data.qpos, dtype=np.float64)
    student_env.data.qvel[:] = np.asarray(teacher_env.data.qvel, dtype=np.float64)
    if student_env.data.act.shape == teacher_env.data.act.shape:
        student_env.data.act[:] = np.asarray(teacher_env.data.act, dtype=np.float64)
    student_env.data.ctrl[:] = 0.0
    if hasattr(student_env, "_zero_disabled_muscles_np"):
        student_env._zero_disabled_muscles_np(student_env.data)
    mujoco.mj_forward(student_env.model, student_env.data)


def make_current_observation(env) -> np.ndarray:
    obs, carry = env._create_observation(env.model, env.data, env._additional_carry)
    env._additional_carry = carry
    env._obs = obs
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def reference_pd_tau(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    q_ref_next: np.ndarray,
    qd_ref_next: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    limits: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(kp, dtype=np.float32) * (np.asarray(q_ref_next, dtype=np.float32) - np.asarray(q, dtype=np.float32))
    raw += np.asarray(kd, dtype=np.float32) * (np.asarray(qd_ref_next, dtype=np.float32) - np.asarray(qd, dtype=np.float32))
    return np.clip(raw, -limits, limits).astype(np.float32)


def remaining_muscle_qfrc(
    *,
    student_env,
    target_remaining: np.ndarray,
    mapping: DistillMapping,
) -> np.ndarray:
    """Apply remaining muscle labels in prosthesis env with prosthesis motors zeroed."""
    action = np.zeros(mapping.student_action_dim, dtype=np.float32)
    action[np.asarray(mapping.student_remaining_action_indices, dtype=np.int32)] = np.asarray(
        target_remaining, dtype=np.float32
    )
    processed_action, carry = student_env._preprocess_action(
        action, student_env.model, student_env.data, student_env._additional_carry
    )
    ctrl_action, carry = student_env._compute_action(processed_action, student_env.model, student_env.data, carry)
    student_env._additional_carry = carry
    student_env.data.ctrl[:] = 0.0
    student_env.data.ctrl[np.asarray(student_env._action_indices, dtype=np.int32)] = np.asarray(
        ctrl_action, dtype=np.float64
    ).reshape(-1)
    if hasattr(student_env, "_zero_disabled_muscles_np"):
        student_env._zero_disabled_muscles_np(student_env.data)
    # Prosthesis motors must be zero here; we want the missing torque they need to supply.
    student_env.data.ctrl[np.asarray(mapping.prosthesis_actuator_ids, dtype=np.int32)] = 0.0
    mujoco.mj_forward(student_env.model, student_env.data)
    return np.asarray(student_env.data.qfrc_actuator, dtype=np.float32).copy()


def executed_remaining_muscle_ctrl(
    *,
    student_env,
    target_remaining: np.ndarray,
    mapping: DistillMapping,
) -> np.ndarray:
    """Return actuator controls actually sent to the remaining muscles."""
    action = np.zeros(mapping.student_action_dim, dtype=np.float32)
    action[np.asarray(mapping.student_remaining_action_indices, dtype=np.int32)] = np.asarray(
        target_remaining, dtype=np.float32
    )
    processed_action, carry = student_env._preprocess_action(
        action, student_env.model, student_env.data, student_env._additional_carry
    )
    ctrl_action, carry = student_env._compute_action(processed_action, student_env.model, student_env.data, carry)
    student_env._additional_carry = carry
    ctrl_action = np.asarray(ctrl_action, dtype=np.float32).reshape(-1)
    return ctrl_action[np.asarray(mapping.student_remaining_action_indices, dtype=np.int32)].copy()


def remaining_muscle_force_for_action(
    *,
    student_env,
    target_remaining: np.ndarray,
    mapping: DistillMapping,
) -> np.ndarray:
    """Estimate remaining-muscle actuator forces for an action at the current student state."""
    qpos = np.asarray(student_env.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(student_env.data.qvel, dtype=np.float64).copy()
    act = np.asarray(student_env.data.act, dtype=np.float64).copy()
    ctrl = np.asarray(student_env.data.ctrl, dtype=np.float64).copy()
    carry = student_env._additional_carry
    try:
        action = np.zeros(mapping.student_action_dim, dtype=np.float32)
        action[np.asarray(mapping.student_remaining_action_indices, dtype=np.int32)] = np.asarray(
            target_remaining, dtype=np.float32
        )
        processed_action, new_carry = student_env._preprocess_action(
            action, student_env.model, student_env.data, student_env._additional_carry
        )
        ctrl_action, new_carry = student_env._compute_action(
            processed_action, student_env.model, student_env.data, new_carry
        )
        student_env._additional_carry = new_carry
        student_env.data.ctrl[:] = 0.0
        student_env.data.ctrl[np.asarray(student_env._action_indices, dtype=np.int32)] = np.asarray(
            ctrl_action, dtype=np.float64
        ).reshape(-1)
        student_env.data.ctrl[np.asarray(mapping.prosthesis_actuator_ids, dtype=np.int32)] = 0.0
        if hasattr(student_env, "_zero_disabled_muscles_np"):
            student_env._zero_disabled_muscles_np(student_env.data)
        mujoco.mj_forward(student_env.model, student_env.data)
        student_remaining_actuator_ids = actuator_ids_for_names(student_env.model, mapping.remaining_muscle_names)
        return np.asarray(student_env.data.actuator_force[student_remaining_actuator_ids], dtype=np.float32).copy()
    finally:
        student_env.data.qpos[:] = qpos
        student_env.data.qvel[:] = qvel
        student_env.data.act[:] = act
        student_env.data.ctrl[:] = ctrl
        student_env._additional_carry = carry
        mujoco.mj_forward(student_env.model, student_env.data)


def inverse_dynamics_prosthesis_tau(
    *,
    student_env,
    target_remaining: np.ndarray,
    mapping: DistillMapping,
    qacc_des: np.ndarray,
    limits: np.ndarray,
) -> np.ndarray:
    """Torque required by prosthesis motors in the prosthesis model for desired acceleration."""
    remaining_qfrc = remaining_muscle_qfrc(
        student_env=student_env,
        target_remaining=target_remaining,
        mapping=mapping,
    )
    student_env.data.qacc[:] = np.asarray(qacc_des, dtype=np.float64)
    mujoco.mj_inverse(student_env.model, student_env.data)
    required_total = np.asarray(student_env.data.qfrc_inverse, dtype=np.float32)
    dofs = np.asarray(mapping.prosthesis_dof_ids, dtype=np.int32)
    raw = required_total[dofs] - remaining_qfrc[dofs]
    return raw.astype(np.float32)


def compute_split_action_labels(
    *,
    teacher_env,
    student_env,
    teacher_action: np.ndarray,
    mapping: DistillMapping,
    label_mode: str,
    limits: np.ndarray,
    pd_kp: np.ndarray,
    pd_kd: np.ndarray,
    clip_tau: bool,
    qvel_full: np.ndarray | None = None,
    q_current: np.ndarray | None = None,
    qd_current: np.ndarray | None = None,
    lowpass_alpha: float | None = None,
    prev_tau: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return remaining/prosthesis labels and updated prev_tau for one timestep."""
    teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
    target_remaining = teacher_action[np.asarray(mapping.teacher_remaining_action_indices, dtype=np.int32)]

    q = (
        np.asarray(q_current, dtype=np.float32)
        if q_current is not None
        else np.asarray(teacher_env.data.qpos[np.asarray(mapping.prosthesis_qpos_indices)], dtype=np.float32)
    )
    qd = (
        np.asarray(qd_current, dtype=np.float32)
        if qd_current is not None
        else np.asarray(teacher_env.data.qvel[np.asarray(mapping.prosthesis_qvel_indices)], dtype=np.float32)
    )
    q_ref_next = np.asarray(teacher_env.data.qpos[np.asarray(mapping.prosthesis_qpos_indices)], dtype=np.float32)
    qd_ref_next = np.asarray(teacher_env.data.qvel[np.asarray(mapping.prosthesis_qvel_indices)], dtype=np.float32)
    tau_pd = reference_pd_tau(
        q=q,
        qd=qd,
        q_ref_next=q_ref_next,
        qd_ref_next=qd_ref_next,
        kp=pd_kp,
        kd=pd_kd,
        limits=np.full_like(limits, np.inf),
    )

    if label_mode == "teacher_qfrc":
        raw_tau = validate_qfrc_tau(teacher_env.model, teacher_env.data, mapping.prosthesis_dof_ids)
    elif label_mode == "reference_pd":
        raw_tau = tau_pd.copy()
    elif label_mode == "inverse_dynamics":
        qvel = qvel_full if qvel_full is not None else np.asarray(teacher_env.data.qvel, dtype=np.float32).copy()
        dt = float(getattr(teacher_env, "dt", 0.01))
        qacc_des = (np.asarray(teacher_env.data.qvel, dtype=np.float32) - qvel) / max(dt, 1e-6)
        raw_tau = inverse_dynamics_prosthesis_tau(
            student_env=student_env,
            target_remaining=target_remaining,
            mapping=mapping,
            qacc_des=qacc_des,
            limits=limits,
        )
    else:
        raise ValueError(f"Unknown label_mode={label_mode!r}")

    if lowpass_alpha is not None and prev_tau is not None:
        raw_tau = float(lowpass_alpha) * raw_tau + (1.0 - float(lowpass_alpha)) * prev_tau
    new_prev_tau = raw_tau.copy()
    tau_required = raw_tau.astype(np.float32)
    tau_residual = (tau_required - tau_pd).astype(np.float32)
    if getattr(student_env, "prosthesis_action_type", "torque") == "pd_residual_torque":
        residual_cmd = np.clip(tau_residual, -limits, limits) if clip_tau else tau_residual
        target_prosthesis_action = np.clip(residual_cmd / limits, -1.0, 1.0).astype(np.float32)
        clipped_tau = np.clip(tau_pd + residual_cmd, -limits, limits) if clip_tau else tau_pd + residual_cmd
    else:
        clipped_tau = np.clip(tau_required, -limits, limits) if clip_tau else tau_required
        target_prosthesis_action = np.clip(clipped_tau / limits, -1.0, 1.0).astype(np.float32)
    return (
        target_remaining.astype(np.float32),
        target_prosthesis_action,
        clipped_tau.astype(np.float32),
        tau_pd.astype(np.float32),
        new_prev_tau,
    )


def rollout_split_action_dagger(
    *,
    teacher_env,
    student_env,
    teacher_policy: PolicyRunner,
    student_policy: PolicyRunner,
    mapping: DistillMapping,
    motion_path: str,
    output_dir: str | Path,
    n_steps: int | None = None,
    teacher_action_prob: float = 0.0,
    dagger_round: int | None = None,
    dagger_beta: float | None = None,
    stop_on_done: bool = True,
    seed: int = 0,
    label_mode: str = "inverse_dynamics",
    lowpass_alpha: float | None = None,
    clip_tau: bool = True,
    reference_pd_kp: tuple[float, float, float, float] = (240.0, 180.0, 60.0, 40.0),
    reference_pd_kd: tuple[float, float, float, float] = (24.0, 18.0, 6.0, 4.0),
) -> dict:
    """Roll out split policy in prosthesis env; label visited states with teacher split targets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    limits = np.asarray(mapping.prosthesis_torque_limits, dtype=np.float32)
    pd_kp = np.asarray(reference_pd_kp, dtype=np.float32)
    pd_kd = np.asarray(reference_pd_kd, dtype=np.float32)
    max_steps = int(
        n_steps or getattr(student_env, "th", None).len_trajectory(0) if getattr(student_env, "th", None) else 1000
    )

    obs_teacher = teacher_env.reset()
    obs_student = student_env.reset()
    obs_teacher_policy = teacher_policy.reset_obs(obs_teacher)
    obs_student_policy = student_policy.reset_obs(obs_student)
    teacher_remaining_actuator_ids = actuator_ids_for_names(teacher_env.model, mapping.remaining_muscle_names)

    rows: dict[str, list] = {
        "obs_student": [],
        "target_remaining_muscle_action": [],
        "target_remaining_muscle_force": [],
        "student_remaining_muscle_force_from_teacher_action": [],
        "target_prosthesis_action": [],
        "target_prosthesis_tau": [],
        "teacher_prosthesis_qfrc_tau": [],
        "tau_pd": [],
        "student_action": [],
        "teacher_split_action": [],
        "rollout_action": [],
        "used_teacher_action": [],
        "root_state": [],
        "done": [],
    }
    prev_tau: np.ndarray | None = None
    total_return = 0.0
    steps = 0
    done = False

    for _step in range(max_steps):
        q_current = np.asarray(teacher_env.data.qpos[np.asarray(mapping.prosthesis_qpos_indices)], dtype=np.float32)
        qd_current = np.asarray(teacher_env.data.qvel[np.asarray(mapping.prosthesis_qvel_indices)], dtype=np.float32)
        qvel_full = np.asarray(teacher_env.data.qvel, dtype=np.float32).copy()
        teacher_action, _teacher_value = teacher_policy.act(obs_teacher_policy)

        obs_teacher_next, _reward_t, _abs_t, teacher_done, _info_t = teacher_env.step(
            np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        )
        teacher_remaining_force = np.asarray(
            teacher_env.data.actuator_force[teacher_remaining_actuator_ids],
            dtype=np.float32,
        ).copy()
        teacher_qfrc_tau = validate_qfrc_tau(teacher_env.model, teacher_env.data, mapping.prosthesis_dof_ids)
        target_remaining, target_prosthesis, clipped_tau, tau_pd, prev_tau = compute_split_action_labels(
            teacher_env=teacher_env,
            student_env=student_env,
            teacher_action=teacher_action,
            mapping=mapping,
            label_mode=label_mode,
            limits=limits,
            pd_kp=pd_kp,
            pd_kd=pd_kd,
            clip_tau=clip_tau,
            qvel_full=qvel_full,
            q_current=q_current,
            qd_current=qd_current,
            lowpass_alpha=lowpass_alpha,
            prev_tau=prev_tau,
        )
        student_force_from_teacher_action = remaining_muscle_force_for_action(
            student_env=student_env,
            target_remaining=target_remaining,
            mapping=mapping,
        )
        teacher_split_action = np.concatenate([target_remaining, target_prosthesis], dtype=np.float32)

        student_action, _student_value = student_policy.act(obs_student_policy)
        student_action = np.asarray(student_action, dtype=np.float32).reshape(-1)
        if student_action.shape[0] != mapping.student_action_dim:
            raise ValueError(
                f"Student action dim mismatch: got={student_action.shape[0]}, expected={mapping.student_action_dim}"
            )
        use_teacher = bool(rng.random() < float(teacher_action_prob))
        rollout_action = teacher_split_action if use_teacher else student_action

        rows["obs_student"].append(np.asarray(obs_student, dtype=np.float32).reshape(-1))
        rows["target_remaining_muscle_action"].append(target_remaining)
        rows["target_remaining_muscle_force"].append(teacher_remaining_force)
        rows["student_remaining_muscle_force_from_teacher_action"].append(student_force_from_teacher_action)
        rows["target_prosthesis_action"].append(target_prosthesis)
        rows["target_prosthesis_tau"].append(clipped_tau)
        rows["teacher_prosthesis_qfrc_tau"].append(teacher_qfrc_tau.astype(np.float32))
        rows["tau_pd"].append(tau_pd.astype(np.float32))
        rows["student_action"].append(student_action)
        rows["teacher_split_action"].append(teacher_split_action)
        rows["rollout_action"].append(np.asarray(rollout_action, dtype=np.float32))
        rows["used_teacher_action"].append(np.asarray(use_teacher, dtype=np.bool_))
        rows["root_state"].append(np.concatenate([student_env.data.qpos[:7], student_env.data.qvel[:6]]).astype(np.float32))

        obs_student_next, reward, _absorbing, done, _info = student_env.step(rollout_action)
        total_return += float(np.asarray(reward).item())
        rows["done"].append(np.asarray(bool(done), dtype=np.bool_))

        obs_teacher = obs_teacher_next
        obs_student = obs_student_next
        obs_teacher_policy = teacher_policy.update_obs(obs_teacher_next)
        obs_student_policy = student_policy.update_obs(obs_student_next)
        steps += 1
        if (done or teacher_done) and stop_on_done:
            break

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["motion_path"] = np.asarray(motion_path)
    arrays["torque_limits"] = limits
    arrays["remaining_muscle_names"] = np.asarray(mapping.remaining_muscle_names)
    arrays["prosthesis_joint_names"] = np.asarray(mapping.prosthesis_joint_names)
    arrays["label_mode"] = np.asarray(label_mode)
    arrays["dagger_teacher_action_prob"] = np.asarray(float(teacher_action_prob), dtype=np.float32)
    if dagger_beta is not None:
        arrays["dagger_beta"] = np.asarray(float(dagger_beta), dtype=np.float32)
    if dagger_round is not None:
        arrays["dagger_round"] = np.asarray(int(dagger_round), dtype=np.int32)
    arrays["dagger_student_checkpoint"] = np.asarray("")
    out_path = output_dir / safe_motion_filename(motion_path)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    np.savez_compressed(str(tmp_path), **arrays)
    if not rollout_npz_is_valid(tmp_path):
        reason = rollout_npz_invalid_reason(tmp_path)
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to write valid DAgger npz for {motion_path}: {reason}")
    os.replace(tmp_path, out_path)

    student_label_mse = float(
        np.mean((arrays["student_action"] - np.concatenate(
            [arrays["target_remaining_muscle_action"], arrays["target_prosthesis_action"]], axis=-1
        )) ** 2)
    ) if steps else 0.0
    tau = arrays["target_prosthesis_tau"]
    action = arrays["target_prosthesis_action"]
    return {
        "path": str(out_path),
        "motion_path": motion_path,
        "frames": int(steps),
        "return": float(total_return),
        "done": bool(done),
        "student_label_mse": student_label_mse,
        "teacher_action_prob": float(teacher_action_prob),
        "tau_mean": tau.mean(axis=0).tolist() if tau.size else [0.0] * 4,
        "clip_ratio": float(np.mean(np.abs(action) >= 0.999)) if action.size else 0.0,
    }


def rollout_motion(
    *,
    teacher_env,
    student_env,
    teacher_policy: PolicyRunner,
    mapping: DistillMapping,
    motion_path: str,
    output_dir: str | Path,
    n_steps: int | None = None,
    lowpass_alpha: float | None = None,
    clip_tau: bool = True,
    label_mode: str = "reference_pd",
    pin_student_state: bool = True,
    reference_pd_kp: tuple[float, float, float, float] = (240.0, 180.0, 60.0, 40.0),
    reference_pd_kd: tuple[float, float, float, float] = (24.0, 18.0, 6.0, 4.0),
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_teacher = teacher_env.reset()
    obs_student = student_env.reset()
    obs_teacher_policy = teacher_policy.reset_obs(obs_teacher)
    teacher_remaining_actuator_ids = actuator_ids_for_names(teacher_env.model, mapping.remaining_muscle_names)
    limits = np.asarray(mapping.prosthesis_torque_limits, dtype=np.float32)
    pd_kp = np.asarray(reference_pd_kp, dtype=np.float32)
    pd_kd = np.asarray(reference_pd_kd, dtype=np.float32)
    max_steps = int(n_steps or getattr(teacher_env, "th", None).len_trajectory(0) if getattr(teacher_env, "th", None) else 1000)
    if label_mode not in {"teacher_qfrc", "reference_pd", "inverse_dynamics"}:
        raise ValueError(f"Unknown label_mode={label_mode!r}")

    rows: dict[str, list] = {
        "obs_teacher": [],
        "obs_student": [],
        "teacher_action_all_muscles": [],
        "target_remaining_muscle_action": [],
        "target_remaining_muscle_force": [],
        "student_remaining_muscle_force_from_teacher_action": [],
        "teacher_prosthesis_qfrc_tau": [],
        "executed_remaining_muscle_ctrl": [],
        "target_prosthesis_action": [],
        "target_prosthesis_tau": [],
        "target_prosthesis_tau_raw": [],
        "tau_required": [],
        "tau_pd": [],
        "tau_residual": [],
        "q_prosthesis": [],
        "qd_prosthesis": [],
        "grf": [],
        "foot_contact": [],
        "root_state": [],
        "pelvis_state": [],
        "phase": [],
        "done": [],
        "success": [],
    }
    prev_tau: np.ndarray | None = None
    total_return = 0.0
    steps = 0
    done = False

    for step in range(max_steps):
        if pin_student_state:
            sync_student_to_teacher_state(teacher_env, student_env)
            obs_student = make_current_observation(student_env)

        q = np.asarray(teacher_env.data.qpos[np.asarray(mapping.prosthesis_qpos_indices)], dtype=np.float32)
        qd = np.asarray(teacher_env.data.qvel[np.asarray(mapping.prosthesis_qvel_indices)], dtype=np.float32)
        qvel_full = np.asarray(teacher_env.data.qvel, dtype=np.float32).copy()
        root_state = np.concatenate([teacher_env.data.qpos[:7], teacher_env.data.qvel[:6]]).astype(np.float32)

        teacher_action, _value = teacher_policy.act(obs_teacher_policy)
        teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        if teacher_action.shape[0] != mapping.teacher_action_dim:
            raise ValueError(f"Teacher action dim mismatch: got={teacher_action.shape[0]}, expected={mapping.teacher_action_dim}")

        target_remaining = teacher_action[np.asarray(mapping.teacher_remaining_action_indices, dtype=np.int32)]
        student_force_from_teacher_action = remaining_muscle_force_for_action(
            student_env=student_env,
            target_remaining=target_remaining,
            mapping=mapping,
        )

        obs_teacher_next, reward, _absorbing, done, info = teacher_env.step(teacher_action)
        total_return += float(np.asarray(reward).item())
        teacher_remaining_force = np.asarray(
            teacher_env.data.actuator_force[teacher_remaining_actuator_ids],
            dtype=np.float32,
        ).copy()
        teacher_qfrc_tau = validate_qfrc_tau(teacher_env.model, teacher_env.data, mapping.prosthesis_dof_ids)

        q_ref_next = np.asarray(teacher_env.data.qpos[np.asarray(mapping.prosthesis_qpos_indices)], dtype=np.float32)
        qd_ref_next = np.asarray(teacher_env.data.qvel[np.asarray(mapping.prosthesis_qvel_indices)], dtype=np.float32)
        tau_pd = reference_pd_tau(
            q=q,
            qd=qd,
            q_ref_next=q_ref_next,
            qd_ref_next=qd_ref_next,
            kp=pd_kp,
            kd=pd_kd,
            limits=np.full_like(limits, np.inf),
        )
        executed_remaining = executed_remaining_muscle_ctrl(
            student_env=student_env,
            target_remaining=target_remaining,
            mapping=mapping,
        )

        if label_mode == "teacher_qfrc":
            raw_tau = validate_qfrc_tau(teacher_env.model, teacher_env.data, mapping.prosthesis_dof_ids)
        elif label_mode == "reference_pd":
            raw_tau = tau_pd.copy()
        else:
            dt = float(getattr(teacher_env, "dt", 0.01))
            qacc_des = (np.asarray(teacher_env.data.qvel, dtype=np.float32) - qvel_full) / max(dt, 1e-6)
            raw_tau = inverse_dynamics_prosthesis_tau(
                student_env=student_env,
                target_remaining=target_remaining,
                mapping=mapping,
                qacc_des=qacc_des,
                limits=limits,
            )
        if lowpass_alpha is not None and prev_tau is not None:
            raw_tau = float(lowpass_alpha) * raw_tau + (1.0 - float(lowpass_alpha)) * prev_tau
        prev_tau = raw_tau.copy()
        tau_required = raw_tau.astype(np.float32)
        tau_residual = (tau_required - tau_pd).astype(np.float32)
        if getattr(student_env, "prosthesis_action_type", "torque") == "pd_residual_torque":
            residual_cmd = np.clip(tau_residual, -limits, limits) if clip_tau else tau_residual
            target_prosthesis_action = np.clip(residual_cmd / limits, -1.0, 1.0).astype(np.float32)
            clipped_tau = np.clip(tau_pd + residual_cmd, -limits, limits) if clip_tau else tau_pd + residual_cmd
        else:
            clipped_tau = np.clip(tau_required, -limits, limits) if clip_tau else tau_required
            target_prosthesis_action = np.clip(clipped_tau / limits, -1.0, 1.0).astype(np.float32)
        student_action = np.concatenate([target_remaining, target_prosthesis_action], dtype=np.float32)

        grf, foot_contact = contact_load_by_bodies(teacher_env.model, teacher_env.data, mapping.prosthesis_body_ids)
        traj_len = max(int(info.get("traj_len", max_steps)) if isinstance(info, dict) else max_steps, 1)
        substep = int(info.get("subtraj_step_no", step)) if isinstance(info, dict) else step

        rows["obs_teacher"].append(np.asarray(obs_teacher_policy, dtype=np.float32))
        rows["obs_student"].append(np.asarray(obs_student, dtype=np.float32).reshape(-1))
        rows["teacher_action_all_muscles"].append(teacher_action)
        rows["target_remaining_muscle_action"].append(target_remaining.astype(np.float32))
        rows["target_remaining_muscle_force"].append(teacher_remaining_force)
        rows["student_remaining_muscle_force_from_teacher_action"].append(student_force_from_teacher_action)
        rows["teacher_prosthesis_qfrc_tau"].append(teacher_qfrc_tau.astype(np.float32))
        rows["executed_remaining_muscle_ctrl"].append(executed_remaining.astype(np.float32))
        rows["target_prosthesis_action"].append(target_prosthesis_action)
        rows["target_prosthesis_tau"].append(clipped_tau.astype(np.float32))
        rows["target_prosthesis_tau_raw"].append(raw_tau.astype(np.float32))
        rows["tau_required"].append(tau_required.astype(np.float32))
        rows["tau_pd"].append(tau_pd.astype(np.float32))
        rows["tau_residual"].append(tau_residual.astype(np.float32))
        rows["q_prosthesis"].append(q)
        rows["qd_prosthesis"].append(qd)
        rows["grf"].append(grf)
        rows["foot_contact"].append(foot_contact)
        rows["root_state"].append(root_state)
        rows["pelvis_state"].append(root_state)
        rows["phase"].append(np.asarray(substep / traj_len, dtype=np.float32))
        rows["done"].append(np.asarray(bool(done), dtype=np.bool_))
        rows["success"].append(np.asarray((not done) or substep >= traj_len - 1, dtype=np.bool_))

        if pin_student_state:
            student_env._last_prosthesis_tau[:] = clipped_tau.astype(np.float32)
        else:
            obs_student_next, *_ = student_env.step(student_action)
            obs_student = obs_student_next
        obs_teacher = obs_teacher_next
        obs_teacher_policy = teacher_policy.update_obs(obs_teacher_next)
        steps += 1
        if done:
            break

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["motion_path"] = np.asarray(motion_path)
    arrays["torque_limits"] = limits
    arrays["remaining_muscle_names"] = np.asarray(mapping.remaining_muscle_names)
    arrays["prosthesis_joint_names"] = np.asarray(mapping.prosthesis_joint_names)
    arrays["label_mode"] = np.asarray(label_mode)
    arrays["pin_student_state"] = np.asarray(bool(pin_student_state))
    arrays["reference_pd_kp"] = pd_kp
    arrays["reference_pd_kd"] = pd_kd
    arrays["prosthesis_action_type"] = np.asarray(getattr(student_env, "prosthesis_action_type", "torque"))
    out_path = output_dir / safe_motion_filename(motion_path)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    np.savez_compressed(str(tmp_path), **arrays)
    if not rollout_npz_is_valid(tmp_path):
        reason = rollout_npz_invalid_reason(tmp_path)
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to write valid rollout npz for {motion_path} "
            f"(frames={steps}, tmp={tmp_path.name}, reason={reason})"
        )
    os.replace(tmp_path, out_path)

    tau = arrays["target_prosthesis_tau"]
    action = arrays["target_prosthesis_action"]
    return {
        "path": str(out_path),
        "motion_path": motion_path,
        "frames": int(steps),
        "return": float(total_return),
        "tau_mean": tau.mean(axis=0).tolist() if tau.size else [0.0] * 4,
        "tau_std": tau.std(axis=0).tolist() if tau.size else [0.0] * 4,
        "tau_min": tau.min(axis=0).tolist() if tau.size else [0.0] * 4,
        "tau_max": tau.max(axis=0).tolist() if tau.size else [0.0] * 4,
        "clip_ratio": float(np.mean(np.abs(action) >= 0.999)) if action.size else 0.0,
        "done": bool(done),
    }
