"""Closed-loop muscle-policy replay (policy actions drive MuJoCo actuators)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np


@dataclass
class MuscleTrajectory:
    motion_path: str
    dt: float
    policy_action: np.ndarray
    actuator_ctrl: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    source_controller: str
    source_checkpoint: str
    replay_mode: str = "muscle_closed_loop"
    metadata: dict[str, Any] | None = None

    @property
    def n_frames(self) -> int:
        return int(self.policy_action.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.n_frames * self.dt)


def save_muscle_trajectory(path: Path, traj: MuscleTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        motion_path=np.asarray(traj.motion_path),
        dt=np.asarray(traj.dt),
        policy_action=traj.policy_action.astype(np.float32),
        actuator_ctrl=traj.actuator_ctrl.astype(np.float32),
        qpos=traj.qpos.astype(np.float64),
        qvel=traj.qvel.astype(np.float64),
        source_controller=np.asarray(traj.source_controller),
        source_checkpoint=np.asarray(traj.source_checkpoint),
        replay_mode=np.asarray(traj.replay_mode),
        metadata=np.asarray(json.dumps(traj.metadata or {})),
    )


def load_muscle_trajectory(path: Path) -> MuscleTrajectory:
    data = np.load(path, allow_pickle=False)
    meta_raw = data["metadata"]
    meta_str = meta_raw.item() if meta_raw.ndim == 0 else str(meta_raw.reshape(-1)[0])
    metadata = json.loads(meta_str) if meta_str else {}
    return MuscleTrajectory(
        motion_path=str(data["motion_path"].item() if data["motion_path"].ndim == 0 else data["motion_path"].reshape(-1)[0]),
        dt=float(np.asarray(data["dt"]).reshape(-1)[0]),
        policy_action=np.asarray(data["policy_action"], dtype=np.float32),
        actuator_ctrl=np.asarray(data["actuator_ctrl"], dtype=np.float32),
        qpos=np.asarray(data["qpos"], dtype=np.float64),
        qvel=np.asarray(data["qvel"], dtype=np.float64),
        source_controller=str(
            data["source_controller"].item()
            if data["source_controller"].ndim == 0
            else data["source_controller"].reshape(-1)[0]
        ),
        source_checkpoint=str(
            data["source_checkpoint"].item()
            if data["source_checkpoint"].ndim == 0
            else data["source_checkpoint"].reshape(-1)[0]
        ),
        replay_mode=str(
            data["replay_mode"].item() if data["replay_mode"].ndim == 0 else data["replay_mode"].reshape(-1)[0]
        ),
        metadata=metadata,
    )


def record_muscle_closed_loop_rollout(
    *,
    env,
    policy,
    n_steps: int,
    width: int = 640,
    height: int = 480,
    fps: int | None = None,
    cam_distance: float = 6.0,
    cam_elevation: float = -20.0,
    cam_azimuth: float = 90.0,
    record_path: Path | None = None,
) -> tuple[MuscleTrajectory, float, int]:
    """Roll out a muscle policy in physics and optionally record mp4."""
    dt = float(getattr(env, "dt", 0.01))
    fps = int(fps if fps is not None else round(1.0 / max(dt, 1e-6)))

    policy_actions: list[np.ndarray] = []
    actuator_ctrls: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    done_flags: list[bool] = []

    obs = env.reset()
    obs_policy = policy.reset_obs(obs)
    episode_return = 0.0
    done_count = 0

    writer = None
    renderer = None
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.model, width=width, height=height)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = cam_distance
        cam.elevation = cam_elevation
        cam.azimuth = cam_azimuth
        writer = imageio.get_writer(str(record_path), fps=fps, quality=8)

    try:
        for _step in range(int(n_steps)):
            action, _value = policy.act(obs_policy)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            policy_actions.append(action.copy())
            qpos_rows.append(np.asarray(env.data.qpos, dtype=np.float64).copy())
            qvel_rows.append(np.asarray(env.data.qvel, dtype=np.float64).copy())

            obs, reward, absorbing, done, _info = env.step(action)
            obs_policy = policy.update_obs(obs)
            actuator_ctrls.append(np.asarray(env.data.ctrl, dtype=np.float32).copy())
            done_flags.append(bool(done))
            episode_return += float(np.asarray(reward).item())
            done_count += int(bool(done))

            if writer is not None and renderer is not None:
                cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    traj = MuscleTrajectory(
        motion_path="",
        dt=dt,
        policy_action=np.stack(policy_actions, axis=0),
        actuator_ctrl=np.stack(actuator_ctrls, axis=0),
        qpos=np.stack(qpos_rows, axis=0),
        qvel=np.stack(qvel_rows, axis=0),
        source_controller="",
        source_checkpoint="",
        metadata={
            "steps": len(policy_actions),
            "done_count": int(done_count),
            "episode_return": float(episode_return),
            "any_done": bool(any(done_flags)),
        },
    )
    return traj, episode_return, done_count


def step_env_with_policy_action_and_disabled_actuators(
    env,
    action: np.ndarray,
    disabled_actuators: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Step env with policy action while zeroing selected muscle actuators each substep."""
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    disabled = np.asarray(disabled_actuators, dtype=np.int32)
    if disabled.size:
        action_indices = np.asarray(env._action_indices, dtype=np.int32)
        disabled_positions = np.where(np.isin(action_indices, disabled))[0]
        action[disabled_positions] = 0.0

    cur_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=action)
    processed_action, carry = env._preprocess_action(action, env._model, env._data, carry)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    for _ in range(env._n_intermediate_steps):
        env._data.qfrc_applied[:] = 0.0
        ctrl_action, carry = env._compute_action(processed_action, env._model, env._data, carry)
        env._data.ctrl[env._action_indices] = ctrl_action
        if disabled.size:
            env._data.ctrl[disabled] = 0.0
            actadr = np.asarray(env._model.actuator_actadr[disabled], dtype=np.int32)
            for adr in actadr:
                if int(adr) >= 0:
                    env._data.act[int(adr)] = 0.0
        mujoco.mj_step(env._model, env._data, env._n_substeps)

    env._data, carry = env._simulation_post_step(env._model, env._data, carry)
    cur_obs, carry = env._create_observation(env._model, env._data, carry)
    cur_obs, env._data, cur_info, carry = env._step_finalize(cur_obs, env._model, env._data, cur_info, carry)
    cur_info = env._update_info_dictionary(cur_info, cur_obs, env._data, carry)
    absorbing, carry = env._is_absorbing(cur_obs, cur_info, env._data, carry)
    reward, carry = env._reward(env._obs, action, cur_obs, absorbing, cur_info, env._model, env._data, carry)
    done = env._is_done(cur_obs, absorbing, cur_info, env._data, carry)
    carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
    env._obs = cur_obs
    env._additional_carry = carry
    return np.asarray(cur_obs), float(np.asarray(reward).item()), bool(done)


def record_muscle_closed_loop_masked_rollout(
    *,
    env,
    policy,
    disabled_actuators: np.ndarray,
    n_steps: int,
    width: int = 640,
    height: int = 480,
    fps: int | None = None,
    cam_distance: float = 6.0,
    cam_elevation: float = -20.0,
    cam_azimuth: float = 90.0,
    record_path: Path | None = None,
) -> tuple[MuscleTrajectory, float, int]:
    """Roll out a muscle policy with selected actuators disabled (ctrl/act forced to zero)."""
    dt = float(getattr(env, "dt", 0.01))
    fps = int(fps if fps is not None else round(1.0 / max(dt, 1e-6)))
    disabled = np.asarray(disabled_actuators, dtype=np.int32)

    policy_actions: list[np.ndarray] = []
    actuator_ctrls: list[np.ndarray] = []
    qpos_rows: list[np.ndarray] = []
    qvel_rows: list[np.ndarray] = []
    done_flags: list[bool] = []

    obs = env.reset()
    obs_policy = policy.reset_obs(obs)
    episode_return = 0.0
    done_count = 0

    writer = None
    renderer = None
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.model, width=width, height=height)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = cam_distance
        cam.elevation = cam_elevation
        cam.azimuth = cam_azimuth
        writer = imageio.get_writer(str(record_path), fps=fps, quality=8)

    try:
        for _step in range(int(n_steps)):
            action, _value = policy.act(obs_policy)
            action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
            if disabled.size:
                action_indices = np.asarray(env._action_indices, dtype=np.int32)
                disabled_positions = np.where(np.isin(action_indices, disabled))[0]
                action[disabled_positions] = 0.0
            policy_actions.append(action)
            qpos_rows.append(np.asarray(env.data.qpos, dtype=np.float64).copy())
            qvel_rows.append(np.asarray(env.data.qvel, dtype=np.float64).copy())

            obs, reward, done = step_env_with_policy_action_and_disabled_actuators(env, action, disabled)
            obs_policy = policy.update_obs(obs)
            actuator_ctrls.append(np.asarray(env.data.ctrl, dtype=np.float32).copy())
            done_flags.append(bool(done))
            episode_return += float(reward)
            done_count += int(bool(done))

            if writer is not None and renderer is not None:
                cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    traj = MuscleTrajectory(
        motion_path="",
        dt=dt,
        policy_action=np.stack(policy_actions, axis=0),
        actuator_ctrl=np.stack(actuator_ctrls, axis=0),
        qpos=np.stack(qpos_rows, axis=0),
        qvel=np.stack(qvel_rows, axis=0),
        source_controller="",
        source_checkpoint="",
        replay_mode="muscle_closed_loop_masked",
        metadata={
            "steps": len(policy_actions),
            "done_count": int(done_count),
            "episode_return": float(episode_return),
            "any_done": bool(any(done_flags)),
            "disabled_actuator_ids": disabled.tolist(),
        },
    )
    return traj, episode_return, done_count


def record_open_loop_muscle_signal_replay(
    *,
    env,
    traj: MuscleTrajectory,
    record_path: Path,
    width: int = 640,
    height: int = 480,
    fps: int | None = None,
    cam_distance: float = 6.0,
    cam_elevation: float = -20.0,
    cam_azimuth: float = 90.0,
) -> tuple[float, int, int]:
    """Replay saved policy_action frame-by-frame without querying the policy."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    fps = int(fps if fps is not None else round(1.0 / max(float(traj.dt), 1e-6)))

    renderer = mujoco.Renderer(env.model, width=width, height=height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance
    cam.elevation = cam_elevation
    cam.azimuth = cam_azimuth

    env.reset()
    episode_return = 0.0
    done_count = 0
    n_steps = min(int(traj.n_frames), int(env.th.len_trajectory(0)) if getattr(env, "th", None) is not None else int(traj.n_frames))

    with imageio.get_writer(record_path, fps=fps, quality=8) as writer:
        for step in range(n_steps):
            action = np.asarray(traj.policy_action[step], dtype=np.float32).reshape(-1)
            _obs, reward, _absorbing, done, _info = env.step(action)
            episode_return += float(np.asarray(reward).item())
            done_count += int(bool(done))

            cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            renderer.update_scene(env.data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()
    return float(episode_return), int(done_count), int(n_steps)


def merge_disabled_actuator_ids(*groups: np.ndarray) -> np.ndarray:
    if not groups:
        return np.asarray([], dtype=np.int32)
    return np.asarray(sorted({int(x) for g in groups for x in np.asarray(g, dtype=np.int32).reshape(-1)}), dtype=np.int32)


def actuator_ids_for_muscle_names(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids: list[int] = []
    missing: list[str] = []
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            missing.append(name)
        else:
            ids.append(int(aid))
    if missing:
        raise KeyError(f"Missing disabled muscle actuators: {missing}")
    return np.asarray(sorted(set(ids)), dtype=np.int32)


def color_disabled_muscle_tendons_blue(model: mujoco.MjModel, names: tuple[str, ...]) -> int:
    changed = 0
    blue = np.asarray([0.0, 0.25, 1.0, 1.0], dtype=model.tendon_rgba.dtype)
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        if int(model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            continue
        tid = int(model.actuator_trnid[aid, 0])
        if tid < 0 or tid >= model.ntendon:
            continue
        rgba = blue.copy()
        rgba[3] = max(float(model.tendon_rgba[tid, 3]), 1.0)
        model.tendon_rgba[tid] = rgba
        changed += 1
    return changed


def step_env_with_recorded_ctrl(
    env,
    recorded_ctrl: np.ndarray,
    disabled_actuators: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Apply recorded low-level muscle ctrl with selected actuators zeroed; no external prosthesis torque."""
    full_action_for_bookkeeping = np.zeros(int(env.info.action_space.shape[0]), dtype=np.float32)
    cur_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=full_action_for_bookkeeping)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    ctrl = np.asarray(recorded_ctrl, dtype=np.float64).reshape(-1).copy()
    if ctrl.shape[0] != env.data.ctrl.shape[0]:
        raise ValueError(f"Recorded ctrl dim {ctrl.shape[0]} != env ctrl dim {env.data.ctrl.shape[0]}")
    ctrl[disabled_actuators] = 0.0

    for _ in range(env._n_intermediate_steps):
        env._data.qfrc_applied[:] = 0.0
        env._data.ctrl[:] = ctrl
        if disabled_actuators.size:
            env._data.ctrl[disabled_actuators] = 0.0
            actadr = np.asarray(env._model.actuator_actadr[disabled_actuators], dtype=np.int32)
            for adr in actadr:
                if int(adr) >= 0:
                    env._data.act[int(adr)] = 0.0
        mujoco.mj_step(env._model, env._data, env._n_substeps)

    env._data, carry = env._simulation_post_step(env._model, env._data, carry)
    cur_obs, carry = env._create_observation(env._model, env._data, carry)
    cur_obs, env._data, cur_info, carry = env._step_finalize(cur_obs, env._model, env._data, cur_info, carry)
    cur_info = env._update_info_dictionary(cur_info, cur_obs, env._data, carry)
    absorbing, carry = env._is_absorbing(cur_obs, cur_info, env._data, carry)
    reward, carry = env._reward(env._obs, full_action_for_bookkeeping, cur_obs, absorbing, cur_info, env._model, env._data, carry)
    done = env._is_done(cur_obs, absorbing, cur_info, env._data, carry)
    carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
    env._obs = cur_obs
    env._additional_carry = carry
    return np.asarray(cur_obs), float(np.asarray(reward).item()), bool(done)


def pin_non_left_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    replay_qpos: np.ndarray,
    replay_qvel: np.ndarray,
    non_left_qpos: np.ndarray,
    non_left_qvel: np.ndarray,
) -> None:
    data.qpos[non_left_qpos] = replay_qpos[non_left_qpos]
    data.qvel[non_left_qvel] = replay_qvel[non_left_qvel]
    mujoco.mj_forward(model, data)


def pin_left_leg_joint_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    replay_qpos: np.ndarray,
    replay_qvel: np.ndarray,
    left_qpos: np.ndarray,
    left_qvel: np.ndarray,
) -> None:
    """Hard-lock left-leg prosthesis qpos/qvel to the replayed joint trajectory frame."""
    data.qpos[left_qpos] = replay_qpos[left_qpos]
    data.qvel[left_qvel] = replay_qvel[left_qvel]
    mujoco.mj_forward(model, data)


def pin_scaffold_hybrid_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    replay_qpos: np.ndarray,
    replay_qvel: np.ndarray,
    non_left_qpos: np.ndarray,
    non_left_qvel: np.ndarray,
    left_qpos: np.ndarray,
    left_qvel: np.ndarray,
) -> None:
    """Pin non-left scaffold + left prosthesis joints (mirrors joint FSM hybrid, inverted DOF split)."""
    pin_non_left_joints(model, data, replay_qpos, replay_qvel, non_left_qpos, non_left_qvel)
    pin_left_leg_joint_state(model, data, replay_qpos, replay_qvel, left_qpos, left_qvel)


def aligned_joint_muscle_replay_steps(joint_frames: int, muscle_frames: int) -> int:
    """Joint npz stores post-step states; muscle npz stores pre-step qpos + per-step ctrl."""
    if muscle_frames < 1:
        raise ValueError("muscle trajectory must contain at least one frame")
    if joint_frames < 1:
        raise ValueError("joint trajectory must contain at least one frame")
    # muscle ctrl[i] spans [i*dt, (i+1)*dt); joint qpos[i] is the post-step state for the same interval.
    return min(int(joint_frames), int(muscle_frames) - 1)


def aligned_hybrid_replay_steps_from_muscle_only(muscle_frames: int) -> int:
    """Single-rollout muscle npz: qpos[i] is pre-step i, qpos[i+1] is post-step i."""
    if muscle_frames < 2:
        raise ValueError("muscle trajectory must contain at least two frames")
    return int(muscle_frames) - 1


def muscle_self_aligned_hybrid_frames(
    step: int,
    muscle_traj: MuscleTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre/post frames from one official muscle rollout (self-consistent)."""
    pre_qpos = np.asarray(muscle_traj.qpos[step], dtype=np.float64)
    pre_qvel = np.asarray(muscle_traj.qvel[step], dtype=np.float64)
    post_qpos = np.asarray(muscle_traj.qpos[step + 1], dtype=np.float64)
    post_qvel = np.asarray(muscle_traj.qvel[step + 1], dtype=np.float64)
    return pre_qpos, pre_qvel, post_qpos, post_qvel


def step_with_scaffold_hybrid_replay(
    env,
    recorded_ctrl: np.ndarray,
    disabled_actuators: np.ndarray,
    pre_replay_qpos: np.ndarray,
    pre_replay_qvel: np.ndarray,
    post_replay_qpos: np.ndarray,
    post_replay_qvel: np.ndarray,
    non_left_qpos: np.ndarray,
    non_left_qvel: np.ndarray,
    left_qpos_indices: np.ndarray,
    left_qvel_indices: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Pin non-left scaffold + left prosthesis; replay masked muscle ctrl between pre/post frames."""
    full_action_for_bookkeeping = np.zeros(int(env.info.action_space.shape[0]), dtype=np.float32)
    cur_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=full_action_for_bookkeeping)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    ctrl = np.asarray(recorded_ctrl, dtype=np.float64).reshape(-1).copy()
    if ctrl.shape[0] != env.data.ctrl.shape[0]:
        raise ValueError(f"Recorded ctrl dim {ctrl.shape[0]} != env ctrl dim {env.data.ctrl.shape[0]}")
    ctrl[disabled_actuators] = 0.0

    pin_scaffold_hybrid_state(
        env._model,
        env._data,
        pre_replay_qpos,
        pre_replay_qvel,
        non_left_qpos,
        non_left_qvel,
        left_qpos_indices,
        left_qvel_indices,
    )

    for _ in range(env._n_intermediate_steps):
        env._data.qfrc_applied[:] = 0.0
        env._data.ctrl[:] = ctrl
        if disabled_actuators.size:
            env._data.ctrl[disabled_actuators] = 0.0
            actadr = np.asarray(env._model.actuator_actadr[disabled_actuators], dtype=np.int32)
            for adr in actadr:
                if int(adr) >= 0:
                    env._data.act[int(adr)] = 0.0
        mujoco.mj_step(env._model, env._data, env._n_substeps)
        env._data.qfrc_applied[:] = 0.0

    pin_scaffold_hybrid_state(
        env._model,
        env._data,
        post_replay_qpos,
        post_replay_qvel,
        non_left_qpos,
        non_left_qvel,
        left_qpos_indices,
        left_qvel_indices,
    )

    env._data, carry = env._simulation_post_step(env._model, env._data, carry)
    cur_obs, carry = env._create_observation(env._model, env._data, carry)
    cur_obs, env._data, cur_info, carry = env._step_finalize(cur_obs, env._model, env._data, cur_info, carry)
    cur_info = env._update_info_dictionary(cur_info, cur_obs, env._data, carry)
    absorbing, carry = env._is_absorbing(cur_obs, cur_info, env._data, carry)
    reward, carry = env._reward(env._obs, full_action_for_bookkeeping, cur_obs, absorbing, cur_info, env._model, env._data, carry)
    done = env._is_done(cur_obs, absorbing, cur_info, env._data, carry)
    carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
    env._obs = cur_obs
    env._additional_carry = carry
    return np.asarray(cur_obs), float(np.asarray(reward).item()), bool(done)


def step_with_recorded_ctrl_and_left_joint_lock(
    env,
    recorded_ctrl: np.ndarray,
    disabled_actuators: np.ndarray,
    pre_left_qpos: np.ndarray,
    pre_left_qvel: np.ndarray,
    post_left_qpos: np.ndarray,
    post_left_qvel: np.ndarray,
    left_qpos_indices: np.ndarray,
    left_qvel_indices: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Replay muscle ctrl; hard-lock only left prosthesis DoF (re-applied every physics substep)."""
    full_action_for_bookkeeping = np.zeros(int(env.info.action_space.shape[0]), dtype=np.float32)
    cur_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=full_action_for_bookkeeping)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    ctrl = np.asarray(recorded_ctrl, dtype=np.float64).reshape(-1).copy()
    if ctrl.shape[0] != env.data.ctrl.shape[0]:
        raise ValueError(f"Recorded ctrl dim {ctrl.shape[0]} != env ctrl dim {env.data.ctrl.shape[0]}")
    ctrl[disabled_actuators] = 0.0

    pre_left_qpos = np.asarray(pre_left_qpos, dtype=np.float64).reshape(-1)
    pre_left_qvel = np.asarray(pre_left_qvel, dtype=np.float64).reshape(-1)
    post_left_qpos = np.asarray(post_left_qpos, dtype=np.float64).reshape(-1)
    post_left_qvel = np.asarray(post_left_qvel, dtype=np.float64).reshape(-1)

    env._data.qpos[left_qpos_indices] = pre_left_qpos
    env._data.qvel[left_qvel_indices] = pre_left_qvel
    mujoco.mj_forward(env._model, env._data)

    n_substeps_total = int(env._n_intermediate_steps) * int(env._n_substeps)
    substep_idx = 0
    for _ in range(env._n_intermediate_steps):
        env._data.qfrc_applied[:] = 0.0
        env._data.ctrl[:] = ctrl
        if disabled_actuators.size:
            env._data.ctrl[disabled_actuators] = 0.0
            actadr = np.asarray(env._model.actuator_actadr[disabled_actuators], dtype=np.int32)
            for adr in actadr:
                if int(adr) >= 0:
                    env._data.act[int(adr)] = 0.0
        for _ in range(env._n_substeps):
            mujoco.mj_step(env._model, env._data, 1)
            substep_idx += 1
            alpha = float(substep_idx) / float(max(n_substeps_total, 1))
            env._data.qpos[left_qpos_indices] = (1.0 - alpha) * pre_left_qpos + alpha * post_left_qpos
            env._data.qvel[left_qvel_indices] = (1.0 - alpha) * pre_left_qvel + alpha * post_left_qvel
            mujoco.mj_forward(env._model, env._data)
        env._data.qfrc_applied[:] = 0.0

    env._data.qpos[left_qpos_indices] = post_left_qpos
    env._data.qvel[left_qvel_indices] = post_left_qvel
    mujoco.mj_forward(env._model, env._data)

    env._data, carry = env._simulation_post_step(env._model, env._data, carry)
    cur_obs, carry = env._create_observation(env._model, env._data, carry)
    cur_obs, env._data, cur_info, carry = env._step_finalize(cur_obs, env._model, env._data, cur_info, carry)
    cur_info = env._update_info_dictionary(cur_info, cur_obs, env._data, carry)
    absorbing, carry = env._is_absorbing(cur_obs, cur_info, env._data, carry)
    reward, carry = env._reward(env._obs, full_action_for_bookkeeping, cur_obs, absorbing, cur_info, env._model, env._data, carry)
    done = env._is_done(cur_obs, absorbing, cur_info, env._data, carry)
    carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
    env._obs = cur_obs
    env._additional_carry = carry
    return np.asarray(cur_obs), float(np.asarray(reward).item()), bool(done)


def joint_muscle_hybrid_frames(
    step: int,
    joint_traj,
    muscle_traj: MuscleTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pre/post full-body frames for hybrid step i.

    muscle ctrl[i] is recorded from pre-step state muscle.qpos[i] to post-step joint.qpos[i].
  """
    post_qpos = np.asarray(joint_traj.qpos[step], dtype=np.float64)
    post_qvel = np.asarray(joint_traj.qvel[step], dtype=np.float64)
    if step == 0:
        pre_qpos = np.asarray(muscle_traj.qpos[0], dtype=np.float64)
        pre_qvel = np.asarray(muscle_traj.qvel[0], dtype=np.float64)
    else:
        pre_qpos = np.asarray(joint_traj.qpos[step - 1], dtype=np.float64)
        pre_qvel = np.asarray(joint_traj.qvel[step - 1], dtype=np.float64)
    return pre_qpos, pre_qvel, post_qpos, post_qvel


def joint_left_reference_frames(
    step: int,
    joint_traj,
    left_qpos_indices: np.ndarray,
    left_qvel_indices: np.ndarray,
    muscle_traj: MuscleTrajectory | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left prosthesis pre/post for hybrid step i; non-left body is never pinned.

    joint.qpos[k] is post-step-k from the official joint rollout; muscle.qpos[0] is reset pose.
    """
    post_left_qpos = np.asarray(joint_traj.qpos[step][left_qpos_indices], dtype=np.float64)
    post_left_qvel = np.asarray(joint_traj.qvel[step][left_qvel_indices], dtype=np.float64)
    if step == 0:
        if muscle_traj is not None:
            pre_left_qpos = np.asarray(muscle_traj.qpos[0][left_qpos_indices], dtype=np.float64)
            pre_left_qvel = np.asarray(muscle_traj.qvel[0][left_qvel_indices], dtype=np.float64)
        else:
            pre_left_qpos = post_left_qpos.copy()
            pre_left_qvel = post_left_qvel.copy()
    else:
        pre_left_qpos = np.asarray(joint_traj.qpos[step - 1][left_qpos_indices], dtype=np.float64)
        pre_left_qvel = np.asarray(joint_traj.qvel[step - 1][left_qvel_indices], dtype=np.float64)
    return pre_left_qpos, pre_left_qvel, post_left_qpos, post_left_qvel


def aligned_joint_left_muscle_ctrl_steps(joint_frames: int, muscle_frames: int) -> int:
    """Steps when left reference comes from joint rollout and ctrl from muscle rollout."""
    return min(int(joint_frames), aligned_hybrid_replay_steps_from_muscle_only(muscle_frames))


def step_with_recorded_ctrl_and_left_joint_pd(
    env,
    recorded_ctrl: np.ndarray,
    disabled_actuators: np.ndarray,
    target_left_qpos: np.ndarray,
    left_qpos_indices: np.ndarray,
    left_qvel_indices: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    torque_limit: np.ndarray,
    torque_slew: np.ndarray | None = None,
    last_torque: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Apply recorded muscle ctrl with masked actuators zeroed; track left leg via qfrc_applied PD."""
    full_action_for_bookkeeping = np.zeros(int(env.info.action_space.shape[0]), dtype=np.float32)
    cur_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=full_action_for_bookkeeping)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    ctrl = np.asarray(recorded_ctrl, dtype=np.float64).reshape(-1).copy()
    if ctrl.shape[0] != env.data.ctrl.shape[0]:
        raise ValueError(f"Recorded ctrl dim {ctrl.shape[0]} != env ctrl dim {env.data.ctrl.shape[0]}")
    ctrl[disabled_actuators] = 0.0

    kp = np.asarray(kp, dtype=np.float64)
    kd = np.asarray(kd, dtype=np.float64)
    torque_limit = np.asarray(torque_limit, dtype=np.float64)
    target_left_qpos = np.asarray(target_left_qpos, dtype=np.float64).reshape(-1)

    for _ in range(env._n_intermediate_steps):
        env._data.qfrc_applied[:] = 0.0
        env._data.ctrl[:] = ctrl
        if disabled_actuators.size:
            env._data.ctrl[disabled_actuators] = 0.0
            actadr = np.asarray(env._model.actuator_actadr[disabled_actuators], dtype=np.int32)
            for adr in actadr:
                if int(adr) >= 0:
                    env._data.act[int(adr)] = 0.0

        q = np.asarray(env._data.qpos[left_qpos_indices], dtype=np.float64)
        qd = np.asarray(env._data.qvel[left_qvel_indices], dtype=np.float64)
        torque = kp * (target_left_qpos - q) - kd * qd
        torque = np.clip(torque, -torque_limit, torque_limit)
        if torque_slew is not None and last_torque is not None:
            slew = np.asarray(torque_slew, dtype=np.float64)
            delta = np.clip(torque - last_torque, -slew, slew)
            torque = last_torque + delta
        if last_torque is not None:
            last_torque[:] = torque
        env._data.qfrc_applied[left_qvel_indices] = torque
        mujoco.mj_step(env._model, env._data, env._n_substeps)
        env._data.qfrc_applied[:] = 0.0

    env._data, carry = env._simulation_post_step(env._model, env._data, carry)
    cur_obs, carry = env._create_observation(env._model, env._data, carry)
    cur_obs, env._data, cur_info, carry = env._step_finalize(cur_obs, env._model, env._data, cur_info, carry)
    cur_info = env._update_info_dictionary(cur_info, cur_obs, env._data, carry)
    absorbing, carry = env._is_absorbing(cur_obs, cur_info, env._data, carry)
    reward, carry = env._reward(env._obs, full_action_for_bookkeeping, cur_obs, absorbing, cur_info, env._model, env._data, carry)
    done = env._is_done(cur_obs, absorbing, cur_info, env._data, carry)
    carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
    env._obs = cur_obs
    env._additional_carry = carry
    return np.asarray(cur_obs), float(np.asarray(reward).item()), bool(done)


def record_left_joint_pinned_actuator_ctrl_masked_replay(
    *,
    env,
    muscle_traj: MuscleTrajectory,
    left_qpos: np.ndarray,
    left_qvel: np.ndarray,
    disabled_actuators: np.ndarray,
    record_path: Path,
    joint_traj=None,
    alignment: str = "muscle",
    width: int = 640,
    height: int = 480,
    fps: int | None = None,
    cam_distance: float = 6.0,
    cam_elevation: float = -20.0,
    cam_azimuth: float = 90.0,
    left_joint_kp: np.ndarray | None = None,
    left_joint_kd: np.ndarray | None = None,
    left_torque_limit: np.ndarray | None = None,
    torque_slew_limit: float = 35.0,
    left_joint_control_mode: str = "lock",
    hybrid_mode: str = "left_only",
    non_left_qpos: np.ndarray | None = None,
    non_left_qvel: np.ndarray | None = None,
) -> tuple[float, int, int]:
    """Replay masked muscle ctrl; left_only locks prosthesis DoF from joint reference only."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    muscle_dt = float(muscle_traj.dt)
    if joint_traj is not None and abs(float(joint_traj.dt) - muscle_dt) > 1e-6:
        raise ValueError(f"joint dt ({joint_traj.dt}) != muscle dt ({muscle_dt})")
    dt = muscle_dt
    alignment_mode = str(alignment).lower()
    fps = int(fps if fps is not None else round(1.0 / max(dt, 1e-6)))

    renderer = mujoco.Renderer(env.model, width=width, height=height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance
    cam.elevation = cam_elevation
    cam.azimuth = cam_azimuth

    kp = np.asarray(left_joint_kp if left_joint_kp is not None else [240.0, 180.0, 60.0, 40.0], dtype=np.float64)
    kd = np.asarray(left_joint_kd if left_joint_kd is not None else [24.0, 18.0, 6.0, 4.0], dtype=np.float64)
    torque_limit = np.asarray(
        left_torque_limit if left_torque_limit is not None else [140.0, 120.0, 60.0, 60.0],
        dtype=np.float64,
    )
    torque_slew = np.full(4, float(torque_slew_limit), dtype=np.float64)
    last_torque = np.zeros(4, dtype=np.float64)

    use_scaffold = str(hybrid_mode).lower() == "scaffold"
    if not use_scaffold and alignment_mode == "joint" and joint_traj is None:
        raise ValueError("alignment='joint' requires official joint_traj for left-leg reference")

    env.reset()
    env.data.qpos[:] = np.asarray(muscle_traj.qpos[0], dtype=np.float64)
    env.data.qvel[:] = np.asarray(muscle_traj.qvel[0], dtype=np.float64)
    if joint_traj is not None and not use_scaffold and alignment_mode == "joint":
        pre_left_qpos, pre_left_qvel, _, _ = joint_left_reference_frames(
            0, joint_traj, left_qpos, left_qvel, muscle_traj
        )
        env.data.qpos[left_qpos] = pre_left_qpos
        env.data.qvel[left_qvel] = pre_left_qvel
    mujoco.mj_forward(env.model, env.data)

    if use_scaffold:
        n_steps = aligned_hybrid_replay_steps_from_muscle_only(muscle_traj.n_frames)
    elif alignment_mode == "muscle":
        n_steps = aligned_hybrid_replay_steps_from_muscle_only(muscle_traj.n_frames)
    elif joint_traj is not None:
        n_steps = aligned_joint_left_muscle_ctrl_steps(joint_traj.n_frames, muscle_traj.n_frames)
    else:
        raise ValueError("alignment='joint' requires joint_traj")

    if getattr(env, "th", None) is not None:
        n_steps = min(n_steps, int(env.th.len_trajectory(0)) - 1)

    if use_scaffold:
        if non_left_qpos is None or non_left_qvel is None:
            raise ValueError("hybrid_mode='scaffold' requires non_left_qpos/non_left_qvel indices")

    effective_disabled_actuators = np.asarray(disabled_actuators, dtype=np.int32)
    use_lock = str(left_joint_control_mode).lower() == "lock"
    if use_lock and not use_scaffold:
        # knee15 leaves 4 thigh muscles active for OSL torque; they fight hard left joint lock.
        from musclemimic.proknee.constants import LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES

        left_lock_muscles = actuator_ids_for_muscle_names(env.model, LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES)
        effective_disabled_actuators = merge_disabled_actuator_ids(disabled_actuators, left_lock_muscles)

    episode_return = 0.0
    done_count = 0
    with imageio.get_writer(record_path, fps=fps, quality=8) as writer:
        for step in range(n_steps):
            ctrl = np.asarray(muscle_traj.actuator_ctrl[step], dtype=np.float64)
            if use_scaffold or alignment_mode == "muscle":
                pre_qpos, pre_qvel, post_qpos, post_qvel = muscle_self_aligned_hybrid_frames(step, muscle_traj)
            else:
                pre_qpos, pre_qvel, post_qpos, post_qvel = joint_muscle_hybrid_frames(step, joint_traj, muscle_traj)

            if joint_traj is not None and not use_scaffold and alignment_mode == "joint":
                pre_left_qpos, pre_left_qvel, post_left_qpos, post_left_qvel = joint_left_reference_frames(
                    step, joint_traj, left_qpos, left_qvel, muscle_traj
                )
            else:
                pre_left_qpos = np.asarray(pre_qpos[left_qpos], dtype=np.float64)
                pre_left_qvel = np.asarray(pre_qvel[left_qvel], dtype=np.float64)
                post_left_qpos = np.asarray(post_qpos[left_qpos], dtype=np.float64)
                post_left_qvel = np.asarray(post_qvel[left_qvel], dtype=np.float64)

            if use_scaffold:
                _obs, reward, done = step_with_scaffold_hybrid_replay(
                    env,
                    ctrl,
                    effective_disabled_actuators,
                    pre_qpos,
                    pre_qvel,
                    post_qpos,
                    post_qvel,
                    non_left_qpos,
                    non_left_qvel,
                    left_qpos,
                    left_qvel,
                )
            elif use_lock:
                _obs, reward, done = step_with_recorded_ctrl_and_left_joint_lock(
                    env,
                    ctrl,
                    effective_disabled_actuators,
                    pre_left_qpos,
                    pre_left_qvel,
                    post_left_qpos,
                    post_left_qvel,
                    left_qpos,
                    left_qvel,
                )
            else:
                env.data.qpos[left_qpos] = pre_left_qpos
                env.data.qvel[left_qvel] = pre_left_qvel
                mujoco.mj_forward(env.model, env.data)
                _obs, reward, done = step_with_recorded_ctrl_and_left_joint_pd(
                    env,
                    ctrl,
                    effective_disabled_actuators,
                    post_left_qpos,
                    left_qpos,
                    left_qvel,
                    kp,
                    kd,
                    torque_limit,
                    torque_slew,
                    last_torque,
                )
                env.data.qpos[left_qpos] = post_left_qpos
                env.data.qvel[left_qvel] = post_left_qvel
                mujoco.mj_forward(env.model, env.data)

            episode_return += float(reward)
            done_count += int(bool(done))

            cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            renderer.update_scene(env.data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()
    return float(episode_return), int(done_count), int(n_steps)


def record_open_loop_actuator_ctrl_masked_replay(
    *,
    env,
    traj: MuscleTrajectory,
    disabled_actuators: np.ndarray,
    record_path: Path,
    width: int = 640,
    height: int = 480,
    fps: int | None = None,
    cam_distance: float = 6.0,
    cam_elevation: float = -20.0,
    cam_azimuth: float = 90.0,
) -> tuple[float, int, int]:
    """Replay saved actuator_ctrl frames; zero disabled muscles; no policy or FSM."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    fps = int(fps if fps is not None else round(1.0 / max(float(traj.dt), 1e-6)))

    renderer = mujoco.Renderer(env.model, width=width, height=height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance
    cam.elevation = cam_elevation
    cam.azimuth = cam_azimuth

    env.reset()
    episode_return = 0.0
    done_count = 0
    n_steps = min(int(traj.n_frames), int(env.th.len_trajectory(0)) if getattr(env, "th", None) is not None else int(traj.n_frames))

    with imageio.get_writer(record_path, fps=fps, quality=8) as writer:
        for step in range(n_steps):
            ctrl = np.asarray(traj.actuator_ctrl[step], dtype=np.float64)
            _obs, reward, done = step_env_with_recorded_ctrl(env, ctrl, disabled_actuators)
            episode_return += float(reward)
            done_count += int(bool(done))

            cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            renderer.update_scene(env.data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()
    return float(episode_return), int(done_count), int(n_steps)


def write_replay_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
