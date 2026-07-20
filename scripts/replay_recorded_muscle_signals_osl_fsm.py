#!/usr/bin/env python
"""Replay recorded muscle signals with 19 left-leg muscles zeroed and OSL FSM torque."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.distill.config import repo_root
from musclemimic.distill.osl_deploy_defaults import (
    DEFAULT_OSL_ANKLE_TORQUE_LIMIT,
    DEFAULT_OSL_FOOT_TORQUE_LIMIT,
    DEFAULT_OSL_KNEE_TORQUE_LIMIT,
    DEFAULT_OSL_LOADCELL_SCALE,
    DEFAULT_OSL_MASK_PRESET,
    DEFAULT_OSL_TORQUE_SLEW_LIMIT,
    loadcell_range_for_body_weight,
    resolve_loadcell_clip_kwargs,
)
from musclemimic.distill.osl_harness import build_osl_harness
from musclemimic.evaluation.joint_replay import write_replay_meta
from musclemimic.evaluation.logger import safe_motion_name
from musclemimic.evaluation.muscle_replay import load_muscle_trajectory
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS
from musclemimic.runner.eval_utils import apply_temporal_params, load_checkpoint, setup_headless

FOUR_REPLAY_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion_path", default=None)
    p.add_argument("--all-four", action="store_true")
    p.add_argument("--controller_type", default="official_mm10m2")
    p.add_argument("--checkpoint_path", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    p.add_argument(
        "--source_root",
        default=None,
        help="Default: outputs/replay/muscle_mimic/<controller_type>",
    )
    p.add_argument(
        "--output_root",
        default=None,
        help="Default: outputs/replay/muscle_mimic_osl_fsm/<controller_type>",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--knee-torque-limit", type=float, default=DEFAULT_OSL_KNEE_TORQUE_LIMIT)
    p.add_argument("--ankle-torque-limit", type=float, default=DEFAULT_OSL_ANKLE_TORQUE_LIMIT)
    p.add_argument("--foot-torque-limit", type=float, default=DEFAULT_OSL_FOOT_TORQUE_LIMIT)
    p.add_argument("--torque-slew-limit", type=float, default=DEFAULT_OSL_TORQUE_SLEW_LIMIT)
    p.add_argument("--loadcell-scale", type=float, default=DEFAULT_OSL_LOADCELL_SCALE)
    p.add_argument(
        "--loadcell-range",
        default=None,
        help="Override OSL loadcell clamp range, e.g. '-826.7,0'. Default clips to [-body_weight_n, 0].",
    )
    p.add_argument("--body-weight-n", type=float, default=None)
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument(
        "--mask-preset",
        choices=["foot1", "foot5", "distal11", "knee15", "strict19", "all"],
        default=DEFAULT_OSL_MASK_PRESET,
        help="Disabled muscle preset. all records foot1, foot5, distal11, knee15, and strict19.",
    )
    return p.parse_args()


def parse_range(value: str | None) -> tuple[float, float] | None:
    if value is None or str(value).strip() == "":
        return None
    vals = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if len(vals) != 2:
        raise ValueError(f"Expected two comma-separated values for --loadcell-range, got {value!r}")
    lo, hi = vals
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def source_root(args: argparse.Namespace) -> Path:
    if args.source_root:
        root = Path(args.source_root)
        return root if root.is_absolute() else repo_root() / root
    return repo_root() / "outputs" / "replay" / "muscle_mimic" / str(args.controller_type)


def output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        root = Path(args.output_root)
        return root if root.is_absolute() else repo_root() / root
    return repo_root() / "outputs" / "replay" / "muscle_mimic_osl_fsm" / str(args.controller_type)


def build_env(config, motion_path: str):
    OmegaConf.set_struct(config, False)
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBody"
    env_params["headless"] = True
    env_params.pop("prosthesis", None)
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    return factory.make(**{**env_params, **task_params})


def actuator_ids_for_names(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
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
    """Color disabled muscle tendons blue when they are rendered by MuJoCo."""
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


def step_with_recorded_ctrl_and_osl(
    env,
    harness,
    recorded_ctrl: np.ndarray,
    disabled_actuators: np.ndarray,
):
    """Step MuJoCo with recorded low-level ctrl and OSL 4-DoF qfrc_applied."""
    full_action_for_bookkeeping = np.zeros(int(env.info.action_space.shape[0]), dtype=np.float32)
    cur_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=full_action_for_bookkeeping)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    raw_loadcell_fz_n = harness.adapter.raw_scaled_loadcell_fz_n(env.model, env.data)
    osl_inputs = harness.adapter.read_inputs(env.model, env.data)
    command = harness.controller.update(osl_inputs)
    _target_qpos, ff_torque, fsm_diag = harness.adapter.command_to_targets(env.model, env.data, command)
    ff_torque = np.asarray(ff_torque, dtype=np.float64)
    diag = {
        "state": fsm_diag.state,
        "loadcell_fz_n": float(osl_inputs.loadcell_fz_n),
        "raw_loadcell_fz_n": raw_loadcell_fz_n,
        "knee_torque": float(fsm_diag.knee_torque),
        "ankle_torque": float(fsm_diag.ankle_torque),
        "subtalar_torque": float(fsm_diag.subtalar_torque),
        "mtp_torque": float(fsm_diag.mtp_torque),
        "prosthesis_torque": ff_torque.astype(np.float32),
    }

    ctrl = np.asarray(recorded_ctrl, dtype=np.float64).reshape(-1).copy()
    if ctrl.shape[0] != env.data.ctrl.shape[0]:
        raise ValueError(f"Recorded ctrl dim {ctrl.shape[0]} != env ctrl dim {env.data.ctrl.shape[0]}")
    ctrl[disabled_actuators] = 0.0

    torque_abs_max = 0.0
    for _ in range(env._n_intermediate_steps):
        env._data.qfrc_applied[:] = 0.0
        env._data.ctrl[:] = ctrl
        if disabled_actuators.size:
            env._data.ctrl[disabled_actuators] = 0.0
            actadr = np.asarray(env._model.actuator_actadr[disabled_actuators], dtype=np.int32)
            for adr in actadr:
                if int(adr) >= 0:
                    env._data.act[int(adr)] = 0.0

        torque = np.clip(ff_torque, -harness.torque_limit, harness.torque_limit)
        if harness.torque_slew is not None and harness.last_torque is not None:
            delta = np.clip(torque - harness.last_torque, -harness.torque_slew, harness.torque_slew)
            torque = harness.last_torque + delta
        harness.last_torque[:] = torque
        torque_abs_max = max(torque_abs_max, float(np.max(np.abs(torque))))
        env._data.qfrc_applied[harness.audit.qvel_indices] = torque
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
    diag["root_height"] = float(env.data.qpos[2])
    diag["disabled_ctrl_norm"] = (
        float(np.linalg.norm(env.data.ctrl[disabled_actuators])) if disabled_actuators.size else 0.0
    )
    diag["prosthesis_torque_abs_max"] = float(torque_abs_max)
    return np.asarray(cur_obs), float(np.asarray(reward).item()), bool(done), diag


def replay_one(args: argparse.Namespace, config, motion_path: str, preset_name: str) -> dict:
    disabled_names = MUSCLE_MASK_PRESETS[preset_name]
    safe = safe_motion_name(motion_path)
    src_traj = source_root(args) / safe / "muscle_trajectory.npz"
    if not src_traj.is_file():
        raise FileNotFoundError(f"Missing source muscle trajectory: {src_traj}")
    traj = load_muscle_trajectory(src_traj)

    out_dir = output_root(args) / preset_name / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f".muscle_ctrl_{preset_name}_osl_fsm_raw.mp4"
    video_path = web_mp4_path(out_dir / f"muscle_ctrl_{preset_name}_osl_fsm.mp4")
    log_path = out_dir / f"muscle_ctrl_{preset_name}_osl_fsm.csv"
    meta_path = out_dir / f"muscle_ctrl_{preset_name}_osl_fsm_meta.json"

    env = build_env(config, motion_path)
    highlighted_tendons = color_disabled_muscle_tendons_blue(env.model, tuple(disabled_names))
    loadcell_range = parse_range(args.loadcell_range)
    harness = build_osl_harness(
        env,
        knee_torque_limit=float(args.knee_torque_limit),
        ankle_torque_limit=float(args.ankle_torque_limit),
        foot_torque_limit=float(args.foot_torque_limit),
        torque_slew_limit=float(args.torque_slew_limit),
        loadcell_scale=float(args.loadcell_scale),
        body_weight_n=args.body_weight_n,
        mask_preset=preset_name,
        **resolve_loadcell_clip_kwargs(loadcell_range),
    )
    disabled_actuators = harness.disabled_actuators

    fps = args.fps if args.fps is not None else int(round(1.0 / max(float(traj.dt), 1e-6)))
    n_steps = min(int(traj.n_frames), int(env.th.len_trajectory(0)) if getattr(env, "th", None) is not None else int(traj.n_frames))
    print(f"[{preset_name} ctrl + OSL FSM replay] {motion_path} -> {video_path} ({n_steps} steps)")
    print(f"disabled_actuators={disabled_actuators.tolist()} highlighted_tendons={highlighted_tendons}")
    effective_range = loadcell_range or loadcell_range_for_body_weight(harness.body_weight_n)
    print(f"loadcell_range={effective_range}")

    env.reset()
    harness.reset()
    renderer = mujoco.Renderer(env.model, width=int(args.width), height=int(args.height))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 6.0
    cam.elevation = -20.0
    cam.azimuth = 90.0

    episode_return = 0.0
    done_count = 0
    disabled_ctrl_max = 0.0

    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(
        [
            "step",
            "fsm_state",
            "raw_loadcell_fz_n",
            "loadcell_fz_n",
            "knee_torque",
            "ankle_torque",
            "subtalar_torque",
            "mtp_torque",
            "disabled_action_norm_before_zero",
            "disabled_ctrl_norm_before_zero",
            "disabled_ctrl_norm",
            "root_height",
            "done",
        ]
    )
    try:
        with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
            for step in range(n_steps):
                ctrl = np.asarray(traj.actuator_ctrl[step], dtype=np.float32).reshape(-1).copy()
                disabled_ctrl_before = float(np.linalg.norm(ctrl[disabled_actuators]))

                _obs, reward, done, diag = step_with_recorded_ctrl_and_osl(
                    env,
                    harness,
                    ctrl,
                    disabled_actuators,
                )
                episode_return += float(reward)
                done_count += int(bool(done))
                disabled_norm = float(diag.get("disabled_ctrl_norm", 0.0))
                disabled_ctrl_max = max(disabled_ctrl_max, disabled_norm)
                tau = np.asarray(diag.get("prosthesis_torque", np.zeros(4)), dtype=np.float64).reshape(-1)
                if tau.size < 4:
                    tau = np.pad(tau, (0, 4 - tau.size))
                log_w.writerow(
                    [
                        step + 1,
                        str(diag.get("state", "osl_fsm")),
                        f"{float(diag.get('raw_loadcell_fz_n', 0.0)):.6f}",
                        f"{float(diag.get('loadcell_fz_n', 0.0)):.6f}",
                        f"{float(tau[0]):.6f}",
                        f"{float(tau[1]):.6f}",
                        f"{float(tau[2]):.6f}",
                        f"{float(tau[3]):.6f}",
                        "0.000000000",
                        f"{disabled_ctrl_before:.9f}",
                        f"{disabled_norm:.9f}",
                        f"{float(env.data.qpos[2]):.6f}",
                        int(bool(done)),
                    ]
                )

                cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())
                if step % 200 == 0:
                    print(
                        f"step={step + 1}/{n_steps} state={diag.get('state', 'osl_fsm')} "
                        f"root={float(env.data.qpos[2]):.3f} disabled_ctrl={disabled_norm:.2e}",
                        flush=True,
                    )
    finally:
        renderer.close()
        env.stop()
        log_f.close()

    encode_web_mp4(raw_path, video_path, remove_src=not args.keep_raw)
    meta = {
        "replay_mode": f"recorded_actuator_ctrl_{preset_name}_with_osl_fsm",
        "description": (
            "Open-loop replay of recorded low-level actuator_ctrl frames; the selected disabled "
            "muscle actuator ctrls and activations are zeroed before stepping; OSL FSM injects "
            "knee/ankle/subtalar/mtp prosthesis torques. No policy is queried."
        ),
        "source_muscle_trajectory_npz": str(src_traj),
        "video": str(video_path),
        "deploy_log_csv": str(log_path),
        "motion_path": motion_path,
        "n_steps": n_steps,
        "fps": fps,
        "duration_s": float(n_steps * traj.dt),
        "episode_return": float(episode_return),
        "done_count": int(done_count),
        "mask_preset": preset_name,
        "disabled_muscle_count": int(len(disabled_names)),
        "disabled_muscle_names": list(disabled_names),
        "disabled_actuator_ids": disabled_actuators.tolist(),
        "disabled_ctrl_max_norm": float(disabled_ctrl_max),
        "highlighted_disabled_muscle_tendons": int(highlighted_tendons),
        "prosthesis_controller": "osl_fsm",
        "knee_torque_limit": float(args.knee_torque_limit),
        "ankle_torque_limit": float(args.ankle_torque_limit),
        "foot_torque_limit": float(args.foot_torque_limit),
        "torque_slew_limit": float(args.torque_slew_limit),
        "loadcell_range": list(effective_range),
    }
    write_replay_meta(meta_path, meta)
    print(f"  frames={n_steps} return={episode_return:.3f} done_count={done_count} video={video_path}")
    return meta


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    motions = list(FOUR_REPLAY_MOTIONS) if args.all_four else [args.motion_path]
    if not motions or motions == [None]:
        raise ValueError("Provide --motion_path or --all-four")

    config, _agent_state, _metadata = load_checkpoint(args.checkpoint_path)
    OmegaConf.set_struct(config, False)
    apply_temporal_params(config)

    presets = ["foot1", "foot5", "distal11", "knee15", "strict19"] if args.mask_preset == "all" else [args.mask_preset]
    all_records = {}
    for preset in presets:
        records = [replay_one(args, config, motion, preset) for motion in motions]
        all_records[preset] = records
        if len(records) > 1:
            summary_dir = output_root(args) / preset / "replay_four"
            summary_dir.mkdir(parents=True, exist_ok=True)
            write_replay_meta(summary_dir / "summary.json", {"mask_preset": preset, "motions": records})
            print(f"Wrote summary: {summary_dir / 'summary.json'}")
    if len(presets) > 1:
        summary_dir = output_root(args) / "replay_four"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_replay_meta(summary_dir / "summary.json", {"presets": all_records})
        print(f"Wrote combined summary: {summary_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
