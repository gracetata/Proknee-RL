#!/usr/bin/env python
"""Hybrid rollout: replay non-left joints from joint_trajectory.npz, OSL FSM on left leg."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
OSL_DIR = REPO_ROOT.parent / "musclebenchmark" / "opensourceleg_fsm"
for path in (str(REPO_ROOT), str(OSL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from official_fsm import OpenSourceLegFSMController  # noqa: E402
from sim_adapter import MuscleMimicOSLAdapter, OSLAdapterConfig, model_body_weight_n  # noqa: E402

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.osl_deploy_defaults import (
    DEFAULT_OSL_ANKLE_TORQUE_LIMIT,
    DEFAULT_OSL_FOOT_TORQUE_LIMIT,
    DEFAULT_OSL_KNEE_TORQUE_LIMIT,
    DEFAULT_OSL_LOADCELL_SCALE,
    DEFAULT_OSL_MASK_PRESET,
    DEFAULT_OSL_PROSTHESIS_MUSCLE_SCALE,
    DEFAULT_OSL_TORQUE_SLEW_LIMIT,
    loadcell_range_for_body_weight,
    resolve_loadcell_clip_kwargs,
)
from musclemimic.evaluation.joint_replay import load_joint_trajectory, write_replay_meta
from musclemimic.proknee.constants import audit_myofullbody_left_leg
from musclemimic.prosthesis.constants import MUSCLE_MASK_PRESETS
from musclemimic.runner.eval_utils import (
    align_agent_state,
    apply_temporal_params,
    configure_goal_visualization,
    load_checkpoint,
)
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from record_stage1_testset_visual import (  # noqa: E402
    IndexedValues,
    highlight_left_leg,
    step_with_stage1_pd,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--joint-trajectory",
        default=str(
            REPO_ROOT
            / "outputs/replay/joint_mimic/official_mm10m2/KIT_3_walk_6m_straight_line04_poses/joint_trajectory.npz"
        ),
    )
    p.add_argument(
        "--record-path",
        default=None,
        help="Default: same dir as joint_trajectory -> joint_replay_osl_fsm_web.mp4",
    )
    p.add_argument("--deploy-log", default=None)
    p.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=100)
    p.add_argument("--prosthesis-muscle-scale", type=float, default=DEFAULT_OSL_PROSTHESIS_MUSCLE_SCALE)
    p.add_argument("--mask-preset", choices=sorted(MUSCLE_MASK_PRESETS), default=DEFAULT_OSL_MASK_PRESET)
    p.add_argument("--knee-torque-limit", type=float, default=DEFAULT_OSL_KNEE_TORQUE_LIMIT)
    p.add_argument("--ankle-torque-limit", type=float, default=DEFAULT_OSL_ANKLE_TORQUE_LIMIT)
    p.add_argument("--foot-torque-limit", type=float, default=DEFAULT_OSL_FOOT_TORQUE_LIMIT)
    p.add_argument(
        "--loadcell-scale",
        type=float,
        default=DEFAULT_OSL_LOADCELL_SCALE,
        help="Scale MuJoCo prosthesis contact force before OSL FSM thresholds.",
    )
    p.add_argument(
        "--loadcell-range",
        default=None,
        help="Override OSL loadcell clamp range, e.g. '-826.7,0'. Default clips to [-body_weight_n, 0].",
    )
    p.add_argument(
        "--body-weight-n",
        type=float,
        default=None,
        help="Body weight for OSL load thresholds; default = sum(model.body_mass)*g.",
    )
    p.add_argument("--torque-slew-limit", type=float, default=DEFAULT_OSL_TORQUE_SLEW_LIMIT)
    p.add_argument("--keep-raw", action="store_true", help="Keep imageio raw mp4 before web encode.")
    p.add_argument("--max-steps", type=int, default=0, help="0 = all frames in npz")
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


def disabled_muscle_actuator_indices(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    ids = []
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            ids.append(int(aid))
    return np.asarray(sorted(set(ids)), dtype=np.int32)


def build_env(checkpoint_path: str, motion_path: str):
    config, agent_state, _ = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True
    config.experiment.task_factory.params.amass_dataset_conf.dataset_group = None
    config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = [motion_path]

    class _ArgsShim:
        no_render = True
        record = False
        mujoco_viewer = False
        use_mujoco = True
        viser_viewer = False
        hide_ghost = True

    configure_goal_visualization(config, _ArgsShim(), "GoalTrajMimicv2", is_mjx_env=False)
    goal_params = config.experiment.env_params.get("goal_params", {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    config.experiment.env_params["goal_params"] = goal_params
    apply_temporal_params(config)

    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)
    if "Mjx" in env_params.get("env_name", ""):
        env_params["env_name"] = env_params["env_name"].replace("Mjx", "")

    th_params = env_params.setdefault("th_params", {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    env = TaskFactory.get_factory_cls(config.experiment.task_factory.name).make(**{**env_params, **task_params})
    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    return env, agent_conf, agent_state


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


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    traj_path = Path(args.joint_trajectory)
    traj = load_joint_trajectory(traj_path)
    out_dir = traj_path.parent
    record_path = Path(args.record_path) if args.record_path else out_dir / "joint_replay_osl_fsm_web.mp4"
    record_path = web_mp4_path(record_path)
    raw_path = out_dir / ".joint_replay_osl_fsm_raw.mp4"
    deploy_log = Path(args.deploy_log) if args.deploy_log else out_dir / "joint_replay_osl_fsm.csv"
    meta_path = (
        out_dir / "joint_replay_osl_fsm_meta.json"
        if args.record_path is None
        else record_path.with_name(f"{record_path.stem.removesuffix('_web')}_meta.json")
    )
    n_steps = int(args.max_steps) if args.max_steps > 0 else traj.n_frames
    loadcell_range = parse_range(args.loadcell_range)

    record_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Joint trajectory: {traj_path} ({traj.n_frames} frames, {traj.duration_s:.2f}s)")
    print(f"Motion: {traj.motion_path}")
    print(f"Record: {record_path}")

    env, _agent_conf, _agent_state = build_env(args.checkpoint, traj.motion_path)
    body_weight_n = float(args.body_weight_n) if args.body_weight_n is not None else model_body_weight_n(env.model)
    audit = audit_myofullbody_left_leg(env.model)
    highlighted = highlight_left_leg(env.model, [j.joint_id for j in audit.joints])
    disabled_names = MUSCLE_MASK_PRESETS[args.mask_preset]
    masked_actuators = disabled_muscle_actuator_indices(env.model, disabled_names)

    left_qpos = np.asarray(audit.qpos_indices, dtype=np.int32)
    left_qvel = np.asarray(audit.qvel_indices, dtype=np.int32)
    all_qpos = np.arange(env.model.nq, dtype=np.int32)
    all_qvel = np.arange(env.model.nv, dtype=np.int32)
    non_left_qpos = np.setdiff1d(all_qpos, left_qpos)
    non_left_qvel = np.setdiff1d(all_qvel, left_qvel)

    clip_kwargs = resolve_loadcell_clip_kwargs(loadcell_range)
    adapter = MuscleMimicOSLAdapter(
        env.model,
        audit,
        OSLAdapterConfig(
            knee_torque_limit=args.knee_torque_limit,
            ankle_torque_limit=args.ankle_torque_limit,
            foot_torque_limit=args.foot_torque_limit,
            loadcell_scale=args.loadcell_scale,
            **clip_kwargs,
        ),
        body_weight_n=body_weight_n,
    )
    controller = OpenSourceLegFSMController(body_weight_n=body_weight_n)
    th = controller.thresholds
    effective_range = loadcell_range or loadcell_range_for_body_weight(body_weight_n)
    print(
        f"OSL calibration: body_weight_n={body_weight_n:.1f}N "
        f"thresholds(lstance={th.load_lstance_n:.1f}, eswing={th.load_eswing_n:.1f}, estance={th.load_estance_n:.1f}) "
        f"loadcell_scale={args.loadcell_scale} loadcell_range={effective_range} "
        f"mask_preset={args.mask_preset} masked_actuators={masked_actuators.size}"
    )

    env.reset()
    pin_non_left_joints(env.model, env.data, traj.qpos[0], traj.qvel[0], non_left_qpos, non_left_qvel)
    env.data.qpos[left_qpos] = traj.qpos[0][left_qpos]
    env.data.qvel[left_qvel] = traj.qvel[0][left_qvel]
    mujoco.mj_forward(env.model, env.data)
    adapter.reset_hold_targets(env.data)
    controller.reset()

    action_dim = int(env.info.action_space.shape[0])
    oracle_action = np.zeros(action_dim, dtype=np.float32)
    zero_kp = np.zeros(4, dtype=np.float64)
    zero_kd = np.zeros(4, dtype=np.float64)
    torque_limit = np.asarray(
        [args.knee_torque_limit, args.ankle_torque_limit, args.foot_torque_limit, args.foot_torque_limit],
        dtype=np.float64,
    )
    torque_slew = np.full(4, float(args.torque_slew_limit), dtype=np.float64)
    last_torque = np.zeros(4, dtype=np.float64)

    log_f = open(deploy_log, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(
        [
            "step",
            "fsm_state",
            "raw_loadcell_fz_n",
            "loadcell_fz_n",
            "root_height",
            "knee_q",
            "ankle_q",
            "knee_target",
            "ankle_target",
            "knee_torque",
            "ankle_torque",
            "left_knee_replay_q",
            "left_knee_sim_q",
        ]
    )

    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 6.0
    cam.elevation = -20.0
    cam.azimuth = 90.0

    try:
        with imageio.get_writer(raw_path, fps=args.fps, quality=8) as writer:
            for step in range(n_steps):
                frame = min(step, traj.n_frames - 1)
                pin_non_left_joints(
                    env.model,
                    env.data,
                    traj.qpos[frame],
                    traj.qvel[frame],
                    non_left_qpos,
                    non_left_qvel,
                )

                raw_loadcell_fz_n = adapter.raw_scaled_loadcell_fz_n(env.model, env.data)
                osl_inputs = adapter.read_inputs(env.model, env.data)
                command = controller.update(osl_inputs)
                _target_qpos, ff_torque, diag = adapter.command_to_targets(env.model, env.data, command)
                current_q = np.asarray(env.data.qpos[audit.qpos_indices], dtype=np.float64)

                _obs, _reward, _absorbing, done, _info = step_with_stage1_pd(
                    env,
                    oracle_action,
                    IndexedValues(audit.qpos_indices, current_q),
                    audit.qvel_indices,
                    masked_actuators,
                    args.prosthesis_muscle_scale,
                    zero_kp,
                    zero_kd,
                    torque_limit,
                    torque_slew,
                    last_torque,
                    ff_torque,
                )

                pin_non_left_joints(
                    env.model,
                    env.data,
                    traj.qpos[frame],
                    traj.qvel[frame],
                    non_left_qpos,
                    non_left_qvel,
                )

                q = np.asarray(env.data.qpos[audit.qpos_indices], dtype=np.float64)
                log_w.writerow(
                    [
                        step + 1,
                        diag.state,
                        f"{raw_loadcell_fz_n:.6f}",
                        f"{osl_inputs.loadcell_fz_n:.6f}",
                        f"{float(env.data.qpos[2]):.6f}",
                        f"{q[0]:.6f}",
                        f"{q[1]:.6f}",
                        f"{diag.knee_theta_target_rad:.6f}",
                        f"{diag.ankle_theta_target_rad:.6f}",
                        f"{diag.knee_torque:.6f}",
                        f"{diag.ankle_torque:.6f}",
                        f"{traj.qpos[frame, audit.qpos_indices[0]]:.6f}",
                        f"{q[0]:.6f}",
                    ]
                )

                cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                renderer.update_scene(env.data, camera=cam)
                writer.append_data(renderer.render())

                if step % 200 == 0:
                    print(
                        f"step={step+1}/{n_steps} fsm={diag.state} "
                        f"knee_sim={q[0]:.3f} knee_replay={traj.qpos[frame, audit.qpos_indices[0]]:.3f}",
                        flush=True,
                    )
                if done:
                    print(f"Done flag at step {step+1}", flush=True)
                    break
    finally:
        renderer.close()
        env.stop()
        log_f.close()

    meta = {
        "replay_mode": "joint_replay_osl_fsm_hybrid",
        "description": (
            "Non-left joints pinned to official rollout qpos from joint_trajectory.npz; "
            "left 4-DOF prosthesis driven by Open-Source Leg FSM with boundary muscles masked."
        ),
        "joint_trajectory_npz": str(traj_path.resolve()),
        "video": str(record_path.resolve()),
        "deploy_log_csv": str(deploy_log.resolve()),
        "motion_path": traj.motion_path,
        "n_steps": n_steps,
        "fsm_controller": "OpenSourceLegFSMController (musclebenchmark/opensourceleg_fsm)",
        "fsm_body_weight_n": body_weight_n,
        "fsm_load_thresholds_n": {
            "load_lstance": th.load_lstance_n,
            "load_eswing": th.load_eswing_n,
            "load_estance": th.load_estance_n,
        },
        "loadcell_scale": args.loadcell_scale,
        "loadcell_range": list(effective_range),
        "prosthesis_muscle_scale": args.prosthesis_muscle_scale,
        "mask_preset": args.mask_preset,
        "disabled_muscle_names": list(disabled_names),
        "left_joint_names": list(audit.joints[i].name for i in range(len(audit.joints))),
        "masked_boundary_muscles": int(masked_actuators.size),
        "highlighted_geoms": highlighted,
        "osl_inputs": ["knee_position_rad", "knee_velocity_rad_s", "ankle_position_rad", "loadcell_fz_n"],
    }
    write_replay_meta(meta_path, meta)

    try:
        encode_web_mp4(raw_path, record_path, remove_src=not args.keep_raw)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"Web encode failed for {record_path}: {exc}") from exc

    print(f"Saved video: {record_path}")
    print(f"Saved log: {deploy_log}")
    print(f"Saved meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
