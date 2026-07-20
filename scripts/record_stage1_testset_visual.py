#!/usr/bin/env python
"""Record Stage1 or Stage2 audit / physical-PD videos on KIT test motions.

The physics rollout is driven by the frozen mm-10m-2 policy. **Visual audit**
replaces only the rendered left-leg qpos with the policy prediction (oracle
physics unchanged). **Physical PD** applies the policy target on the four
prosthesis DOFs via ``qfrc_applied`` (same hybrid as Stage1 training).

Use ``--kind stage1`` (Teacher: obs+priv) or ``--kind stage2`` (Student:
obs+proprio history, no privileged info at inference).
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
from dataclasses import dataclass

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.smpl.const import KIT_KINESIS_TESTING_MOTIONS
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.algorithms.ppo.inference import ObservationHistoryBuffer
from musclemimic.proknee.constants import LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES, audit_myofullbody_left_leg
from musclemimic.proknee.models import MuscleProKneeStudent, MuscleProKneeTeacher
from musclemimic.proknee.observation import ProprioHistory, build_global_observation, build_priv_info
from musclemimic.runner.eval_utils import (
    align_agent_state,
    apply_temporal_params,
    configure_goal_visualization,
    load_checkpoint,
)


@dataclass
class BuiltEnv:
    env: object
    config: object
    agent_conf: object
    agent_state: object


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2")
    parser.add_argument("--policy", default="/home/user/Workspace/musclemimic/outputs/proknee_stage1_full/stage1_best.pt")
    parser.add_argument(
        "--kind",
        choices=["stage1", "stage2"],
        default="stage1",
        help="stage1: Teacher(obs,priv). stage2: Student(obs,hist) only.",
    )
    parser.add_argument("--motion-path", nargs="*", default=None, help="Explicit motions. Defaults to first N KIT test motions.")
    parser.add_argument("--num-motions", type=int, default=5)
    parser.add_argument("--steps-per-motion", type=int, default=900)
    parser.add_argument("--record-path", default="/home/user/Workspace/musclemimic/videos/stage1_testset_5motions_visual.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pd-override",
        action="store_true",
        help="Physically execute policy targets via qfrc_applied PD on prosthesis DOFs.",
    )
    parser.add_argument("--pd-kp", type=float, default=250.0)
    parser.add_argument("--pd-kd", type=float, default=25.0)
    parser.add_argument(
        "--joint-kp",
        default=None,
        help="Per-joint kp for knee, ankle, subtalar, mtp. Defaults to checkpoint pd_kp if present.",
    )
    parser.add_argument(
        "--joint-kd",
        default=None,
        help="Per-joint kd for knee, ankle, subtalar, mtp. Defaults to checkpoint pd_kd if present.",
    )
    parser.add_argument("--pd-torque-limit", type=float, default=120.0)
    parser.add_argument(
        "--pd-torque-limit-list",
        default=None,
        help="Per-joint torque limit for knee, ankle, subtalar, mtp; overrides --pd-torque-limit.",
    )
    parser.add_argument(
        "--pd-torque-slew-limit-list",
        default=None,
        help="Optional per-substep torque change cap for knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--oracle-torque-ff-scale",
        type=float,
        default=None,
        help="Scale oracle qfrc_actuator on prosthesis DOFs as torque feedforward during --pd-override.",
    )
    parser.add_argument(
        "--oracle-torque-ff-limit-list",
        default=None,
        help="Optional per-joint absolute cap for oracle torque feedforward: knee, ankle, subtalar, mtp.",
    )
    parser.add_argument(
        "--predicted-torque-ff-scale",
        type=float,
        default=None,
        help="Scale Stage1 torque head before feedforward; defaults to checkpoint value.",
    )
    parser.add_argument(
        "--disable-predicted-torque-head",
        action="store_true",
        help="Ignore a checkpoint torque head and use oracle ghost torque feedforward instead.",
    )
    parser.add_argument(
        "--residual-horizon",
        type=float,
        default=20.0,
        help="Convert one-step Stage1 residuals to a short-horizon PD position target.",
    )
    parser.add_argument(
        "--residual-filter",
        type=float,
        default=0.85,
        help="Low-pass coefficient for Stage1 residuals; higher is smoother.",
    )
    parser.add_argument(
        "--no-mask-conflict-muscles",
        action="store_true",
        help="Do not zero left knee/ankle crossing muscle controls before PD stepping.",
    )
    parser.add_argument(
        "--prosthesis-muscle-scale",
        type=float,
        default=0.2,
        help="Scale oracle controls on prosthesis-boundary muscles (same semantics as training).",
    )
    parser.add_argument(
        "--conflict-muscle-scale",
        type=float,
        default=None,
        help="Deprecated: use --prosthesis-muscle-scale. If set, overrides it.",
    )
    parser.add_argument(
        "--no-web-encode",
        action="store_true",
        help="Skip ffmpeg re-encode to *_web.mp4 (baseline H.264 for IDE players).",
    )
    parser.add_argument(
        "--max-qpos-target-step",
        type=float,
        default=None,
        help="Cap per-step change (rad) of commanded prosthesis qpos before PD; reduces tremor vs oracle.",
    )
    parser.add_argument(
        "--max-qpos-target-step-list",
        default=None,
        help="Per-joint target step cap (rad) for knee, ankle, subtalar, mtp; overrides --max-qpos-target-step.",
    )
    parser.add_argument(
        "--pd-gain-scale",
        type=float,
        default=1.0,
        help="Scale joint_kp/joint_kd during --pd-override (often 0.5–0.8 calms stiff PD).",
    )
    parser.add_argument(
        "--deploy-log",
        default=None,
        help="With --pd-override: append CSV per-step diagnostics (contacts, qvel, tracking error, toe height).",
    )
    return parser.parse_args()


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_gain_list(text: str, fallback: float, n: int) -> np.ndarray:
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    except Exception:
        vals = []
    if len(vals) != n:
        vals = [float(fallback)] * n
    return np.asarray(vals, dtype=np.float64)


def ckpt_vector(ckpt: dict, key: str, n: int) -> np.ndarray | None:
    value = ckpt.get(key, None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != n:
        return None
    return arr


def parse_optional_gain_list(text: str | None, fallback: float | None, n: int) -> np.ndarray | None:
    if text is None:
        if fallback is None:
            return None
        return np.full(n, float(fallback), dtype=np.float64)
    return parse_gain_list(text, fallback if fallback is not None else 0.0, n)


def _build_obs_buffer(config, env):
    exp_cfg = config.experiment
    len_obs_history = int(getattr(exp_cfg, "len_obs_history", 1))
    if len_obs_history <= 1:
        return None
    split_goal = bool(getattr(exp_cfg, "split_goal", False))
    if not split_goal:
        return ObservationHistoryBuffer(len_obs_history)

    goal_indices = np.asarray(env.obs_container.get_obs_ind_by_group("goal"), dtype=int)
    raw_obs_dim = env.info.observation_space.shape[0]
    state_mask = np.ones(raw_obs_dim, dtype=bool)
    state_mask[goal_indices] = False
    state_indices = np.arange(raw_obs_dim, dtype=int)[state_mask]
    return ObservationHistoryBuffer(
        len_obs_history,
        split_goal=True,
        state_indices=state_indices,
        goal_indices=goal_indices,
    )


def build_official_eval_env(checkpoint_path: str, motion_path: str) -> BuiltEnv:
    config, agent_state, _metadata = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)

    env_name = config.experiment.env_params.get("env_name")
    config.experiment.env_params["headless"] = True
    config.experiment.task_factory.params.amass_dataset_conf.dataset_group = None
    config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = [motion_path]

    class _ArgsShim:
        no_render = True
        record = True
        mujoco_viewer = False
        use_mujoco = True
        viser_viewer = False

    configure_goal_visualization(config, _ArgsShim(), "GoalTrajMimicv2", is_mjx_env="Mjx" in env_name)
    apply_temporal_params(config)

    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)
    if "Mjx" in env_params.get("env_name", ""):
        env_params["env_name"] = env_params["env_name"].replace("Mjx", "")

    th_params = env_params.setdefault("th_params", {})
    th_params["random_start"] = False
    th_params["fixed_start_conf"] = [0, 0]
    th_params["start_from_random_step"] = False

    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    env = TaskFactory.get_factory_cls(config.experiment.task_factory.name).make(**{**env_params, **task_params})
    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    agent_state.train_state.params["log_std"] = np.ones_like(agent_state.train_state.params["log_std"]) * -np.inf
    return BuiltEnv(env=env, config=config, agent_conf=agent_conf, agent_state=agent_state)


def highlight_left_leg(model: mujoco.MjModel, joint_ids: list[int]) -> int:
    left_body_ids = {int(model.jnt_bodyid[jid]) for jid in joint_ids}
    changed = 0
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) in left_body_ids:
            alpha = float(model.geom_rgba[gid, 3])
            model.geom_rgba[gid] = np.asarray([0.95, 0.08, 0.08, alpha], dtype=model.geom_rgba.dtype)
            changed += 1
    return changed


def prosthesis_boundary_actuator_indices(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for name in LEFT_PROSTHESIS_BOUNDARY_MUSCLE_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            ids.append(int(aid))
    return np.asarray(sorted(set(ids)), dtype=np.int32)


def resolve_prosthesis_muscle_scale(args) -> float:
    if args.conflict_muscle_scale is not None:
        print("[record_stage1_testset_visual] WARNING: --conflict-muscle-scale is deprecated.", flush=True)
        return float(args.conflict_muscle_scale)
    return float(args.prosthesis_muscle_scale)


def backup_env_state(env) -> dict:
    data = env.data
    return {
        "time": float(data.time),
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "act": np.asarray(data.act).copy(),
        "ctrl": np.asarray(data.ctrl).copy(),
        "qfrc_applied": np.asarray(data.qfrc_applied).copy(),
        "qacc_warmstart": np.asarray(data.qacc_warmstart).copy(),
        "carry": env._additional_carry,
        "obs": np.asarray(env._obs).copy(),
        "info": dict(env._info),
    }


def restore_env_state(env, state: dict) -> None:
    data = env.data
    data.time = state["time"]
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.act[:] = state["act"]
    data.ctrl[:] = state["ctrl"]
    data.qfrc_applied[:] = state["qfrc_applied"]
    data.qacc_warmstart[:] = state["qacc_warmstart"]
    env._additional_carry = state["carry"]
    env._obs = state["obs"]
    env._info = state["info"]
    mujoco.mj_forward(env.model, env.data)


def oracle_prosthesis_torque(env, oracle_action, qvel_indices: np.ndarray) -> np.ndarray:
    """Ghost-step the full oracle once and return its generalized actuator force on prosthesis DOFs."""

    state = backup_env_state(env)
    env.step(oracle_action)
    torque = np.asarray(env.data.qfrc_actuator[qvel_indices], dtype=np.float64).copy()
    restore_env_state(env, state)
    return torque


def write_web_compatible_mp4(src_path: str) -> str:
    """Re-encode to H.264 baseline + faststart for in-IDE / browser playback."""

    base, ext = os.path.splitext(src_path)
    dst = f"{base}_web{ext or '.mp4'}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-movflags",
        "+faststart",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-an",
        dst,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def copy_data_for_render(model: mujoco.MjModel, src: mujoco.MjData) -> mujoco.MjData:
    dst = mujoco.MjData(model)
    dst.qpos[:] = src.qpos
    dst.qvel[:] = src.qvel
    dst.act[:] = src.act
    dst.ctrl[:] = src.ctrl
    if model.nmocap:
        dst.mocap_pos[:] = src.mocap_pos
        dst.mocap_quat[:] = src.mocap_quat
    mujoco.mj_forward(model, dst)
    return dst


def step_with_stage1_pd(
    env,
    oracle_action,
    target_qpos: np.ndarray,
    qvel_indices: np.ndarray,
    masked_actuators: np.ndarray,
    masked_actuator_scale: float,
    kp: np.ndarray,
    kd: np.ndarray,
    torque_limit: np.ndarray,
    torque_slew_limit: np.ndarray | None = None,
    last_torque: np.ndarray | None = None,
    feedforward_torque: np.ndarray | None = None,
):
    """Replicate env.step while injecting per-substep PD torques on selected DOFs."""

    cur_info = env._info.copy()
    carry = env._additional_carry.replace(last_action=oracle_action)
    processed_action, carry = env._preprocess_action(oracle_action, env._model, env._data, carry)
    env._model, env._data, carry = env._simulation_pre_step(env._model, env._data, carry)

    for _ in range(env._n_intermediate_steps):
        env._data.qfrc_applied[:] = 0.0
        ctrl_action, carry = env._compute_action(processed_action, env._model, env._data, carry)
        env._data.ctrl[env._action_indices] = np.asarray(ctrl_action).reshape(-1)
        if masked_actuators.size:
            scale = float(np.clip(masked_actuator_scale, 0.0, 1.0))
            env._data.ctrl[masked_actuators] *= scale
            if scale <= 0.0:
                actadr = np.asarray(env._model.actuator_actadr[masked_actuators], dtype=np.int32)
                for adr in actadr:
                    if int(adr) >= 0:
                        env._data.act[int(adr)] = 0.0

        q = np.asarray(env._data.qpos[target_qpos.indices], dtype=np.float64)
        qd = np.asarray(env._data.qvel[qvel_indices], dtype=np.float64)
        ff_torque = (
            np.zeros_like(np.asarray(kp, dtype=np.float64))
            if feedforward_torque is None
            else np.asarray(feedforward_torque, dtype=np.float64)
        )
        torque = ff_torque + np.asarray(kp, dtype=np.float64) * (
            np.asarray(target_qpos.values, dtype=np.float64) - q
        ) - np.asarray(kd, dtype=np.float64) * qd
        torque_limit = np.asarray(torque_limit, dtype=np.float64)
        torque = np.clip(torque, -torque_limit, torque_limit)
        if torque_slew_limit is not None and last_torque is not None:
            slew = np.asarray(torque_slew_limit, dtype=np.float64)
            delta = np.clip(torque - last_torque, -slew, slew)
            torque = last_torque + delta
        if last_torque is not None:
            last_torque[:] = torque
        env._data.qfrc_applied[qvel_indices] = torque
        mujoco.mj_step(env._model, env._data, env._n_substeps)
        env._data.qfrc_applied[:] = 0.0

    env._data, carry = env._simulation_post_step(env._model, env._data, carry)
    cur_obs, carry = env._create_observation(env._model, env._data, carry)
    cur_obs, env._data, cur_info, carry = env._step_finalize(cur_obs, env._model, env._data, cur_info, carry)
    cur_info = env._update_info_dictionary(cur_info, cur_obs, env._data, carry)
    absorbing, carry = env._is_absorbing(cur_obs, cur_info, env._data, carry)
    reward, carry = env._reward(env._obs, oracle_action, cur_obs, absorbing, cur_info, env._model, env._data, carry)
    done = env._is_done(cur_obs, absorbing, cur_info, env._data, carry)
    carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
    env._obs = cur_obs
    env._additional_carry = carry
    return np.asarray(cur_obs), reward, absorbing, done, cur_info


class IndexedValues:
    def __init__(self, indices: np.ndarray, values: np.ndarray):
        self.indices = indices
        self.values = values


def _left_foot_site_z(env, audit) -> float:
    for name, sid in audit.sites:
        if "toe" in name.lower() or "ankle" in name.lower():
            return float(env.data.site_xpos[int(sid), 2])
    return float("nan")


def main() -> int:
    args = parse_args()
    muscle_scale = resolve_prosthesis_muscle_scale(args)
    os.makedirs(os.path.dirname(args.record_path), exist_ok=True)
    ckpt = load_pickle(args.policy)
    action_dim = int(ckpt["action_dim"])
    if args.kind == "stage1":
        policy_model = MuscleProKneeTeacher(action_dim=action_dim)
    else:
        policy_model = MuscleProKneeStudent(action_dim=action_dim)
    ckpt_kp = ckpt_vector(ckpt, "pd_kp", action_dim)
    ckpt_kd = ckpt_vector(ckpt, "pd_kd", action_dim)
    joint_kp = (
        ckpt_kp.copy()
        if args.joint_kp is None and ckpt_kp is not None
        else parse_gain_list(args.joint_kp or "300,200,200,120", args.pd_kp, action_dim)
    ) * float(np.clip(args.pd_gain_scale, 1e-6, 10.0))
    joint_kd = (
        ckpt_kd.copy()
        if args.joint_kd is None and ckpt_kd is not None
        else parse_gain_list(args.joint_kd or "30,20,20,12", args.pd_kd, action_dim)
    ) * float(np.clip(args.pd_gain_scale, 1e-6, 10.0))
    max_qpos_target_step = parse_optional_gain_list(args.max_qpos_target_step_list, args.max_qpos_target_step, action_dim)
    if max_qpos_target_step is None and ckpt.get("max_prosthesis_qpos_step", None) is not None:
        max_qpos_target_step = ckpt_vector(ckpt, "max_prosthesis_qpos_step", action_dim)
    torque_limit = parse_optional_gain_list(args.pd_torque_limit_list, None, action_dim)
    if torque_limit is None:
        torque_limit = ckpt_vector(ckpt, "pd_torque_limit", action_dim)
    if torque_limit is None:
        torque_limit = parse_optional_gain_list(None, args.pd_torque_limit, action_dim)
    torque_slew_limit = parse_optional_gain_list(args.pd_torque_slew_limit_list, None, action_dim)
    if torque_slew_limit is None and ckpt.get("pd_torque_slew_limit", None) is not None:
        torque_slew_limit = ckpt_vector(ckpt, "pd_torque_slew_limit", action_dim)
    oracle_torque_ff_scale = (
        float(args.oracle_torque_ff_scale)
        if args.oracle_torque_ff_scale is not None
        else float(ckpt.get("oracle_torque_ff_scale", 0.0))
    )
    oracle_torque_ff_limit = parse_optional_gain_list(args.oracle_torque_ff_limit_list, None, action_dim)
    if oracle_torque_ff_limit is None and ckpt.get("oracle_torque_ff_limit", None) is not None:
        oracle_torque_ff_limit = np.asarray(ckpt["oracle_torque_ff_limit"], dtype=np.float64)
    use_predicted_torque_head = bool(ckpt.get("torque_head", False)) and not args.disable_predicted_torque_head
    predicted_torque_ff_scale = (
        float(args.predicted_torque_ff_scale)
        if args.predicted_torque_ff_scale is not None
        else float(ckpt.get("predicted_torque_ff_scale", 1.0))
    )
    target_mode = str(ckpt.get("target_mode", "residual"))

    hist_placeholder = jnp.zeros(
        (1, int(ckpt.get("history_len", 30)), int(ckpt["obs_dim"])),
        dtype=jnp.float32,
    )

    if args.kind == "stage1":

        @jax.jit
        def policy_predict(params, obs_b, priv_b, hist_b):
            del hist_b
            if use_predicted_torque_head:
                pred, pred_torque, _latent = policy_model.apply(params, obs_b, priv_b, return_torque=True)
                return pred, pred_torque
            pred, _latent = policy_model.apply(params, obs_b, priv_b)
            return pred, jnp.zeros_like(pred)

    else:

        @jax.jit
        def policy_predict(params, obs_b, priv_b, hist_b):
            pred, _latent = policy_model.apply(params, obs_b, priv_b, hist_b, mode="student")
            return pred, jnp.zeros_like(pred)

    def sample_actions(agent_conf):
        @jax.jit
        def _sample(train_state, obs, rng):
            obs_b = jnp.atleast_2d(obs) if hasattr(obs, "ndim") and obs.ndim == 1 else obs
            y, updates = agent_conf.network.apply(
                {"params": train_state.params, "run_stats": train_state.run_stats},
                obs_b,
                mutable=["run_stats"],
            )
            pi, _value = y
            action = pi.sample(seed=rng)
            train_state = train_state.replace(run_stats=updates["run_stats"])
            return jnp.atleast_2d(action), train_state

        return _sample

    motions = args.motion_path or list(KIT_KINESIS_TESTING_MOTIONS[: args.num_motions])
    print(f"Recording ProKnee {args.kind} policy on motions:")
    for i, motion in enumerate(motions, start=1):
        print(f"  {i}. {motion}")
    print(f"Output: {args.record_path}")
    print(f"kind={args.kind} target_mode: {target_mode}")
    if args.pd_override:
        print(
            "Mode: physical PD override "
            f"(joint_kp={joint_kp.tolist()}, joint_kd={joint_kd.tolist()}, "
            f"torque_limit={None if torque_limit is None else torque_limit.tolist()}, "
            f"torque_slew_limit={None if torque_slew_limit is None else torque_slew_limit.tolist()}, "
            f"max_qpos_target_step={None if max_qpos_target_step is None else max_qpos_target_step.tolist()}, "
            f"residual_horizon={args.residual_horizon}, residual_filter={args.residual_filter}, "
            f"mask_boundary_muscles={not args.no_mask_conflict_muscles}, "
            f"prosthesis_muscle_scale={muscle_scale}, "
            f"oracle_torque_ff_scale={oracle_torque_ff_scale}, "
            f"oracle_torque_ff_limit={None if oracle_torque_ff_limit is None else oracle_torque_ff_limit.tolist()}, "
            f"use_predicted_torque_head={use_predicted_torque_head}, "
            f"predicted_torque_ff_scale={predicted_torque_ff_scale})"
        )
    else:
        print(f"Mode: render-only visual audit (oracle physics, {args.kind} red-leg render overlay)")

    rng = jax.random.key(args.seed)
    mse_values: list[float] = []
    deploy_f = None
    deploy_w = None
    if args.deploy_log:
        if not args.pd_override:
            print("[warn] --deploy-log ignored without --pd-override.", flush=True)
        else:
            log_abs = os.path.abspath(args.deploy_log)
            log_dir = os.path.dirname(log_abs)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            deploy_f = open(log_abs, "w", newline="")
            deploy_w = csv.writer(deploy_f)
            deploy_w.writerow(
                [
                    "motion",
                    "step",
                    "done",
                    "ncon",
                    "qvel_maa",
                    "knee_qvel",
                    "ankle_qvel",
                    "subtalar_qvel",
                    "mtp_qvel",
                    "err_cmd_exec_mae",
                    "knee_err_cmd_exec",
                    "ankle_err_cmd_exec",
                    "subtalar_err_cmd_exec",
                    "mtp_err_cmd_exec",
                    "err_pred_exec_mae",
                    "ff_torque_abs_mean",
                    "ff_knee",
                    "ff_ankle",
                    "ff_subtalar",
                    "ff_mtp",
                    "toe_z",
                ]
            )
            print(f"[deploy-log] writing {log_abs}", flush=True)

    try:
        with imageio.get_writer(args.record_path, fps=args.fps, quality=8) as writer:
            for motion_idx, motion in enumerate(motions, start=1):
                built = build_official_eval_env(args.checkpoint, motion)
                env = built.env
                audit = audit_myofullbody_left_leg(env.model)
                changed = highlight_left_leg(env.model, [j.joint_id for j in audit.joints])
                masked_actuators = (
                    np.zeros(0, dtype=np.int32)
                    if args.no_mask_conflict_muscles
                    else prosthesis_boundary_actuator_indices(env.model)
                )
                masked_names = [
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                    for aid in masked_actuators
                ]
                print(
                    f"[{motion_idx}/{len(motions)}] {motion}: highlighted {changed} geoms, "
                    f"masked_actuators={len(masked_actuators)}"
                )
                if args.pd_override and masked_names:
                    print("  masked:", ", ".join(masked_names))

                renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.distance = 6.0
                cam.elevation = -20.0
                cam.azimuth = 90.0

                obs = env.reset()
                obs_buffer = _build_obs_buffer(built.config, env)
                if obs_buffer is not None:
                    obs = obs_buffer.reset(obs)
                proprio_history = ProprioHistory(int(ckpt.get("history_len", 30)), int(ckpt["obs_dim"]))
                train_state = built.agent_state.train_state
                policy_fn = sample_actions(built.agent_conf)
                previous_speed = None
                previous_target = np.zeros(len(audit.joints), dtype=np.float32)
                if target_mode == "qpos":
                    filtered_pred = np.asarray(env.data.qpos[audit.qpos_indices], dtype=np.float32)
                    last_commanded_q = filtered_pred.copy()
                else:
                    filtered_pred = np.zeros(len(audit.joints), dtype=np.float32)
                    last_commanded_q = None
                last_pd_torque = np.zeros(len(audit.joints), dtype=np.float64)
                motion_mse = []

                for step in range(1, args.steps_per_motion + 1):
                    qpos_before = np.asarray(env.data.qpos, dtype=np.float32).copy()
                    stage_obs = build_global_observation(env, audit, previous_speed)
                    proprio_history.append(stage_obs)
                    stage_priv = build_priv_info(env, audit, previous_target, previous_speed)
                    obs_b = jnp.asarray(stage_obs[None, :], dtype=jnp.float32)
                    priv_b = jnp.asarray(stage_priv[None, :], dtype=jnp.float32)
                    hist_b = (
                        hist_placeholder
                        if args.kind == "stage1"
                        else jnp.asarray(proprio_history.as_array()[None, :, :], dtype=jnp.float32)
                    )
                    pred, pred_torque = policy_predict(ckpt["params"], obs_b, priv_b, hist_b)
                    pred_np = np.asarray(pred)[0]
                    pred_torque_np = np.asarray(pred_torque)[0]
                    filt = float(np.clip(args.residual_filter, 0.0, 0.999))
                    filtered_pred = filt * filtered_pred + (1.0 - filt) * pred_np

                    rng, action_key = jax.random.split(rng)
                    action, train_state = policy_fn(train_state, obs, action_key)
                    if target_mode == "qpos":
                        target_qpos = filtered_pred if args.pd_override else pred_np
                    else:
                        residual_for_control = filtered_pred if args.pd_override else pred_np
                        target_qpos = qpos_before[audit.qpos_indices] + float(args.residual_horizon) * residual_for_control
                    target_qpos = np.asarray(
                        [
                            np.clip(value, joint.joint_range[0], joint.joint_range[1])
                            for value, joint in zip(target_qpos, audit.joints, strict=True)
                        ],
                        dtype=np.float32,
                    )
                    if args.pd_override and max_qpos_target_step is not None and last_commanded_q is not None:
                        lim = max_qpos_target_step.astype(np.float64)
                        d = target_qpos.astype(np.float64) - last_commanded_q.astype(np.float64)
                        d = np.clip(d, -lim, lim)
                        target_qpos = (last_commanded_q.astype(np.float64) + d).astype(np.float32)
                        target_qpos = np.asarray(
                            [
                                np.clip(value, joint.joint_range[0], joint.joint_range[1])
                                for value, joint in zip(target_qpos, audit.joints, strict=True)
                            ],
                            dtype=np.float32,
                        )
                        last_commanded_q = target_qpos.copy()
                    elif args.pd_override and target_mode == "qpos":
                        last_commanded_q = target_qpos.copy()
                    if args.pd_override:
                        ff_torque = np.zeros(len(audit.joints), dtype=np.float64)
                        if use_predicted_torque_head:
                            ff_torque = predicted_torque_ff_scale * pred_torque_np.astype(np.float64)
                            if oracle_torque_ff_limit is not None:
                                ff_torque = np.clip(ff_torque, -oracle_torque_ff_limit, oracle_torque_ff_limit)
                        elif oracle_torque_ff_scale != 0.0:
                            ff_torque = oracle_torque_ff_scale * oracle_prosthesis_torque(
                                env, action, audit.qvel_indices
                            )
                            if oracle_torque_ff_limit is not None:
                                ff_torque = np.clip(ff_torque, -oracle_torque_ff_limit, oracle_torque_ff_limit)
                        obs, _reward, _absorbing, done, _info = step_with_stage1_pd(
                            env,
                            action,
                            IndexedValues(audit.qpos_indices, target_qpos),
                            audit.qvel_indices,
                            masked_actuators,
                            muscle_scale,
                            joint_kp,
                            joint_kd,
                            torque_limit,
                            torque_slew_limit,
                            last_pd_torque,
                            ff_torque,
                        )
                    else:
                        obs, _reward, _absorbing, done, _info = env.step(action)
                    if obs_buffer is not None:
                        obs = obs_buffer.step(obs)

                    qpos_after = np.asarray(env.data.qpos, dtype=np.float32).copy()
                    if target_mode == "qpos":
                        executed_target = qpos_after[audit.qpos_indices].astype(np.float32)
                        motion_mse.append(float(np.mean((pred_np - executed_target) ** 2)))
                        previous_target = executed_target
                    else:
                        executed_delta = (qpos_after[audit.qpos_indices] - qpos_before[audit.qpos_indices]).astype(np.float32)
                        motion_mse.append(float(np.mean((pred_np - executed_delta) ** 2)))
                        previous_target = executed_delta
                    previous_speed = float(np.linalg.norm(np.asarray(env.data.qvel[:2], dtype=np.float64)))

                    if deploy_w is not None and args.pd_override:
                        exe4 = qpos_after[audit.qpos_indices].astype(np.float64)
                        qvel4 = np.asarray(env.data.qvel[audit.qvel_indices], dtype=np.float64)
                        err_cmd4 = target_qpos.astype(np.float64) - exe4
                        qv_maa = float(
                            np.mean(np.abs(qvel4))
                        )
                        err_cmd = float(np.mean(np.abs(err_cmd4)))
                        err_pred = float(np.mean(np.abs(pred_np.astype(np.float64) - exe4)))
                        ff_abs = float(np.mean(np.abs(ff_torque)))
                        deploy_w.writerow(
                            [
                                motion,
                                step,
                                int(bool(done)),
                                int(env.data.ncon),
                                f"{qv_maa:.6f}",
                                f"{qvel4[0]:.6f}",
                                f"{qvel4[1]:.6f}",
                                f"{qvel4[2]:.6f}",
                                f"{qvel4[3]:.6f}",
                                f"{err_cmd:.6f}",
                                f"{err_cmd4[0]:.6f}",
                                f"{err_cmd4[1]:.6f}",
                                f"{err_cmd4[2]:.6f}",
                                f"{err_cmd4[3]:.6f}",
                                f"{err_pred:.6f}",
                                f"{ff_abs:.6f}",
                                f"{ff_torque[0]:.6f}",
                                f"{ff_torque[1]:.6f}",
                                f"{ff_torque[2]:.6f}",
                                f"{ff_torque[3]:.6f}",
                                f"{_left_foot_site_z(env, audit):.4f}",
                            ]
                        )
                        if step % 200 == 0:
                            deploy_f.flush()

                    if args.pd_override:
                        render_data = env.data
                    else:
                        render_data = copy_data_for_render(env.model, env.data)
                        for value, joint in zip(target_qpos, audit.joints, strict=True):
                            render_data.qpos[joint.qposadr] = float(value)
                        mujoco.mj_forward(env.model, render_data)
                    cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                    renderer.update_scene(render_data, camera=cam)
                    writer.append_data(renderer.render())

                    if done:
                        break

                mse = float(np.mean(motion_mse)) if motion_mse else float("nan")
                mse_values.append(mse)
                print(f"[{motion_idx}/{len(motions)}] done steps={step} policy_delta_mse={mse:.6f}")
                renderer.close()
                env.stop()

    finally:
        if deploy_f is not None:
            deploy_f.close()

    print(f"Saved video: {args.record_path}")
    print(f"Mean policy delta MSE over motions: {float(np.mean(mse_values)):.6f}")
    if not args.no_web_encode:
        try:
            web_path = write_web_compatible_mp4(args.record_path)
            print(f"Web-compatible copy: {web_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[warn] web encode skipped: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
