"""Evaluate MyoFullBody prosthesis policies or external prosthesis controllers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.rollout import contact_load_by_bodies
from musclemimic.prosthesis.controllers import FSMImpedanceProsthesisController, ReferencePDProsthesisController
from musclemimic.runner.eval_utils import align_agent_state, apply_trajectory_selection, load_checkpoint, setup_headless
from loco_mujoco.task_factories import TaskFactory

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Checkpoint path")
    parser.add_argument("--motion_path", type=str, nargs="+", default=None)
    parser.add_argument("--motion_group", type=str, default=None)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--use_mujoco", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--mujoco_viewer", action="store_true")
    parser.add_argument("--viser_viewer", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--metrics_only", action="store_true")
    parser.add_argument("--train_state_seed", type=int, default=0)
    parser.add_argument("--traj_index", type=int, default=None)
    parser.add_argument("--traj_start_step", type=int, default=None)
    parser.add_argument(
        "--prosthesis_control_mode",
        choices=["eval_policy", "eval_external_controller"],
        default="eval_policy",
    )
    parser.add_argument(
        "--prosthesis_controller",
        choices=["reference_pd", "fsm_impedance"],
        default="reference_pd",
    )
    return parser.parse_args()


def resolve_checkpoint_path(path: str) -> str:
    p = Path(path)
    if p.is_dir() and (p / "checkpoint_path.txt").is_file():
        return (p / "checkpoint_path.txt").read_text(encoding="utf-8").strip()
    if p.is_file() and p.suffix in {"", ".txt"}:
        text = p.read_text(encoding="utf-8").strip()
        if text and Path(text).exists():
            q = Path(text)
            if q.is_dir() and (q / "checkpoint_path.txt").is_file():
                return (q / "checkpoint_path.txt").read_text(encoding="utf-8").strip()
            return text
    return path


def build_controller(name: str):
    if name == "reference_pd":
        return ReferencePDProsthesisController()
    if name == "fsm_impedance":
        return FSMImpedanceProsthesisController()
    raise ValueError(f"Unknown prosthesis controller: {name}")


def run_mujoco_metrics(env, agent_conf, agent_state, args):
    runner = PolicyRunner.from_agent_state(
        agent_conf,
        agent_state,
        env,
        deterministic=not args.stochastic,
        seed=0,
    )
    obs = env.reset()
    obs_policy = runner.reset_obs(obs)
    episode_return = 0.0
    torques = []
    torque_deltas = []
    joint_errs = []
    disabled_ctrl = []
    disabled_force = []
    site_errs = []
    root_errs = []
    grfs = []
    prev_tau = None
    done_reason = "max_steps"
    steps = 0

    for step in range(int(args.n_steps)):
        action, _ = runner.act(obs_policy)
        obs, reward, _absorbing, done, info = env.step(action)
        obs_policy = runner.update_obs(obs)
        episode_return += float(np.asarray(reward).item())
        steps = step + 1

        prosthesis_obs = env.get_prosthesis_obs() if hasattr(env, "get_prosthesis_obs") else {}
        tau = np.asarray(info.get("prosthesis_tau", prosthesis_obs.get("prev_tau", np.zeros(4))), dtype=np.float32)
        torques.append(tau)
        if prev_tau is not None:
            torque_deltas.append(tau - prev_tau)
        prev_tau = tau
        if "q" in prosthesis_obs and "ref_q" in prosthesis_obs:
            joint_errs.append(np.asarray(prosthesis_obs["q"]) - np.asarray(prosthesis_obs["ref_q"]))
        body_ids = tuple(getattr(env, "_prosthesis_body_ids", ()))
        if body_ids:
            grf_pair, _contact_pair = contact_load_by_bodies(env.model, env.data, body_ids)
            grfs.append(grf_pair)
        elif "grf" in prosthesis_obs:
            grfs.append(np.asarray([float(prosthesis_obs["grf"]), np.nan], dtype=np.float32))
        disabled_ctrl.append(float(info.get("disabled_muscle_ctrl_norm", 0.0)))
        disabled_force.append(float(info.get("disabled_muscle_force_norm", 0.0)))
        for key, value in info.items():
            if key.startswith("err_site") or key in {"err_rpos", "err_site_abs"}:
                site_errs.append(float(value))
            if key.startswith("err_root"):
                root_errs.append(float(value))

        if not args.no_render or args.record:
            env.render(record=args.record)
        if bool(done):
            done_reason = "terminated" if bool(info.get("terminated", False)) else "done"
            break

    torques_arr = np.asarray(torques, dtype=np.float32) if torques else np.zeros((0, 4), dtype=np.float32)
    dtau_arr = np.asarray(torque_deltas, dtype=np.float32) if torque_deltas else np.zeros((0, 4), dtype=np.float32)
    joint_arr = np.asarray(joint_errs, dtype=np.float32) if joint_errs else np.zeros((0, 4), dtype=np.float32)
    grf_arr = np.asarray(grfs, dtype=np.float32) if grfs else np.zeros((0, 2), dtype=np.float32)
    if grf_arr.size:
        left_grf = grf_arr[:, 0]
        right_grf = grf_arr[:, 1]
        denom = np.maximum(left_grf + right_grf, 1e-6)
        grf_sym = 1.0 - np.abs(left_grf - right_grf) / denom
    else:
        left_grf = right_grf = grf_sym = np.asarray([], dtype=np.float32)
    metrics = {
        "episode_return": episode_return,
        "episode_length": steps,
        "done_reason": done_reason,
        "joint_angle_error": float(np.sqrt(np.mean(joint_arr**2))) if joint_arr.size else 0.0,
        "site_position_error": float(np.mean(site_errs)) if site_errs else 0.0,
        "root_error": float(np.mean(root_errs)) if root_errs else 0.0,
        "prosthesis_torque_rms": float(np.sqrt(np.mean(torques_arr**2))) if torques_arr.size else 0.0,
        "prosthesis_torque_smoothness": float(np.sqrt(np.mean(dtau_arr**2))) if dtau_arr.size else 0.0,
        "left_grf_mean": float(np.mean(left_grf)) if left_grf.size else 0.0,
        "right_grf_mean": float(np.mean(right_grf)) if right_grf.size else 0.0,
        "left_right_grf_symmetry": float(np.nanmean(grf_sym)) if grf_sym.size else float("nan"),
        "disabled_muscle_ctrl_max": float(np.max(disabled_ctrl)) if disabled_ctrl else 0.0,
        "disabled_muscle_force_norm": float(np.max(disabled_force)) if disabled_force else 0.0,
    }
    print("\n=== PROSTHESIS CLOSED-LOOP METRICS ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


def main() -> int:
    args = parse_args()
    setup_headless(args)
    checkpoint_path = resolve_checkpoint_path(args.path)
    config, agent_state, _metadata = load_checkpoint(checkpoint_path)
    OmegaConf.set_struct(config, False)

    env_params = config.experiment.env_params
    env_params["env_name"] = "MyoFullBodyProsthesisEnv" if args.use_mujoco else "MjxMyoFullBodyProsthesisEnv"
    env_params["headless"] = args.no_render
    if args.prosthesis_control_mode == "eval_external_controller":
        args.use_mujoco = True
        env_params["env_name"] = "MyoFullBodyProsthesisEnv"
    prosthesis = dict(env_params.get("prosthesis", {}) or {})
    prosthesis["enabled"] = True
    prosthesis["control_mode"] = args.prosthesis_control_mode
    env_params["prosthesis"] = prosthesis

    if args.motion_path is not None:
        task_params = config.experiment.task_factory.params
        amass_conf = dict(task_params.get("amass_dataset_conf", {}) or {})
        amass_conf["rel_dataset_path"] = args.motion_path
        amass_conf["dataset_group"] = None
        task_params["amass_dataset_conf"] = amass_conf
    elif args.motion_group is not None:
        task_params = config.experiment.task_factory.params
        amass_conf = dict(task_params.get("amass_dataset_conf", {}) or {})
        amass_conf["dataset_group"] = args.motion_group
        amass_conf.pop("rel_dataset_path", None)
        task_params["amass_dataset_conf"] = amass_conf

    apply_trajectory_selection(config, args.traj_index, args.traj_start_step)

    controller = None
    if args.prosthesis_control_mode == "eval_external_controller":
        controller = build_controller(args.prosthesis_controller)

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    merged_params = {
        **OmegaConf.to_container(config.experiment.env_params, resolve=True),
        **OmegaConf.to_container(config.experiment.task_factory.params, resolve=True),
    }
    if controller is not None:
        merged_params["prosthesis_controller"] = controller
    env = factory.make(**merged_params)

    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    if args.use_mujoco and not args.mujoco_viewer:
        run_mujoco_metrics(env, agent_conf, agent_state, args)
    elif args.use_mujoco:
        PPOJax.play_policy_mujoco(
            env,
            agent_conf,
            agent_state,
            deterministic=not args.stochastic,
            n_steps=args.n_steps,
            render=not args.no_render or args.record,
            record=args.record,
            train_state_seed=args.train_state_seed,
        )
    else:
        PPOJax.play_policy(
            env,
            agent_conf,
            agent_state,
            deterministic=not args.stochastic,
            n_steps=args.n_steps,
            n_envs=args.num_envs,
            render=not args.no_render or args.record,
            record=args.record,
            train_state_seed=args.train_state_seed,
            sequential_mjx=args.num_envs == 1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
