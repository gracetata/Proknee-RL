#!/usr/bin/env python
"""Diagnostics for masked-obs full-muscle policy + OSL FSM deployment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import load_fullbody_config, repo_root
from musclemimic.distill.obs_mask import apply_obs_mask, build_masked_obs_spec
from musclemimic.distill.osl_harness import (
    build_osl_harness,
    compare_mask_specs,
    summarize_run_stats,
    summarize_trajectory,
)
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.rollout import safe_motion_filename
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless
from record_masked_obs_full_muscle_replay import find_mask_spec


DEFAULT_MOTION = "KIT/3/walk_6m_straight_line04_poses"
DEFAULT_CHECKPOINT = (
    "outputs/full_muscle_masked_obs_dagger1_focused_round3_distill/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_DATASET_DIR = "data/full_muscle_masked_obs_rollouts/smoke_1motion"
DEFAULT_OUTPUT_DIR = "outputs/_nonformal_runs/osl_diag"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--motion-path", default=DEFAULT_MOTION)
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--scales", default="1.0,0.5,0.2,0.0")
    p.add_argument("--skip-policy", action="store_true")
    p.add_argument("--skip-action-replay", action="store_true")
    p.add_argument("--strict-terminal", action="store_true")
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def build_env(config, motion_path: str, *, strict_terminal: bool):
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
    if not strict_terminal:
        apply_eval_terminal_defaults(env_params, config, strict_termination=False)
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass
    control_dt = apply_temporal_params(config)
    env = TaskFactory.get_factory_cls(config.experiment.task_factory.name).make(**{**env_params, **task_params})
    return env, control_dt


def rollout_policy_osl(env, spec, runner, harness, *, disabled_muscle_scale: float, steps: int) -> list[dict]:
    obs = env.reset()
    harness.reset()
    obs_policy = runner.reset_obs(apply_obs_mask(obs, spec))
    rows: list[dict] = []
    for step in range(steps):
        action, _value = runner.act(obs_policy)
        obs, _reward, done, diag = harness.step(action, disabled_muscle_scale=disabled_muscle_scale, use_osl_torque=True)
        rows.append({"step": step + 1, "done": done, **diag})
        obs_policy = runner.update_obs(apply_obs_mask(obs, spec))
        if done and step < 5:
            break
    return rows


def replay_dataset_actions_osl(env, data, harness, *, disabled_muscle_scale: float, steps: int) -> list[dict]:
    actions = np.asarray(data["target_full_muscle_action"], dtype=np.float32)
    obs = env.reset()
    harness.reset()
    rows: list[dict] = []
    n = min(steps, int(actions.shape[0]))
    for step in range(n):
        obs, _reward, done, diag = harness.step(
            actions[step], disabled_muscle_scale=disabled_muscle_scale, use_osl_torque=True
        )
        rows.append({"step": step + 1, "done": done, **diag})
        if done and step < 5:
            break
    return rows


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve(args.checkpoint)
    dataset_dir = resolve(args.dataset_dir)
    npz_path = dataset_dir / safe_motion_filename(args.motion_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset npz not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as npz:
        data = {k: np.asarray(npz[k]) for k in npz.files}

    config, agent_state, metadata = load_checkpoint(str(checkpoint))
    env, control_dt = build_env(config, args.motion_path, strict_terminal=args.strict_terminal)
    checkpoint_spec = find_mask_spec(checkpoint, env)
    env_spec = build_masked_obs_spec(env)
    harness = build_osl_harness(env)

    initial_obs = env.reset()
    obs0 = apply_obs_mask(initial_obs, checkpoint_spec)
    dataset_obs0 = np.asarray(data["obs_student_masked"][0], dtype=np.float32).reshape(-1)
    obs0_l2 = float(np.linalg.norm(obs0 - dataset_obs0))
    obs0_linf = float(np.max(np.abs(obs0 - dataset_obs0)))

    env_view_spec = checkpoint_spec
    from musclemimic.distill.obs_mask import MaskedObservationEnvView

    env_view = MaskedObservationEnvView(env, env_view_spec)
    agent_conf = PPOJax.init_agent_conf(env_view, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    runner = PolicyRunner.from_agent_state(agent_conf, agent_state, env_view, deterministic=True, seed=0)

    scales = [float(x.strip()) for x in str(args.scales).split(",") if x.strip()]
    policy_by_scale: dict[str, dict] = {}
    action_replay_by_scale: dict[str, dict] = {}

    for scale in scales:
        key = f"scale_{scale:g}"
        if not args.skip_policy:
            env.reset()
            harness.reset()
            runner.reset_obs(apply_obs_mask(env.reset(), checkpoint_spec))
            rows = rollout_policy_osl(env, checkpoint_spec, runner, harness, disabled_muscle_scale=scale, steps=args.steps)
            policy_by_scale[key] = summarize_trajectory(rows)
            policy_by_scale[key]["disabled_muscle_scale"] = scale
        if not args.skip_action_replay:
            env.reset()
            harness.reset()
            rows = replay_dataset_actions_osl(env, data, harness, disabled_muscle_scale=scale, steps=args.steps)
            action_replay_by_scale[key] = summarize_trajectory(rows)
            action_replay_by_scale[key]["disabled_muscle_scale"] = scale

    terminal_compare = {}
    if not args.strict_terminal:
        env.reset()
        harness.reset()
        runner.reset_obs(apply_obs_mask(env.reset(), checkpoint_spec))
        rows_no_term = rollout_policy_osl(
            env, checkpoint_spec, runner, harness, disabled_muscle_scale=0.0, steps=min(args.steps, 150)
        )
        env_term, _ = build_env(config, args.motion_path, strict_terminal=True)
        harness_term = build_osl_harness(env_term)
        env_view_term = MaskedObservationEnvView(env_term, checkpoint_spec)
        agent_conf_term = PPOJax.init_agent_conf(env_view_term, config)
        agent_state_term = align_agent_state(agent_state, agent_conf_term)
        runner_term = PolicyRunner.from_agent_state(
            agent_conf_term, agent_state_term, env_view_term, deterministic=True, seed=0
        )
        runner_term.reset_obs(apply_obs_mask(env_term.reset(), checkpoint_spec))
        rows_term = rollout_policy_osl(
            env_term, checkpoint_spec, runner_term, harness_term, disabled_muscle_scale=0.0, steps=min(args.steps, 150)
        )
        terminal_compare = {
            "no_terminal_handler": summarize_trajectory(rows_no_term),
            "validation_terminal": summarize_trajectory(rows_term),
        }
        env_term.stop()

    summary = {
        "motion_path": args.motion_path,
        "checkpoint": str(checkpoint),
        "dataset_npz": str(npz_path),
        "control_dt": control_dt,
        "mask_spec": compare_mask_specs(checkpoint_spec, env_spec),
        "obs0_l2_dataset_vs_env_reset": obs0_l2,
        "obs0_linf_dataset_vs_env_reset": obs0_linf,
        "obs_normalizer": {
            "len_obs_history": int(getattr(config.experiment, "len_obs_history", 1)),
            "split_goal": bool(getattr(config.experiment, "split_goal", False)),
            "run_stats": summarize_run_stats(agent_state.train_state.run_stats),
        },
        "policy_rollout_by_scale": policy_by_scale,
        "dataset_action_replay_by_scale": action_replay_by_scale,
        "terminal_compare_scale_0": terminal_compare,
        "checkpoint_metadata": str(metadata),
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
