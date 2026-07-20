#!/usr/bin/env python
"""Sweep prosthesis PD + execution smoothing on a fixed checkpoint (no video)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import apply_prosthesis_overrides, parse_optional_vec4, repo_root
from musclemimic.distill.policy import PolicyRunner
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint


DEFAULT_CKPT = "outputs/prosthesis_distill_knee15_kit3_focus/2026-06-13/20-11-40/checkpoints/checkpoint_distilled"
DEFAULT_MOTION = "KIT/3/walk_6m_straight_line04_poses"


@dataclass(frozen=True)
class Variant:
    name: str
    kp: str | None = None
    kd: str | None = None
    slew: str | None = None
    lowpass: float | None = None


VARIANTS = [
    Variant("baseline_default_pd"),
    Variant("slew35", slew="35"),
    Variant("soft_ankle_pd", kp="240,80,25,30", kd="24,8,3,3"),
    Variant("soft_ankle_slew35", kp="240,80,25,30", kd="24,8,3,3", slew="35"),
    Variant("smooth_yaml_like", kp="240,80,25,30", kd="24,8,3,3", slew="60,35,12,12", lowpass=0.35),
    Variant("soft_ankle_lowpass", kp="240,80,25,30", kd="24,8,3,3", slew="35", lowpass=0.5),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--motion-path", default=DEFAULT_MOTION)
    p.add_argument("--output-dir", default="outputs/_nonformal_runs/knee15_pd_execution_sweep")
    p.add_argument("--steps", type=int, default=736)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def parse_slew(value: str | None):
    if value is None:
        return None
    parts = [float(x.strip()) for x in value.split(",") if x.strip()]
    return tuple(parts) if len(parts) == 4 else float(parts[0])


def rollout(checkpoint: Path, motion_path: str, variant: Variant, steps: int, seed: int) -> dict:
    config, agent_state, metadata = load_checkpoint(str(checkpoint))
    OmegaConf.set_struct(config, False)
    apply_prosthesis_overrides(
        config,
        action_type="pd_residual_torque",
        residual_pd_kp=parse_optional_vec4(variant.kp),
        residual_pd_kd=parse_optional_vec4(variant.kd),
        torque_slew_limit=parse_slew(variant.slew),
        tau_lowpass_alpha=variant.lowpass,
    )
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    env_params["env_name"] = "MyoFullBodyProsthesisEnv"
    env_params["headless"] = True
    env_params["terminal_state_type"] = "NoTerminalStateHandler"
    prosthesis = dict(env_params.get("prosthesis", {}) or {})
    prosthesis["enabled"] = True
    prosthesis["control_mode"] = "train_policy"
    env_params["prosthesis"] = prosthesis
    goal_params = dict(env_params.get("goal_params", {}) or {})
    goal_params["visualize_goal"] = False
    goal_params["n_visual_geoms"] = 0
    env_params["goal_params"] = goal_params
    apply_eval_terminal_defaults(env_params, config, strict_termination=False)
    th_params = dict(env_params.get("th_params", {}) or {})
    th_params.update({"random_start": False, "fixed_start_conf": [0, 0], "start_from_random_step": False})
    env_params["th_params"] = th_params
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    amass = dict(task_params.get("amass_dataset_conf", {}) or {})
    amass["rel_dataset_path"] = [motion_path]
    amass["dataset_group"] = None
    task_params["amass_dataset_conf"] = amass

    apply_temporal_params(config)
    env = TaskFactory.get_factory_cls(config.experiment.task_factory.name).make(**{**env_params, **task_params})
    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    runner = PolicyRunner.from_agent_state(agent_conf, agent_state, env, deterministic=True, seed=seed)

    obs = env.reset()
    obs_policy = runner.reset_obs(obs)
    episode_return = 0.0
    done_count = 0
    tau_log: list[np.ndarray] = []
    q_log: list[np.ndarray] = []
    n = min(steps, int(env.th.len_trajectory(0)))
    for _ in range(n):
        action, _ = runner.act(obs_policy)
        obs, reward, _abs, done, info = env.step(action)
        obs_policy = runner.update_obs(obs)
        episode_return += float(np.asarray(reward).item())
        done_count += int(bool(done))
        tau_log.append(np.asarray(info.get("prosthesis_tau", np.zeros(4)), dtype=np.float32))
        q_log.append(np.asarray(info.get("prosthesis_q", np.zeros(4)), dtype=np.float32))

    tau = np.stack(tau_log, axis=0)
    q = np.stack(q_log, axis=0)
    names = ["knee", "ankle", "subtalar", "mtp"]
    tau_stats = {f"{n}_tau_std": float(np.std(tau[:, i])) for i, n in enumerate(names)}
    q_stats = {f"{n}_q_std": float(np.std(q[:, i])) for i, n in enumerate(names)}
    env.stop()
    return {
        "variant": variant.name,
        "kp": variant.kp,
        "kd": variant.kd,
        "slew": variant.slew,
        "lowpass": variant.lowpass,
        "episode_return": episode_return,
        "done_count": done_count,
        "steps": n,
        **tau_stats,
        **q_stats,
        "ankle_tau_rate_std": float(np.std(np.diff(tau[:, 1]))) if tau.shape[0] > 1 else 0.0,
        "metadata": str(metadata),
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve(args.checkpoint)
    rows = []
    for variant in VARIANTS:
        print(f"Running {variant.name}...", flush=True)
        row = rollout(checkpoint, args.motion_path, variant, args.steps, args.seed)
        rows.append(row)
        print(
            f"  return={row['episode_return']:.1f} done={row['done_count']} "
            f"ankle_tau_std={row['ankle_tau_std']:.2f} ankle_rate_std={row['ankle_tau_rate_std']:.2f}",
            flush=True,
        )
    (out_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
