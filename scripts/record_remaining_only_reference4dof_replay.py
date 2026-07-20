#!/usr/bin/env python
"""Record replay for a remaining-only policy with reference 4-DoF PD."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.distill.mapping import DistillMapping, build_distill_mapping
from musclemimic.distill.obs_mask import MaskedObsSpec, build_masked_obs_spec
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.remaining_only import RemainingOnlyEnvView, build_reference4dof_harness
from musclemimic.evaluation.video_encode import encode_web_mp4, web_mp4_path
from musclemimic.runner.eval_utils import align_agent_state, apply_temporal_params, load_checkpoint, setup_headless

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_CHECKPOINT = "outputs/remaining_only_reference4dof_distill/latest/checkpoints/checkpoint_distilled"
DEFAULT_OUTPUT_DIR = "outputs/_nonformal_runs/remaining_only_reference4dof_replay"
DEFAULT_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--motion-path", action="append", default=[])
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-steps", type=int, default=0, help="0 = full trajectory length")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--reference-pd-kp", default="240,180,60,40")
    p.add_argument("--reference-pd-kd", default="24,18,6,4")
    p.add_argument("--reference-torque-limit", default="140,120,60,60")
    p.add_argument("--torque-slew-limit", type=float, default=35.0)
    p.add_argument("--reference-control-mode", choices=["pd", "lock"], default="pd")
    p.add_argument("--keep-raw", action="store_true")
    return p.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def parse_vec4(value: str) -> tuple[float, float, float, float]:
    vals = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if len(vals) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got {value!r}")
    return vals


def _candidate_parent_paths(checkpoint: Path) -> list[Path]:
    return [checkpoint, checkpoint.parent, checkpoint.parent.parent, checkpoint.parent.parent.parent]


def load_mapping(checkpoint: Path, env, student_env) -> DistillMapping:
    for base in _candidate_parent_paths(checkpoint):
        path = base / "mapping.json"
        if path.exists():
            return DistillMapping.load_json(path)
    return build_distill_mapping(env, student_env)


def load_mask_spec(checkpoint: Path, env) -> MaskedObsSpec:
    for base in _candidate_parent_paths(checkpoint):
        path = base / "masked_obs_spec.json"
        if path.exists():
            return MaskedObsSpec.load_json(path)
        meta_path = base / "distilled_metadata.json"
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            spec_payload = payload.get("masked_obs_spec")
            if spec_payload:
                return MaskedObsSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec_payload.items()})
    return build_masked_obs_spec(env)


def build_envs(config, motion_path: str):
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
    env = factory.make(**{**env_params, **task_params})
    student_env = factory.make(
        **{**env_params, "env_name": "MyoFullBodyProsthesisEnv"},
        **task_params,
        prosthesis={"enabled": True},
    )
    return env, student_env


def record_motion(args, checkpoint: Path, config, agent_state, metadata, motion_path: str, output_dir: Path, control_dt: float) -> dict:
    env, student_env = build_envs(config, motion_path)
    try:
        mapping = load_mapping(checkpoint, env, student_env)
        spec = load_mask_spec(checkpoint, env)
        env_view = RemainingOnlyEnvView(env, spec, mapping.n_remaining_muscles)
        policy_obs_dim = int(env_view.info.observation_space.shape[0])
        agent_conf = PPOJax.init_agent_conf(env_view, config)
        aligned_state = align_agent_state(agent_state, agent_conf)
        runner = PolicyRunner.from_agent_state(
            agent_conf,
            aligned_state,
            env_view,
            deterministic=not args.stochastic,
            seed=args.seed,
        )
        harness = build_reference4dof_harness(
            env,
            mapping,
            spec,
            kp=parse_vec4(args.reference_pd_kp),
            kd=parse_vec4(args.reference_pd_kd),
            torque_limit=parse_vec4(args.reference_torque_limit),
            torque_slew_limit=float(args.torque_slew_limit),
            mode=str(args.reference_control_mode),
        )
        n_steps = int(args.n_steps) if int(args.n_steps) > 0 else int(env.th.len_trajectory(0))
        fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
        safe_motion = motion_path.replace("/", "_")
        motion_dir = output_dir / "per_motion" / safe_motion
        motion_dir.mkdir(parents=True, exist_ok=True)
        raw_path = motion_dir / f".{safe_motion}_remaining_only_reference4dof_raw.mp4"
        video_path = web_mp4_path(motion_dir / f"{safe_motion}_remaining_only_reference4dof.mp4")
        log_path = motion_dir / "remaining_only_reference4dof.csv"
        meta_path = motion_dir / "remaining_only_reference4dof_meta.json"

        renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 6.0
        cam.elevation = -20.0
        cam.azimuth = 90.0

        obs_policy = runner.reset_obs(harness.reset())
        episode_return = 0.0
        done_count = 0
        first_fall_step = None
        disabled_ctrl_max = 0.0
        prosthesis_ref_error_max = 0.0
        prosthesis_ref_error_rms_sum = 0.0
        rows = 0
        with log_path.open("w", newline="", encoding="utf-8") as log_f:
            writer_csv = csv.writer(log_f)
            writer_csv.writerow(
                [
                    "step",
                    "root_height",
                    "done",
                    "disabled_ctrl_norm",
                    "knee_q",
                    "ankle_q",
                    "subtalar_q",
                    "mtp_q",
                    "knee_ref",
                    "ankle_ref",
                    "subtalar_ref",
                    "mtp_ref",
                    "knee_tau",
                    "ankle_tau",
                    "subtalar_tau",
                    "mtp_tau",
                    "prosthesis_ref_error_max",
                    "prosthesis_ref_error_rms",
                ]
            )
            with imageio.get_writer(str(raw_path), fps=fps, quality=8) as writer:
                for step in range(n_steps):
                    action, _value = runner.act(obs_policy)
                    obs_policy_next, reward, done, diag = harness.step(action)
                    obs_policy = runner.update_obs(obs_policy_next)
                    episode_return += float(reward)
                    done_count += int(bool(done))
                    root_height = float(diag["root_height"])
                    if first_fall_step is None and root_height < 0.8:
                        first_fall_step = step + 1
                    disabled_ctrl_max = max(disabled_ctrl_max, float(diag["disabled_ctrl_norm"]))
                    err_max = float(diag.get("prosthesis_ref_error_max", 0.0))
                    err_rms = float(diag.get("prosthesis_ref_error_rms", 0.0))
                    prosthesis_ref_error_max = max(prosthesis_ref_error_max, err_max)
                    prosthesis_ref_error_rms_sum += err_rms
                    q = np.asarray(diag["prosthesis_qpos"], dtype=np.float32)
                    ref = np.asarray(diag["ref_prosthesis_qpos"], dtype=np.float32)
                    tau = np.asarray(diag["prosthesis_tau"], dtype=np.float32)
                    writer_csv.writerow(
                        [
                            step + 1,
                            f"{root_height:.6f}",
                            int(bool(done)),
                            f"{float(diag['disabled_ctrl_norm']):.9f}",
                            *[f"{float(x):.6f}" for x in q],
                            *[f"{float(x):.6f}" for x in ref],
                            *[f"{float(x):.6f}" for x in tau],
                            f"{err_max:.9f}",
                            f"{err_rms:.9f}",
                        ]
                    )
                    rows += 1
                    cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                    renderer.update_scene(env.data, camera=cam)
                    writer.append_data(renderer.render())
        renderer.close()
        encode_web_mp4(raw_path, video_path, remove_src=not args.keep_raw)
        meta = {
            "replay_mode": "remaining_only_reference4dof",
            "checkpoint": str(checkpoint),
            "motion_path": motion_path,
            "n_steps": n_steps,
            "recorded_steps": rows,
            "fps": fps,
            "duration_s": float(rows * control_dt),
            "episode_return": float(episode_return),
            "done_count": int(done_count),
            "first_fall_step_lt_0.8m": first_fall_step,
            "disabled_ctrl_max_norm": float(disabled_ctrl_max),
            "prosthesis_ref_error_max": float(prosthesis_ref_error_max),
            "prosthesis_ref_error_rms_mean": float(prosthesis_ref_error_rms_sum / max(rows, 1)),
            "masked_obs_dim": int(spec.masked_obs_dim),
            "policy_obs_dim": policy_obs_dim,
            "prosthesis_state_feature_dim": int(policy_obs_dim - spec.masked_obs_dim),
            "remaining_action_dim": int(mapping.n_remaining_muscles),
            "reference_control_mode": str(args.reference_control_mode),
            "reference_pd_kp": parse_vec4(args.reference_pd_kp),
            "reference_pd_kd": parse_vec4(args.reference_pd_kd),
            "reference_torque_limit": parse_vec4(args.reference_torque_limit),
            "video": str(video_path),
            "deploy_log_csv": str(log_path),
            "distill_metadata": metadata,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(
            f"{motion_path}: first_fall={first_fall_step} return={episode_return:.3f} "
            f"done_count={done_count} video={video_path}"
        )
        return meta
    finally:
        env.stop()
        student_env.stop()


def main() -> int:
    args = parse_args()
    setup_headless(argparse.Namespace(no_render=True, mujoco_viewer=False, viser_viewer=False))
    checkpoint = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, agent_state, metadata = load_checkpoint(str(checkpoint))
    control_dt = apply_temporal_params(config)
    motions = args.motion_path or DEFAULT_MOTIONS
    records = [
        record_motion(args, checkpoint, config, agent_state, metadata, motion, output_dir, control_dt)
        for motion in motions
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "motions": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
