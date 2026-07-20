#!/usr/bin/env python
"""Collect remaining-muscle-only distillation rollouts with reference 4-DoF PD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from musclemimic.distill.config import filter_motions, load_fullbody_config, make_env, motion_list_from_group, repo_root
from musclemimic.distill.mapping import build_distill_mapping
from musclemimic.distill.obs_mask import build_masked_obs_spec
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.remaining_only import (
    _reference_prosthesis_state,
    build_reference4dof_harness,
    teacher_action_to_remaining,
)
from train_prosthesis_distillation import DEFAULT_TEACHER_CHECKPOINT

DEFAULT_OUTPUT_DIR = "data/remaining_only_reference4dof/KIT_KINESIS_TRAINING_MOTIONS"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    p.add_argument("--dataset_group", default="KIT_KINESIS_TRAINING_MOTIONS")
    p.add_argument("--motion_path", nargs="*", default=None)
    p.add_argument("--motion_filter", default="none")
    p.add_argument("--max_motions", type=int, default=None)
    p.add_argument("--n_steps_per_motion", default="auto")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    p.add_argument("--reference_pd_kp", default="240,180,60,40")
    p.add_argument("--reference_pd_kd", default="24,18,6,4")
    p.add_argument("--reference_torque_limit", default="140,120,60,60")
    p.add_argument("--torque_slew_limit", type=float, default=35.0)
    p.add_argument("--reference_control_mode", choices=["pd", "lock"], default="pd")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stop_on_done", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def parse_vec4(value: str) -> tuple[float, float, float, float]:
    vals = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if len(vals) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got {value!r}")
    return vals


def resolve_steps(value: str) -> int | None:
    return None if str(value).lower() == "auto" else int(value)


def safe_motion_filename(motion_path: str) -> str:
    return motion_path.replace("/", "_") + ".npz"


def collect_motion(
    *,
    cfg,
    teacher_checkpoint: str,
    motion_path: str,
    output_dir: Path,
    n_steps: int | None,
    kp: tuple[float, float, float, float],
    kd: tuple[float, float, float, float],
    torque_limit: tuple[float, float, float, float],
    torque_slew_limit: float,
    reference_control_mode: str,
    seed: int,
    stop_on_done: bool,
) -> dict:
    env = make_env(cfg, env_name="MyoFullBody", motion_paths=[motion_path], use_mujoco=True, fixed_start=True)
    student_env = make_env(
        cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[motion_path], use_mujoco=True, fixed_start=True
    )
    try:
        mapping = build_distill_mapping(env, student_env)
        spec = build_masked_obs_spec(env)
        teacher = PolicyRunner.from_checkpoint(teacher_checkpoint, env, deterministic=True, seed=seed)
        harness = build_reference4dof_harness(
            env,
            mapping,
            spec,
            kp=kp,
            kd=kd,
            torque_limit=torque_limit,
            torque_slew_limit=torque_slew_limit,
            mode=reference_control_mode,
        )
        obs_student = harness.reset()
        obs_teacher_policy = teacher.reset_obs(np.asarray(env._obs, dtype=np.float32))
        max_steps = int(n_steps or env.th.len_trajectory(0))

        rows: dict[str, list] = {
            "obs_student": [],
            "obs_policy": [],
            "target_remaining_muscle_action": [],
            "ref_prosthesis_qpos": [],
            "ref_prosthesis_qvel": [],
            "root_state": [],
            "prosthesis_qpos": [],
            "prosthesis_qvel": [],
            "prosthesis_tau": [],
            "prosthesis_ref_error_max": [],
            "prosthesis_ref_error_rms": [],
            "done": [],
        }
        total_return = 0.0
        ref_error_max = 0.0
        ref_error_rms_sum = 0.0
        for _step in range(max_steps):
            teacher_action, _ = teacher.act(obs_teacher_policy)
            target_remaining = teacher_action_to_remaining(teacher_action, mapping)
            ref_q, ref_qd = _reference_prosthesis_state(env, harness.qpos_indices, harness.qvel_indices)
            rows["obs_student"].append(np.asarray(obs_student, dtype=np.float32).reshape(-1))
            rows["obs_policy"].append(np.asarray(obs_student, dtype=np.float32).reshape(-1))
            rows["target_remaining_muscle_action"].append(target_remaining.astype(np.float32))
            rows["ref_prosthesis_qpos"].append(ref_q.astype(np.float32))
            rows["ref_prosthesis_qvel"].append(ref_qd.astype(np.float32))
            rows["root_state"].append(np.concatenate([env.data.qpos[:7], env.data.qvel[:6]]).astype(np.float32))
            rows["prosthesis_qpos"].append(np.asarray(env.data.qpos[harness.qpos_indices], dtype=np.float32))
            rows["prosthesis_qvel"].append(np.asarray(env.data.qvel[harness.qvel_indices], dtype=np.float32))

            obs_student, reward, done, diag = harness.step(target_remaining)
            obs_teacher_policy = teacher.update_obs(np.asarray(env._obs, dtype=np.float32))
            total_return += float(reward)
            rows["prosthesis_tau"].append(np.asarray(diag["prosthesis_tau"], dtype=np.float32))
            err_max = float(diag.get("prosthesis_ref_error_max", 0.0))
            err_rms = float(diag.get("prosthesis_ref_error_rms", 0.0))
            ref_error_max = max(ref_error_max, err_max)
            ref_error_rms_sum += err_rms
            rows["prosthesis_ref_error_max"].append(np.asarray(err_max, dtype=np.float32))
            rows["prosthesis_ref_error_rms"].append(np.asarray(err_rms, dtype=np.float32))
            rows["done"].append(np.asarray(bool(done), dtype=np.bool_))
            if done and stop_on_done:
                break

        arrays = {key: np.asarray(value) for key, value in rows.items()}
        arrays["motion_path"] = np.asarray(motion_path)
        arrays["mask_keep_indices"] = np.asarray(spec.keep_indices, dtype=np.int32)
        arrays["mask_removed_indices"] = np.asarray(spec.removed_indices, dtype=np.int32)
        arrays["policy_obs_dim"] = np.asarray(arrays["obs_policy"].shape[-1], dtype=np.int32)
        arrays["masked_obs_dim"] = np.asarray(spec.masked_obs_dim, dtype=np.int32)
        arrays["prosthesis_state_feature_dim"] = np.asarray(16, dtype=np.int32)
        arrays["reference_control_mode"] = np.asarray(reference_control_mode)
        out_path = output_dir / safe_motion_filename(motion_path)
        np.savez_compressed(out_path, **arrays)
        n_recorded = max(int(arrays["obs_student"].shape[0]), 1)
        return {
            "path": str(out_path),
            "motion_path": motion_path,
            "frames": int(arrays["obs_student"].shape[0]),
            "return": float(total_return),
            "done_count": int(np.sum(arrays["done"])),
            "reference_control_mode": reference_control_mode,
            "prosthesis_ref_error_max": float(ref_error_max),
            "prosthesis_ref_error_rms_mean": float(ref_error_rms_sum / n_recorded),
        }
    finally:
        env.stop()
        student_env.stop()


def main() -> int:
    args = parse_args()
    cfg = load_fullbody_config(args.config_name)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    motions = list(args.motion_path) if args.motion_path else motion_list_from_group(args.dataset_group)
    motions = filter_motions(motions, args.motion_filter)
    if args.max_motions is not None:
        motions = motions[: int(args.max_motions)]
    if not motions:
        raise ValueError("No motions selected")

    # Build and persist mapping/spec from the first motion for downstream validation.
    env0 = make_env(cfg, env_name="MyoFullBody", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True)
    student0 = make_env(cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True)
    mapping = build_distill_mapping(env0, student0)
    spec = build_masked_obs_spec(env0)
    mapping.save_json(output_dir / "mapping.json")
    spec.save_json(output_dir / "masked_obs_spec.json")
    env0.stop()
    student0.stop()

    n_steps = resolve_steps(args.n_steps_per_motion)
    kp = parse_vec4(args.reference_pd_kp)
    kd = parse_vec4(args.reference_pd_kd)
    torque_limit = parse_vec4(args.reference_torque_limit)
    records = []
    for idx, motion in enumerate(motions, start=1):
        out_path = output_dir / safe_motion_filename(motion)
        if args.skip_existing and out_path.exists():
            print(f"[{idx}/{len(motions)}] skip existing {motion}")
            continue
        record = collect_motion(
            cfg=cfg,
            teacher_checkpoint=args.teacher_checkpoint,
            motion_path=motion,
            output_dir=output_dir,
            n_steps=n_steps,
            kp=kp,
            kd=kd,
            torque_limit=torque_limit,
            torque_slew_limit=float(args.torque_slew_limit),
            reference_control_mode=str(args.reference_control_mode),
            seed=args.seed + idx,
            stop_on_done=bool(args.stop_on_done),
        )
        records.append(record)
        print(f"[{idx}/{len(motions)}] {motion} frames={record['frames']} return={record['return']:.3f}")

    summary = {
        "records": records,
        "total_frames": int(sum(r["frames"] for r in records)),
        "n_remaining_muscles": int(mapping.n_remaining_muscles),
        "masked_obs_dim": int(spec.masked_obs_dim),
        "policy_obs_dim": int(spec.masked_obs_dim + 16),
        "prosthesis_state_feature_dim": 16,
        "reference_control_mode": str(args.reference_control_mode),
        "prosthesis_ref_error_max": float(max((r.get("prosthesis_ref_error_max", 0.0) for r in records), default=0.0)),
        "reference_pd_kp": kp,
        "reference_pd_kd": kd,
        "reference_torque_limit": torque_limit,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
