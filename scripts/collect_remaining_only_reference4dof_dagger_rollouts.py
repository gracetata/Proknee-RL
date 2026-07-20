#!/usr/bin/env python
"""Collect remaining-only DAgger rollouts under reference 4-DoF PD harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import filter_motions, load_fullbody_config, make_env, motion_list_from_group, repo_root
from musclemimic.distill.mapping import DistillMapping, build_distill_mapping
from musclemimic.distill.obs_mask import MaskedObsSpec, build_masked_obs_spec
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.remaining_only import (
    RemainingOnlyEnvView,
    build_reference4dof_harness,
    teacher_action_to_remaining,
)
from musclemimic.distill.rollout import safe_motion_filename
from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint
from train_prosthesis_distillation import DEFAULT_TEACHER_CHECKPOINT

DEFAULT_STUDENT_CHECKPOINT = (
    "outputs/_nonformal_runs/remaining_only_augobs_KIT100_80ep/2026-06-11/00-21-02/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "data/remaining_only_reference4dof_dagger/replay_four"
DEFAULT_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    p.add_argument("--student_checkpoint", default=DEFAULT_STUDENT_CHECKPOINT)
    p.add_argument("--motion_path", nargs="*", default=None)
    p.add_argument("--dataset_group", default=None)
    p.add_argument("--max_motions", type=int, default=None)
    p.add_argument("--motion_filter", default="none")
    p.add_argument("--n_steps_per_motion", default="auto")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    p.add_argument("--teacher_action_prob", type=float, default=0.3)
    p.add_argument("--reference_pd_kp", default="240,180,60,40")
    p.add_argument("--reference_pd_kd", default="24,18,6,4")
    p.add_argument("--reference_torque_limit", default="140,120,60,60")
    p.add_argument("--torque_slew_limit", type=float, default=35.0)
    p.add_argument("--reference_control_mode", choices=["pd", "lock"], default="pd")
    p.add_argument("--stop_on_done", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
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


def load_mask_spec(checkpoint: Path, env) -> MaskedObsSpec:
    for base in (checkpoint, checkpoint.parent, checkpoint.parent.parent, checkpoint.parent.parent.parent):
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


def load_remaining_student_policy(checkpoint: str, env, mapping: DistillMapping, spec: MaskedObsSpec, *, seed: int) -> PolicyRunner:
    config, agent_state, _metadata = load_checkpoint(checkpoint)
    OmegaConf.set_struct(config, False)
    env_view = RemainingOnlyEnvView(env, spec, mapping.n_remaining_muscles)
    agent_conf = PPOJax.init_agent_conf(env_view, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    return PolicyRunner.from_agent_state(agent_conf, agent_state, env_view, deterministic=True, seed=seed)


def rollout_dagger_motion(
    *,
    cfg,
    teacher_checkpoint: str,
    student_policy: PolicyRunner,
    mapping: DistillMapping,
    spec: MaskedObsSpec,
    motion_path: str,
    output_dir: Path,
    n_steps: int | None,
    teacher_action_prob: float,
    kp: tuple[float, float, float, float],
    kd: tuple[float, float, float, float],
    torque_limit: tuple[float, float, float, float],
    torque_slew_limit: float,
    reference_control_mode: str,
    stop_on_done: bool,
    seed: int,
) -> dict:
    env = make_env(cfg, env_name="MyoFullBody", motion_paths=[motion_path], use_mujoco=True, fixed_start=True)
    try:
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
        rng = np.random.default_rng(seed)
        obs_policy = harness.reset()
        teacher_obs_policy = teacher.reset_obs(np.asarray(env._obs, dtype=np.float32))
        student_obs_policy = student_policy.reset_obs(obs_policy)
        max_steps = int(n_steps or env.th.len_trajectory(0))

        rows: dict[str, list] = {
            "obs_student": [],
            "obs_policy": [],
            "target_remaining_muscle_action": [],
            "student_remaining_action": [],
            "rollout_remaining_action": [],
            "used_teacher_action": [],
            "ref_prosthesis_qpos": [],
            "ref_prosthesis_qvel": [],
            "root_state": [],
            "prosthesis_ref_error_max": [],
            "prosthesis_ref_error_rms": [],
            "done": [],
        }
        total_return = 0.0
        ref_error_max = 0.0
        ref_error_rms_sum = 0.0
        done = False
        steps = 0
        for _step in range(max_steps):
            teacher_action, _ = teacher.act(teacher_obs_policy)
            target_remaining = teacher_action_to_remaining(teacher_action, mapping)
            student_remaining, _ = student_policy.act(student_obs_policy)
            student_remaining = np.asarray(student_remaining, dtype=np.float32).reshape(-1)
            use_teacher = bool(rng.random() < float(teacher_action_prob))
            rollout_remaining = target_remaining if use_teacher else student_remaining

            rows["obs_student"].append(np.asarray(obs_policy, dtype=np.float32).reshape(-1))
            rows["obs_policy"].append(np.asarray(obs_policy, dtype=np.float32).reshape(-1))
            rows["target_remaining_muscle_action"].append(target_remaining.astype(np.float32))
            rows["student_remaining_action"].append(student_remaining.astype(np.float32))
            rows["rollout_remaining_action"].append(rollout_remaining.astype(np.float32))
            rows["used_teacher_action"].append(np.asarray(use_teacher, dtype=np.bool_))
            rows["root_state"].append(np.concatenate([env.data.qpos[:7], env.data.qvel[:6]]).astype(np.float32))

            obs_policy, reward, done, diag = harness.step(rollout_remaining)
            total_return += float(reward)
            ref_q = np.asarray(diag["ref_prosthesis_qpos"], dtype=np.float32)
            ref_qd = np.asarray(diag["ref_prosthesis_qvel"], dtype=np.float32)
            rows["ref_prosthesis_qpos"].append(ref_q)
            rows["ref_prosthesis_qvel"].append(ref_qd)
            err_max = float(diag.get("prosthesis_ref_error_max", 0.0))
            err_rms = float(diag.get("prosthesis_ref_error_rms", 0.0))
            ref_error_max = max(ref_error_max, err_max)
            ref_error_rms_sum += err_rms
            rows["prosthesis_ref_error_max"].append(np.asarray(err_max, dtype=np.float32))
            rows["prosthesis_ref_error_rms"].append(np.asarray(err_rms, dtype=np.float32))
            rows["done"].append(np.asarray(bool(done), dtype=np.bool_))

            teacher_obs_policy = teacher.update_obs(np.asarray(env._obs, dtype=np.float32))
            student_obs_policy = student_policy.update_obs(obs_policy)
            steps += 1
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
        arrays["dagger_teacher_action_prob"] = np.asarray(float(teacher_action_prob), dtype=np.float32)
        arrays["dagger_student_checkpoint"] = np.asarray(str(student_policy.checkpoint_path))
        out_path = output_dir / safe_motion_filename(motion_path)
        np.savez_compressed(out_path, **arrays)
        student_label_mse = float(
            np.mean((arrays["student_remaining_action"] - arrays["target_remaining_muscle_action"]) ** 2)
        ) if steps else 0.0
        n_recorded = max(int(arrays["obs_policy"].shape[0]), 1)
        return {
            "path": str(out_path),
            "motion_path": motion_path,
            "frames": int(steps),
            "return": float(total_return),
            "done": bool(done),
            "student_label_mse": student_label_mse,
            "teacher_action_prob": float(teacher_action_prob),
            "reference_control_mode": reference_control_mode,
            "prosthesis_ref_error_max": float(ref_error_max),
            "prosthesis_ref_error_rms_mean": float(ref_error_rms_sum / n_recorded),
        }
    finally:
        env.stop()


def summarize(records: list[dict]) -> dict:
    return {
        "collected_motions": len(records),
        "total_frames": int(sum(r["frames"] for r in records)),
        "average_rollout_length": float(np.mean([r["frames"] for r in records])) if records else 0.0,
        "average_student_label_mse": float(np.mean([r["student_label_mse"] for r in records])) if records else 0.0,
        "teacher_action_prob": float(records[0]["teacher_action_prob"]) if records else 0.0,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = load_fullbody_config(args.config_name)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    motions = list(args.motion_path) if args.motion_path else (
        motion_list_from_group(args.dataset_group) if args.dataset_group else list(DEFAULT_MOTIONS)
    )
    motions = filter_motions(motions, args.motion_filter)
    if args.max_motions is not None:
        motions = motions[: int(args.max_motions)]
    if not motions:
        raise ValueError("No motions selected")
    pending = [m for m in motions if not (output_dir / safe_motion_filename(m)).is_file()] if args.skip_existing else motions

    env0 = make_env(cfg, env_name="MyoFullBody", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True)
    student0 = make_env(cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True)
    mapping = build_distill_mapping(env0, student0)
    spec = load_mask_spec(resolve_path(args.student_checkpoint), env0)
    mapping.save_json(output_dir / "mapping.json")
    spec.save_json(output_dir / "masked_obs_spec.json")
    env0.stop()
    student0.stop()

    kp = parse_vec4(args.reference_pd_kp)
    kd = parse_vec4(args.reference_pd_kd)
    torque_limit = parse_vec4(args.reference_torque_limit)
    n_steps = resolve_steps(args.n_steps_per_motion)
    print(
        f"Collecting {len(pending)} remaining-only DAgger motions ({len(motions) - len(pending)} skipped) -> {output_dir}\n"
        f"policy_obs_dim={spec.masked_obs_dim + 16} teacher_action_prob={float(args.teacher_action_prob):.3f} "
        f"reference_control_mode={args.reference_control_mode}"
    )

    records: list[dict] = []
    failures: list[dict] = []
    for idx, motion in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] {motion}", flush=True)
        try:
            env = make_env(cfg, env_name="MyoFullBody", motion_paths=[motion], use_mujoco=True, fixed_start=True)
            try:
                student_policy = load_remaining_student_policy(
                    args.student_checkpoint,
                    env,
                    mapping,
                    spec,
                    seed=int(args.seed) + idx,
                )
                student_policy.checkpoint_path = str(resolve_path(args.student_checkpoint))
                record = rollout_dagger_motion(
                    cfg=cfg,
                    teacher_checkpoint=args.teacher_checkpoint,
                    student_policy=student_policy,
                    mapping=mapping,
                    spec=spec,
                    motion_path=motion,
                    output_dir=output_dir,
                    n_steps=n_steps,
                    teacher_action_prob=float(args.teacher_action_prob),
                    kp=kp,
                    kd=kd,
                    torque_limit=torque_limit,
                    torque_slew_limit=float(args.torque_slew_limit),
                    reference_control_mode=str(args.reference_control_mode),
                    stop_on_done=bool(args.stop_on_done),
                    seed=int(args.seed) + idx,
                )
            finally:
                env.stop()
            records.append(record)
            print(
                f"  frames={record['frames']} return={record['return']:.3f} "
                f"student_label_mse={record['student_label_mse']:.4f}",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {error}", flush=True)
            failures.append({"motion_path": motion, "error": error})
            if not args.continue_on_error:
                raise

    payload = {
        "summary": summarize(records),
        "records": records,
        "failed_count": len(failures),
        "failures": failures,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": args.student_checkpoint,
        "reference_control_mode": args.reference_control_mode,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
