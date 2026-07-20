#!/usr/bin/env python
"""Collect DAgger samples: student-visited masked observations with teacher action labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from musclemimic.distill.config import filter_motions, load_fullbody_config, make_env, motion_list_from_group, repo_root
from musclemimic.distill.obs_mask import MaskedObservationEnvView, apply_obs_mask, build_masked_obs_spec
from musclemimic.distill.osl_harness import build_osl_harness
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.rollout import safe_motion_filename

DEFAULT_TEACHER_CHECKPOINT = "/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2"
DEFAULT_STUDENT_CHECKPOINT = (
    "/home/user/Workspace/musclemimic/musclemimic/outputs/full_muscle_masked_obs_distill/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "data/full_muscle_masked_obs_dagger/KIT_KINESIS_TRAINING_MOTIONS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--student_checkpoint", default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--dataset_group", default="KIT_KINESIS_TRAINING_MOTIONS")
    parser.add_argument("--motion_path", nargs="*", default=None)
    parser.add_argument("--max_motions", type=int, default=None)
    parser.add_argument("--motion_filter", default="none")
    parser.add_argument("--n_steps_per_motion", default="auto")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    parser.add_argument("--teacher_action_prob", type=float, default=0.0)
    parser.add_argument("--stop_on_done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disabled-muscle-scale",
        type=float,
        default=1.0,
        help="Scale 19 left prosthesis-boundary muscle controls during rollout stepping.",
    )
    parser.add_argument(
        "--use-osl-fsm",
        action="store_true",
        help="Inject OSL FSM torques on knee/ankle/subtalar/mtp during DAgger rollout.",
    )
    return parser.parse_args()


def resolve_output(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def resolve_steps(value: str) -> int | None:
    return None if value == "auto" else int(value)


def rollout_dagger_motion(
    env,
    teacher_policy: PolicyRunner,
    student_policy: PolicyRunner,
    motion_path: str,
    output_dir: Path,
    n_steps: int | None,
    teacher_action_prob: float,
    stop_on_done: bool,
    seed: int,
    *,
    disabled_muscle_scale: float = 1.0,
    use_osl_fsm: bool = False,
) -> dict:
    """Roll out the student, but label every visited state with the teacher action."""
    rng = np.random.default_rng(seed)
    spec = build_masked_obs_spec(env)
    harness = build_osl_harness(env) if (use_osl_fsm or disabled_muscle_scale < 1.0) else None
    obs = env.reset()
    if harness is not None:
        harness.reset()
    teacher_obs_policy = teacher_policy.reset_obs(obs)
    student_obs_policy = student_policy.reset_obs(apply_obs_mask(obs, spec))
    max_steps = int(n_steps or env.th.len_trajectory(0) if getattr(env, "th", None) else 1000)
    rows: dict[str, list] = {
        "obs_student_masked": [],
        "target_full_muscle_action": [],
        "student_action": [],
        "rollout_action": [],
        "used_teacher_action": [],
        "root_state": [],
        "done": [],
    }
    total_return = 0.0
    done = False
    steps = 0
    for _step in range(max_steps):
        teacher_action, _teacher_value = teacher_policy.act(teacher_obs_policy)
        student_action, _student_value = student_policy.act(student_obs_policy)
        teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        student_action = np.asarray(student_action, dtype=np.float32).reshape(-1)
        use_teacher = bool(rng.random() < teacher_action_prob)
        rollout_action = teacher_action if use_teacher else student_action

        rows["obs_student_masked"].append(apply_obs_mask(obs, spec))
        rows["target_full_muscle_action"].append(teacher_action)
        rows["student_action"].append(student_action)
        rows["rollout_action"].append(np.asarray(rollout_action, dtype=np.float32).reshape(-1))
        rows["used_teacher_action"].append(np.asarray(use_teacher, dtype=np.bool_))
        rows["root_state"].append(np.concatenate([env.data.qpos[:7], env.data.qvel[:6]]).astype(np.float32))

        if harness is not None:
            obs_next, reward, done, _diag = harness.step(
                rollout_action,
                disabled_muscle_scale=disabled_muscle_scale,
                use_osl_torque=use_osl_fsm,
            )
        else:
            obs_next, reward, _absorbing, done, _info = env.step(rollout_action)
            reward = float(np.asarray(reward).item())
        total_return += float(np.asarray(reward).item())
        rows["done"].append(np.asarray(bool(done), dtype=np.bool_))
        obs = obs_next
        teacher_obs_policy = teacher_policy.update_obs(obs_next)
        student_obs_policy = student_policy.update_obs(apply_obs_mask(obs_next, spec))
        steps += 1
        if done and stop_on_done:
            break

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["motion_path"] = np.asarray(motion_path)
    arrays["mask_keep_indices"] = np.asarray(spec.keep_indices, dtype=np.int32)
    arrays["mask_removed_indices"] = np.asarray(spec.removed_indices, dtype=np.int32)
    arrays["masked_muscle_names"] = np.asarray(spec.masked_muscle_names)
    arrays["prosthesis_joint_names"] = np.asarray(spec.prosthesis_joint_names)
    arrays["raw_obs_dim"] = np.asarray(spec.raw_obs_dim, dtype=np.int32)
    arrays["masked_obs_dim"] = np.asarray(spec.masked_obs_dim, dtype=np.int32)
    arrays["action_dim"] = np.asarray(spec.action_dim, dtype=np.int32)
    arrays["dagger_teacher_action_prob"] = np.asarray(float(teacher_action_prob), dtype=np.float32)
    arrays["disabled_muscle_scale"] = np.asarray(float(disabled_muscle_scale), dtype=np.float32)
    arrays["use_osl_fsm"] = np.asarray(bool(use_osl_fsm))
    out_path = output_dir / safe_motion_filename(motion_path)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    np.savez_compressed(str(tmp_path), **arrays)
    os.replace(tmp_path, out_path)
    student_teacher_mse = float(np.mean((arrays["student_action"] - arrays["target_full_muscle_action"]) ** 2)) if steps else 0.0
    return {
        "path": str(out_path),
        "motion_path": motion_path,
        "frames": steps,
        "return": float(total_return),
        "done": bool(done),
        "student_teacher_mse": student_teacher_mse,
        "teacher_action_prob": float(teacher_action_prob),
        "masked_obs_dim": spec.masked_obs_dim,
        "raw_obs_dim": spec.raw_obs_dim,
        "action_dim": spec.action_dim,
        "removed_obs_count": len(spec.removed_obs_names),
        "disabled_muscle_scale": float(disabled_muscle_scale),
        "use_osl_fsm": bool(use_osl_fsm),
    }


def _append_failure_log(output_dir: Path, motion: str, error: str) -> None:
    path = output_dir / "failed_motions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"motion_path": motion, "error": error}, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    return {
        "collected_motions": len(records),
        "total_frames": int(sum(r["frames"] for r in records)),
        "average_rollout_length": float(np.mean([r["frames"] for r in records])) if records else 0.0,
        "average_student_teacher_mse": float(np.mean([r["student_teacher_mse"] for r in records])) if records else 0.0,
        "masked_obs_dim": int(records[0]["masked_obs_dim"]) if records else 0,
        "raw_obs_dim": int(records[0]["raw_obs_dim"]) if records else 0,
        "action_dim": int(records[0]["action_dim"]) if records else 0,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = load_fullbody_config(args.config_name)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    motions = list(args.motion_path) if args.motion_path else motion_list_from_group(args.dataset_group)
    motions = filter_motions(motions, args.motion_filter)
    if args.max_motions is not None:
        motions = motions[: int(args.max_motions)]
    if not motions:
        raise ValueError("No motions selected")
    pending = [m for m in motions if not (output_dir / safe_motion_filename(m)).is_file()] if args.skip_existing else motions

    env0 = make_env(cfg, env_name="MyoFullBody", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True)
    spec = build_masked_obs_spec(env0)
    spec.save_json(output_dir / "masked_obs_spec.json")
    env0.stop()
    print(
        f"Collecting {len(pending)} DAgger motions ({len(motions) - len(pending)} skipped existing) -> {output_dir}\n"
        f"raw_obs_dim={spec.raw_obs_dim} masked_obs_dim={spec.masked_obs_dim} action_dim={spec.action_dim} "
        f"teacher_action_prob={float(args.teacher_action_prob):.3f}"
    )

    records: list[dict] = []
    failures: list[dict] = []
    for idx, motion in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] {motion}", flush=True)
        env = None
        try:
            env = make_env(cfg, env_name="MyoFullBody", motion_paths=[motion], use_mujoco=True, fixed_start=True)
            teacher_policy = PolicyRunner.from_checkpoint(args.teacher_checkpoint, env, deterministic=True, seed=args.seed)
            student_view = MaskedObservationEnvView(env, build_masked_obs_spec(env))
            student_policy = PolicyRunner.from_checkpoint(args.student_checkpoint, student_view, deterministic=True, seed=args.seed)
            record = rollout_dagger_motion(
                env,
                teacher_policy,
                student_policy,
                motion,
                output_dir,
                resolve_steps(args.n_steps_per_motion),
                float(args.teacher_action_prob),
                bool(args.stop_on_done),
                int(args.seed) + idx,
                disabled_muscle_scale=float(args.disabled_muscle_scale),
                use_osl_fsm=bool(args.use_osl_fsm),
            )
            records.append(record)
            print(
                f"  frames={record['frames']} return={record['return']:.3f} "
                f"student_teacher_mse={record['student_teacher_mse']:.4f}",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {error}", flush=True)
            failures.append({"motion_path": motion, "error": error})
            _append_failure_log(output_dir, motion, error)
            if not args.continue_on_error:
                raise
        finally:
            if env is not None:
                env.stop()

    payload = {
        "summary": summarize(records),
        "records": records,
        "failed_count": len(failures),
        "failures": failures,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": args.student_checkpoint,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
