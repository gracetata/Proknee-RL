#!/usr/bin/env python
"""Collect full-muscle teacher actions paired with prosthesis-masked observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from musclemimic.distill.config import filter_motions, load_fullbody_config, make_env, motion_list_from_group, repo_root
from musclemimic.distill.obs_mask import apply_obs_mask, build_masked_obs_spec
from musclemimic.distill.osl_harness import build_osl_harness
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.rollout import safe_motion_filename

DEFAULT_TEACHER_CHECKPOINT = "/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2"
DEFAULT_OUTPUT_DIR = "data/full_muscle_masked_obs_rollouts/KIT_KINESIS_TRAINING_MOTIONS"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    p.add_argument("--dataset_group", default="KIT_KINESIS_TRAINING_MOTIONS")
    p.add_argument("--motion_path", nargs="*", default=None)
    p.add_argument("--max_motions", type=int, default=None)
    p.add_argument("--motion_filter", default="none")
    p.add_argument("--n_steps_per_motion", default="auto")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log and skip motions that fail instead of aborting (default: true)",
    )
    p.add_argument(
        "--disabled-muscle-scale",
        type=float,
        default=1.0,
        help="Scale 19 left prosthesis-boundary muscle controls during env stepping (0=hard zero).",
    )
    p.add_argument(
        "--use-osl-fsm",
        action="store_true",
        help="Inject OSL FSM torques on knee/ankle/subtalar/mtp while collecting.",
    )
    return p.parse_args()


def resolve_output(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def resolve_steps(value: str) -> int | None:
    return None if value == "auto" else int(value)


def rollout_motion(
    env,
    teacher_policy: PolicyRunner,
    motion_path: str,
    output_dir: Path,
    n_steps: int | None,
    *,
    disabled_muscle_scale: float = 1.0,
    use_osl_fsm: bool = False,
) -> dict:
    obs = env.reset()
    obs_policy = teacher_policy.reset_obs(obs)
    spec = build_masked_obs_spec(env)
    harness = build_osl_harness(env) if (use_osl_fsm or disabled_muscle_scale < 1.0) else None
    if harness is not None:
        harness.reset()
    max_steps = int(n_steps or env.th.len_trajectory(0) if getattr(env, "th", None) else 1000)
    rows: dict[str, list] = {
        "obs_teacher": [],
        "obs_student_masked": [],
        "target_full_muscle_action": [],
        "root_state": [],
        "done": [],
    }
    total_return = 0.0
    done = False
    steps = 0
    for _step in range(max_steps):
        action, _value = teacher_policy.act(obs_policy)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        rows["obs_teacher"].append(np.asarray(obs_policy, dtype=np.float32).reshape(-1))
        rows["obs_student_masked"].append(apply_obs_mask(obs, spec))
        rows["target_full_muscle_action"].append(action)
        rows["root_state"].append(np.concatenate([env.data.qpos[:7], env.data.qvel[:6]]).astype(np.float32))
        if harness is not None:
            obs_next, reward, done, _diag = harness.step(
                action,
                disabled_muscle_scale=disabled_muscle_scale,
                use_osl_torque=use_osl_fsm,
            )
        else:
            obs_next, reward, _absorbing, done, _info = env.step(action)
            reward = float(np.asarray(reward).item())
        total_return += float(np.asarray(reward).item())
        rows["done"].append(np.asarray(bool(done), dtype=np.bool_))
        obs = obs_next
        obs_policy = teacher_policy.update_obs(obs_next)
        steps += 1
        if done:
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
    arrays["disabled_muscle_scale"] = np.asarray(float(disabled_muscle_scale), dtype=np.float32)
    arrays["use_osl_fsm"] = np.asarray(bool(use_osl_fsm))
    out_path = output_dir / safe_motion_filename(motion_path)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    np.savez_compressed(str(tmp_path), **arrays)
    os.replace(tmp_path, out_path)
    return {
        "path": str(out_path),
        "motion_path": motion_path,
        "frames": steps,
        "return": float(total_return),
        "done": bool(done),
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


def _load_existing_records(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(output_dir.glob("*.npz")):
        if path.stat().st_size == 0:
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                if "obs_student_masked" not in data.files or "target_full_muscle_action" not in data.files:
                    continue
                frames = int(data["obs_student_masked"].shape[0])
                if frames <= 0:
                    continue
        except (OSError, EOFError, KeyError, ValueError):
            continue
        motion_path = path.stem.replace("_", "/")
        records.append({"path": str(path), "motion_path": motion_path, "frames": frames, "skipped_existing": True})
    return records


def summarize(records: list[dict]) -> dict:
    return {
        "collected_motions": len(records),
        "total_frames": int(sum(r["frames"] for r in records)),
        "masked_obs_dim": int(records[0]["masked_obs_dim"]) if records else 0,
        "raw_obs_dim": int(records[0]["raw_obs_dim"]) if records else 0,
        "action_dim": int(records[0]["action_dim"]) if records else 0,
        "removed_obs_count": int(records[0]["removed_obs_count"]) if records else 0,
        "average_rollout_length": float(np.mean([r["frames"] for r in records])) if records else 0.0,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
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
        f"Collecting {len(pending)} motions ({len(motions) - len(pending)} skipped existing) -> {output_dir}\n"
        f"raw_obs_dim={spec.raw_obs_dim} masked_obs_dim={spec.masked_obs_dim} action_dim={spec.action_dim} "
        f"removed_obs={len(spec.removed_obs_names)}"
    )

    records: list[dict] = _load_existing_records(output_dir) if args.skip_existing else []
    failures: list[dict] = []
    for idx, motion in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] {motion}", flush=True)
        try:
            env = make_env(cfg, env_name="MyoFullBody", motion_paths=[motion], use_mujoco=True, fixed_start=True)
            teacher_policy = PolicyRunner.from_checkpoint(
                args.teacher_checkpoint, env, deterministic=True, seed=args.seed
            )
            record = rollout_motion(
                env,
                teacher_policy,
                motion,
                output_dir,
                resolve_steps(args.n_steps_per_motion),
                disabled_muscle_scale=float(args.disabled_muscle_scale),
                use_osl_fsm=bool(args.use_osl_fsm),
            )
            env.stop()
            records.append(record)
            print(f"  frames={record['frames']} return={record['return']:.3f}", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {error}", flush=True)
            failures.append({"motion_path": motion, "error": error})
            _append_failure_log(output_dir, motion, error)
            if not args.continue_on_error:
                raise

    payload = {
        "summary": summarize(records),
        "records": records,
        "failed_count": len(failures),
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
