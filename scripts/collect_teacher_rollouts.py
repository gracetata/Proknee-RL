#!/usr/bin/env python
"""Collect official MyoFullBody teacher rollouts for prosthesis distillation."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from musclemimic.distill.config import (
    apply_prosthesis_overrides,
    filter_motions,
    load_fullbody_config,
    make_env,
    motion_list_from_group,
    parse_optional_csv,
    parse_optional_vec4,
    repo_root,
)
from musclemimic.distill.mapping import (
    DistillMapping,
    build_distill_mapping,
    rollout_npz_is_valid,
    spot_check_rollout_npz,
    validate_distill_dimensions,
)
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.rollout import rollout_motion, safe_motion_filename

DEFAULT_TEACHER_CHECKPOINT = "/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2"
DEFAULT_OUTPUT_DIR = "data/teacher_rollouts/KIT_KINESIS_TRAINING_MOTIONS"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--dataset_group", default="KIT_KINESIS_TRAINING_MOTIONS")
    parser.add_argument("--motion_path", nargs="*", default=None)
    parser.add_argument("--max_motions", type=int, default=None)
    parser.add_argument("--motion_filter", default="none")
    parser.add_argument("--n_steps_per_motion", default="auto")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    parser.add_argument("--lowpass_alpha", type=float, default=None)
    parser.add_argument("--no_clip_tau", action="store_true")
    parser.add_argument(
        "--label_mode",
        choices=["reference_pd", "teacher_qfrc", "inverse_dynamics"],
        default="inverse_dynamics",
        help="How to generate the 4-DOF prosthesis torque labels.",
    )
    parser.add_argument(
        "--pin_student_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record student observations from teacher/reference kinematics instead of free-rolling student env.",
    )
    parser.add_argument("--reference_pd_kp", default="240,180,60,40")
    parser.add_argument("--reference_pd_kd", default="24,18,6,4")
    parser.add_argument("--disable_muscles_mode", default=None, help="Override prosthesis.disable_muscles.mode")
    parser.add_argument(
        "--mask-preset",
        default=None,
        choices=["strict19", "knee15", "distal11", "foot5", "foot1"],
        help="Disabled-muscle preset for prosthesis distillation (default strict19 from config).",
    )
    parser.add_argument("--disable_muscles_include", default=None, help="Comma-separated extra disabled muscle names")
    parser.add_argument(
        "--disable_muscles_exclude",
        default=None,
        help="Comma-separated default disabled muscles to keep active",
    )
    parser.add_argument(
        "--prosthesis_torque_limits",
        default=None,
        help="Override knee,ankle,subtalar,mtp torque limits, e.g. 240,160,30,20",
    )
    parser.add_argument(
        "--prosthesis_action_type",
        default=None,
        choices=[None, "torque", "pd_residual_torque"],
        help="Override the final 4 action dimensions' control semantics.",
    )
    parser.add_argument("--residual_pd_kp", default=None, help="PD gains for pd_residual_torque action type")
    parser.add_argument("--residual_pd_kd", default=None, help="Damping gains for pd_residual_torque action type")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", help="Skip motions whose .npz already exists")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel motion workers (CPU MuJoCo sim)")
    parser.add_argument("--partial_summary_every", type=int, default=25, help="Write summary.partial.json every N new records")
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log and skip motions that fail instead of aborting the run (default: true)",
    )
    return parser.parse_args()


def resolve_output(path: str) -> Path:
    out = Path(path)
    return out if out.is_absolute() else repo_root() / out


def resolve_steps(value: str) -> int | None:
    if value == "auto":
        return None
    return int(value)


def parse_vec4(value: str) -> tuple[float, float, float, float]:
    vals = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if len(vals) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got {value!r}")
    return vals


def apply_cli_prosthesis_overrides(cfg, args_or_payload):
    get = args_or_payload.get if isinstance(args_or_payload, dict) else lambda k, default=None: getattr(args_or_payload, k, default)
    return apply_prosthesis_overrides(
        cfg,
        disable_mode=get("disable_muscles_mode"),
        disable_preset=get("mask_preset"),
        disable_include=parse_optional_csv(get("disable_muscles_include")),
        disable_exclude=parse_optional_csv(get("disable_muscles_exclude")),
        torque_limits=parse_optional_vec4(get("prosthesis_torque_limits")),
        action_type=get("prosthesis_action_type"),
        residual_pd_kp=parse_optional_vec4(get("residual_pd_kp")),
        residual_pd_kd=parse_optional_vec4(get("residual_pd_kd")),
    )


def summarize(records: list[dict], mapping: DistillMapping) -> dict:
    frames = int(sum(r["frames"] for r in records))
    tau_min = np.asarray([r["tau_min"] for r in records], dtype=np.float32)
    tau_max = np.asarray([r["tau_max"] for r in records], dtype=np.float32)
    clip = np.asarray([r["clip_ratio"] for r in records], dtype=np.float32)
    lengths = np.asarray([r["frames"] for r in records], dtype=np.float32)
    return {
        "collected_motions": len(records),
        "total_frames": frames,
        "obs_student_dim": mapping.student_obs_dim,
        "teacher_obs_dim": mapping.teacher_obs_dim,
        "n_remaining_muscles": mapping.n_remaining_muscles,
        "n_disabled_muscles": len(mapping.disabled_muscle_names),
        "target_action_dim": mapping.target_action_dim,
        "prosthesis_tau_min": tau_min.min(axis=0).tolist() if len(records) else [0.0] * 4,
        "prosthesis_tau_max": tau_max.max(axis=0).tolist() if len(records) else [0.0] * 4,
        "target_prosthesis_action_clipping_ratio": float(clip.mean()) if len(records) else 0.0,
        "average_rollout_length": float(lengths.mean()) if len(records) else 0.0,
    }


def _collect_motion_worker(payload: dict) -> dict:
    if payload.get("parallel_worker"):
        # Spawned workers must not touch CUDA (parallel JAX compiles crash/OOM the GPU).
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from musclemimic.distill.config import load_fullbody_config, make_env
    from musclemimic.distill.mapping import DistillMapping
    from musclemimic.distill.policy import PolicyRunner
    from musclemimic.distill.rollout import rollout_motion

    cfg = apply_cli_prosthesis_overrides(load_fullbody_config(payload["config_name"]), payload)
    mapping = DistillMapping.load_json(payload["mapping_json"])
    motion = payload["motion"]
    output_dir = Path(payload["output_dir"])

    teacher_env = make_env(cfg, env_name="MyoFullBody", motion_paths=[motion], use_mujoco=True, fixed_start=True)
    student_env = make_env(
        cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[motion], use_mujoco=True, fixed_start=True
    )
    teacher_policy = PolicyRunner.from_checkpoint(
        payload["teacher_checkpoint"],
        teacher_env,
        deterministic=True,
        seed=int(payload["seed"]),
    )
    record = rollout_motion(
        teacher_env=teacher_env,
        student_env=student_env,
        teacher_policy=teacher_policy,
        mapping=mapping,
        motion_path=motion,
        output_dir=output_dir,
        n_steps=payload["n_steps"],
        lowpass_alpha=payload["lowpass_alpha"],
        clip_tau=payload["clip_tau"],
        label_mode=payload["label_mode"],
        pin_student_state=payload["pin_student_state"],
        reference_pd_kp=payload["reference_pd_kp"],
        reference_pd_kd=payload["reference_pd_kd"],
    )
    record["motion_path"] = motion
    return record


def _load_existing_records(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(output_dir.glob("*.npz")):
        if not rollout_npz_is_valid(path):
            print(f"WARNING: skipping corrupt/incomplete npz (will re-collect if pending): {path.name}", flush=True)
            continue
        try:
            with np.load(path, allow_pickle=True) as data:
                motion_path = str(data["motion_path"].item()) if data["motion_path"].shape == () else str(data["motion_path"][0])
                frames = int(data["obs_student"].shape[0])
                tau = np.asarray(data["target_prosthesis_tau"], dtype=np.float32)
                action = np.asarray(data["target_prosthesis_action"], dtype=np.float32)
        except (OSError, EOFError, KeyError, ValueError) as exc:
            print(f"WARNING: skipping unreadable npz {path.name}: {exc}", flush=True)
            continue
        records.append(
            {
                "path": str(path),
                "motion_path": motion_path,
                "frames": frames,
                "return": 0.0,
                "tau_mean": tau.mean(axis=0).tolist() if tau.size else [0.0] * 4,
                "tau_std": tau.std(axis=0).tolist() if tau.size else [0.0] * 4,
                "tau_min": tau.min(axis=0).tolist() if tau.size else [0.0] * 4,
                "tau_max": tau.max(axis=0).tolist() if tau.size else [0.0] * 4,
                "clip_ratio": float(np.mean(np.abs(action) >= 0.999)) if action.size else 0.0,
                "skipped_existing": True,
            }
        )
    return records


def _write_partial_summary(output_dir: Path, records: list[dict], mapping: DistillMapping) -> None:
    payload = {"summary": summarize(records, mapping), "records": records}
    (output_dir / "summary.partial.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_failure_log(output_dir: Path, motion: str, error: str) -> None:
    path = output_dir / "failed_motions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"motion_path": motion, "error": error}, ensure_ascii=False) + "\n")


def _write_failures_summary(output_dir: Path, failures: list[dict]) -> None:
    payload = {
        "failed_count": len(failures),
        "failures": failures,
        "retry_hint": (
            "uv run python scripts/collect_teacher_rollouts.py --skip_existing "
            f"--motion_path $(jq -r .motion_path {output_dir / 'failed_motions.jsonl'} | tr '\\n' ' ')"
            if failures
            else ""
        ),
    }
    (output_dir / "failures.summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _handle_motion_result(
    *,
    motion: str,
    record: dict | None,
    error: str | None,
    output_dir: Path,
    failures: list[dict],
    new_records: list[dict],
    continue_on_error: bool,
) -> None:
    if record is not None:
        new_records.append(record)
        return
    assert error is not None
    failures.append({"motion_path": motion, "error": error})
    _append_failure_log(output_dir, motion, error)
    if not continue_on_error:
        raise RuntimeError(f"Teacher rollout failed for {motion}: {error}")


def _motion_output_path(output_dir: Path, motion: str) -> Path:
    return output_dir / safe_motion_filename(motion)


def _record_from_npz(path: Path, motion: str) -> dict:
    with np.load(path, allow_pickle=True) as data:
        frames = int(data["obs_student"].shape[0])
        tau = np.asarray(data["target_prosthesis_tau"], dtype=np.float32)
        action = np.asarray(data["target_prosthesis_action"], dtype=np.float32)
    return {
        "path": str(path),
        "motion_path": motion,
        "frames": frames,
        "return": 0.0,
        "tau_mean": tau.mean(axis=0).tolist() if tau.size else [0.0] * 4,
        "tau_std": tau.std(axis=0).tolist() if tau.size else [0.0] * 4,
        "tau_min": tau.min(axis=0).tolist() if tau.size else [0.0] * 4,
        "tau_max": tau.max(axis=0).tolist() if tau.size else [0.0] * 4,
        "clip_ratio": float(np.mean(np.abs(action) >= 0.999)) if action.size else 0.0,
        "recovered_existing": True,
    }


class _ResilientProcessPool:
    """Process pool that recreates itself after worker crashes (OOM/segfault)."""

    def __init__(self, num_workers: int, mp_ctx: mp.context.BaseContext):
        self.num_workers = int(num_workers)
        self.mp_ctx = mp_ctx
        self.pool: ProcessPoolExecutor | None = None
        self.recreate_count = 0
        self.recreate()

    def recreate(self) -> None:
        if self.pool is not None:
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        self.pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=self.mp_ctx,
            max_tasks_per_child=1,
        )
        self.recreate_count += 1
        if self.recreate_count > 1:
            print(f"WARNING: recreated process pool (#{self.recreate_count - 1})", flush=True)

    def submit(self, fn, payload: dict):
        assert self.pool is not None
        try:
            return self.pool.submit(fn, payload)
        except BrokenProcessPool:
            self.recreate()
            return self.pool.submit(fn, payload)

    def shutdown(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=False)
            self.pool = None


def _requeue_motion_if_needed(
    motion: str,
    pending_queue: list[str],
    output_dir: Path,
    retry_counts: dict[str, int],
    max_retries: int,
) -> bool:
    """Return True if motion was re-queued for another attempt."""
    if rollout_npz_is_valid(_motion_output_path(output_dir, motion)):
        return False
    attempts = int(retry_counts.get(motion, 0)) + 1
    retry_counts[motion] = attempts
    if attempts > max_retries:
        return False
    pending_queue.insert(0, motion)
    print(f"  re-queue {motion} (attempt {attempts}/{max_retries})", flush=True)
    return True


def _run_parallel_collection(
    *,
    pending: list[str],
    worker_payload: dict,
    num_workers: int,
    output_dir: Path,
    mapping: DistillMapping,
    partial_summary_every: int,
    continue_on_error: bool,
    max_motion_retries: int = 2,
) -> tuple[list[dict], list[dict]]:
    mp_ctx = mp.get_context("spawn")
    pending_queue = list(pending)
    new_records: list[dict] = []
    failures: list[dict] = []
    retry_counts: dict[str, int] = {}
    done_count = 0
    total = len(pending)

    pool_mgr = _ResilientProcessPool(num_workers, mp_ctx)
    in_flight: dict = {}

    def submit_next() -> None:
        while pending_queue and len(in_flight) < num_workers:
            motion = pending_queue.pop(0)
            out_path = _motion_output_path(output_dir, motion)
            if rollout_npz_is_valid(out_path):
                done_count += 1
                record = _record_from_npz(out_path, motion)
                new_records.append(record)
                print(f"[{done_count}/{total}] skip existing {motion} frames={record['frames']}", flush=True)
                continue
            payload = {**worker_payload, "motion": motion, "parallel_worker": True}
            try:
                future = pool_mgr.submit(_collect_motion_worker, payload)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[{done_count + 1}/{total}] FAILED submit {motion}: {error}", flush=True)
                if _requeue_motion_if_needed(motion, pending_queue, output_dir, retry_counts, max_motion_retries):
                    pool_mgr.recreate()
                    in_flight.clear()
                    return
                _handle_motion_result(
                    motion=motion,
                    record=None,
                    error=error,
                    output_dir=output_dir,
                    failures=failures,
                    new_records=new_records,
                    continue_on_error=continue_on_error,
                )
                done_count += 1
                continue
            in_flight[future] = motion

    try:
        submit_next()
        while in_flight or pending_queue:
            if not in_flight:
                submit_next()
                if not in_flight and not pending_queue:
                    break
                if not in_flight:
                    continue

            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            crashed_motions: list[str] = []
            for future in finished:
                motion = in_flight.pop(future)
                done_count += 1
                record = None
                error = None
                try:
                    record = future.result()
                except BrokenProcessPool as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    crashed_motions.append(motion)
                    print(f"[{done_count}/{total}] POOL CRASH on {motion}: {error}", flush=True)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"[{done_count}/{total}] FAILED {motion}: {error}", flush=True)

                if record is not None:
                    print(
                        f"[{done_count}/{total}] {motion} frames={record['frames']} "
                        f"clip={record['clip_ratio']:.4f} return={record['return']:.3f}",
                        flush=True,
                    )
                    _handle_motion_result(
                        motion=motion,
                        record=record,
                        error=None,
                        output_dir=output_dir,
                        failures=failures,
                        new_records=new_records,
                        continue_on_error=continue_on_error,
                    )
                elif motion in crashed_motions:
                    out_path = _motion_output_path(output_dir, motion)
                    if rollout_npz_is_valid(out_path):
                        record = _record_from_npz(out_path, motion)
                        print(f"[{done_count}/{total}] recovered existing {motion} after pool crash", flush=True)
                        new_records.append(record)
                    elif _requeue_motion_if_needed(
                        motion, pending_queue, output_dir, retry_counts, max_motion_retries
                    ):
                        done_count -= 1
                    else:
                        _handle_motion_result(
                            motion=motion,
                            record=None,
                            error=error or "BrokenProcessPool",
                            output_dir=output_dir,
                            failures=failures,
                            new_records=new_records,
                            continue_on_error=continue_on_error,
                        )
                else:
                    _handle_motion_result(
                        motion=motion,
                        record=None,
                        error=error or "unknown error",
                        output_dir=output_dir,
                        failures=failures,
                        new_records=new_records,
                        continue_on_error=continue_on_error,
                    )

                if partial_summary_every > 0 and len(new_records) % partial_summary_every == 0:
                    _write_partial_summary(output_dir, _load_existing_records(output_dir), mapping)

            if crashed_motions:
                for future, motion in list(in_flight.items()):
                    in_flight.pop(future, None)
                    if motion in crashed_motions:
                        continue
                    out_path = _motion_output_path(output_dir, motion)
                    if rollout_npz_is_valid(out_path):
                        record = _record_from_npz(out_path, motion)
                        new_records.append(record)
                        print(f"[{done_count}/{total}] recovered existing {motion} after pool crash", flush=True)
                        continue
                    if _requeue_motion_if_needed(
                        motion, pending_queue, output_dir, retry_counts, max_motion_retries
                    ):
                        print(f"  re-queue in-flight {motion} after pool crash", flush=True)
                    else:
                        _handle_motion_result(
                            motion=motion,
                            record=None,
                            error="BrokenProcessPool: in-flight worker lost",
                            output_dir=output_dir,
                            failures=failures,
                            new_records=new_records,
                            continue_on_error=continue_on_error,
                        )
                pool_mgr.recreate()
            submit_next()
    finally:
        pool_mgr.shutdown()

    return new_records, failures


def main() -> int:
    args = parse_args()
    cfg = apply_cli_prosthesis_overrides(load_fullbody_config(args.config_name), args)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motions = list(args.motion_path) if args.motion_path else motion_list_from_group(args.dataset_group)
    motions = filter_motions(motions, args.motion_filter)
    if args.max_motions is not None:
        motions = motions[: args.max_motions]
    if not motions:
        raise ValueError("No motions selected for teacher rollout collection")

    teacher_env0 = make_env(cfg, env_name="MyoFullBody", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True)
    student_env0 = make_env(
        cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True
    )
    mapping = build_distill_mapping(teacher_env0, student_env0)
    mapping_path = output_dir / "mapping.json"
    mapping.save_json(mapping_path)
    validate_distill_dimensions(mapping, obs_dim=mapping.student_obs_dim, n_remaining=mapping.n_remaining_muscles)

    pending = motions
    if args.skip_existing:
        pending = [
            m
            for m in motions
            if not rollout_npz_is_valid(output_dir / safe_motion_filename(m))
        ]

    print(f"Collecting {len(pending)} motions ({len(motions) - len(pending)} skipped existing) -> {output_dir}")
    print(
        f"student_obs_dim={mapping.student_obs_dim} teacher_obs_dim={mapping.teacher_obs_dim} "
        f"target_action_dim={mapping.target_action_dim} disabled={len(mapping.disabled_muscle_names)} "
        f"workers={args.num_workers}"
    )

    existing_npz = [p for p in sorted(output_dir.glob("*.npz")) if rollout_npz_is_valid(p)]
    if existing_npz:
        spot_check_rollout_npz(existing_npz[0], mapping)
        print(f"Spot-check OK: {existing_npz[0].name}")

    n_steps = resolve_steps(args.n_steps_per_motion)
    worker_payload = {
        "config_name": args.config_name,
        "mapping_json": str(mapping_path),
        "output_dir": str(output_dir),
        "teacher_checkpoint": args.teacher_checkpoint,
        "seed": args.seed,
        "n_steps": n_steps,
        "lowpass_alpha": args.lowpass_alpha,
        "clip_tau": not args.no_clip_tau,
        "label_mode": args.label_mode,
        "pin_student_state": bool(args.pin_student_state),
        "reference_pd_kp": parse_vec4(args.reference_pd_kp),
        "reference_pd_kd": parse_vec4(args.reference_pd_kd),
        "disable_muscles_mode": args.disable_muscles_mode,
        "disable_muscles_include": args.disable_muscles_include,
        "disable_muscles_exclude": args.disable_muscles_exclude,
        "prosthesis_torque_limits": args.prosthesis_torque_limits,
        "prosthesis_action_type": args.prosthesis_action_type,
        "residual_pd_kp": args.residual_pd_kp,
        "residual_pd_kd": args.residual_pd_kd,
    }

    new_records: list[dict] = []
    failures: list[dict] = []
    if args.num_workers <= 1:
        for idx, motion in enumerate(pending, start=1):
            print(f"[{idx}/{len(pending)}] {motion}", flush=True)
            record = None
            error = None
            try:
                record = _collect_motion_worker({**worker_payload, "motion": motion})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED: {error}", flush=True)
            if record is not None:
                print(
                    f"  frames={record['frames']} clip_ratio={record['clip_ratio']:.4f} return={record['return']:.3f}",
                    flush=True,
                )
            _handle_motion_result(
                motion=motion,
                record=record,
                error=error,
                output_dir=output_dir,
                failures=failures,
                new_records=new_records,
                continue_on_error=args.continue_on_error,
            )
            if args.partial_summary_every > 0 and len(new_records) % args.partial_summary_every == 0:
                _write_partial_summary(output_dir, _load_existing_records(output_dir), mapping)
    else:
        new_records, failures = _run_parallel_collection(
            pending=pending,
            worker_payload=worker_payload,
            num_workers=args.num_workers,
            output_dir=output_dir,
            mapping=mapping,
            partial_summary_every=args.partial_summary_every,
            continue_on_error=args.continue_on_error,
        )

    if failures:
        _write_failures_summary(output_dir, failures)
        print(f"WARNING: {len(failures)} motions failed; see {output_dir / 'failed_motions.jsonl'}", flush=True)

    all_records = _load_existing_records(output_dir)
    summary = summarize(all_records, mapping)
    summary["failed_this_run"] = len(failures)
    summary["pending_motions"] = max(len(motions) - len(all_records), 0)
    payload = {"summary": summary, "records": all_records}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures and not args.continue_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
