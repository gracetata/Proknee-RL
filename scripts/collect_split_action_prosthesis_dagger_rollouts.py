#!/usr/bin/env python
"""Collect split-action DAgger rollouts in MyoFullBodyProsthesisEnv."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.distill.config import (
    apply_prosthesis_overrides,
    filter_motions,
    load_fullbody_config,
    make_env,
    motion_list_from_group,
    parse_optional_vec4,
    repo_root,
)
from musclemimic.distill.dagger_schedule import beta_for_round
from musclemimic.distill.mapping import DistillMapping, build_distill_mapping
from musclemimic.distill.policy import PolicyRunner
from musclemimic.distill.rollout import rollout_split_action_dagger, safe_motion_filename
from musclemimic.runner.eval_utils import align_agent_state, load_checkpoint

DEFAULT_TEACHER_CHECKPOINT = "/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2"
DEFAULT_STUDENT_CHECKPOINT = (
    "outputs/split_action_prosthesis_distill/replay_four_refpd_60ep/latest/checkpoints/checkpoint_distilled"
)
DEFAULT_OUTPUT_DIR = "data/split_action_prosthesis_distill/dagger_replay_four"
DEFAULT_MOTIONS = [
    "KIT/3/walk_6m_straight_line04_poses",
    "KIT/425/walking_slow07_poses",
    "KIT/359/walking_run04_poses",
    "KIT/9/WalkingStraightForwards07_poses",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--student_checkpoint", default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--motion_path", nargs="*", default=None)
    parser.add_argument("--dataset_group", default=None)
    parser.add_argument("--max_motions", type=int, default=None)
    parser.add_argument("--motion_filter", default="none")
    parser.add_argument("--n_steps_per_motion", default="auto")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-name", default="conf_fullbody_prosthesis_gmr_resnet")
    parser.add_argument(
        "--mask-preset",
        default=None,
        choices=["strict19", "knee15", "distal11", "foot5", "foot1"],
        help="Disabled-muscle preset; must match BC / student checkpoint.",
    )
    parser.add_argument("--teacher_action_prob", type=float, default=None)
    parser.add_argument(
        "--dagger_round",
        type=int,
        default=None,
        help="1-indexed formal DAgger round. When set with --beta_schedule, overrides --teacher_action_prob.",
    )
    parser.add_argument(
        "--beta_schedule",
        choices=["constant", "inverse", "inverse_high_first", "exponential", "polynomial", "linear"],
        default="inverse_high_first",
        help="Expert mix probability schedule for formal DAgger.",
    )
    parser.add_argument("--beta_constant", type=float, default=0.3, help="Used when beta_schedule=constant")
    parser.add_argument("--beta_decay", type=float, default=0.7, help="Decay for exponential/polynomial schedules")
    parser.add_argument("--beta_min", type=float, default=0.0)
    parser.add_argument("--beta_max_rounds", type=int, default=None, help="Required for linear schedule")
    parser.add_argument(
        "--label_mode",
        choices=["reference_pd", "teacher_qfrc", "inverse_dynamics"],
        default="reference_pd",
    )
    parser.add_argument("--reference_pd_kp", default=None, help="Label PD kp; default uses config yaml")
    parser.add_argument("--reference_pd_kd", default=None, help="Label PD kd; default uses config yaml")
    parser.add_argument(
        "--prosthesis_action_type",
        default=None,
        choices=[None, "torque", "pd_residual_torque"],
        help="Must match the student checkpoint / rollout action semantics.",
    )
    parser.add_argument("--stop_on_done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_teacher_action_prob(args: argparse.Namespace) -> tuple[float, int | None]:
    if args.dagger_round is not None:
        beta = beta_for_round(
            int(args.dagger_round),
            schedule=args.beta_schedule,
            constant=float(args.beta_constant),
            decay=float(args.beta_decay),
            beta_min=float(args.beta_min),
            max_rounds=args.beta_max_rounds,
        )
        return beta, int(args.dagger_round)
    if args.teacher_action_prob is not None:
        return float(args.teacher_action_prob), None
    return float(args.beta_constant), None


def apply_cli_prosthesis_overrides(cfg, args) -> object:
    return apply_prosthesis_overrides(
        cfg,
        disable_preset=args.mask_preset,
        action_type=args.prosthesis_action_type,
        residual_pd_kp=parse_optional_vec4(args.reference_pd_kp) if args.reference_pd_kp else None,
        residual_pd_kd=parse_optional_vec4(args.reference_pd_kd) if args.reference_pd_kd else None,
    )


def reference_pd_from_config(cfg, args) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    kp = parse_optional_vec4(args.reference_pd_kp) if args.reference_pd_kp else None
    kd = parse_optional_vec4(args.reference_pd_kd) if args.reference_pd_kd else None
    prosthesis = dict(OmegaConf.to_container(cfg.experiment.env_params.prosthesis, resolve=True) or {})
    residual = dict(prosthesis.get("residual_pd") or {})
    if kp is None:
        kp_vals = residual.get("kp", (240.0, 180.0, 60.0, 40.0))
        kp = tuple(float(x) for x in kp_vals)
    if kd is None:
        kd_vals = residual.get("kd", (24.0, 18.0, 6.0, 4.0))
        kd = tuple(float(x) for x in kd_vals)
    return kp, kd


def resolve_output(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def resolve_steps(value: str) -> int | None:
    return None if value == "auto" else int(value)


def parse_vec4(value: str) -> tuple[float, float, float, float]:
    vals = tuple(float(x.strip()) for x in str(value).split(",") if x.strip())
    if len(vals) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got {value!r}")
    return vals


def enable_split_action_actor(cfg, mapping: DistillMapping) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.split_action_actor = {
        "enabled": True,
        "remaining_action_dim": int(mapping.n_remaining_muscles),
        "prosthesis_action_dim": int(len(mapping.prosthesis_joint_names)),
    }


def load_student_policy(checkpoint: str, env, *, seed: int) -> PolicyRunner:
    """Load a student policy for DAgger rollouts.

    Unified 343-dim prosthesis distill checkpoints are supported directly.
    Split-action checkpoints keep using the split actor layout.
    """
    config, agent_state, _metadata = load_checkpoint(checkpoint)
    OmegaConf.set_struct(config, False)
    if config.experiment.get("split_action_actor", {}).get("enabled", False):
        agent_conf = PPOJax.init_agent_conf(env, config)
        agent_state = align_agent_state(agent_state, agent_conf)
        return PolicyRunner.from_agent_state(agent_conf, agent_state, env, deterministic=True, seed=seed)

    return PolicyRunner.from_checkpoint(checkpoint, env, deterministic=True, seed=seed)


def summarize(records: list[dict], mapping: DistillMapping, *, teacher_action_prob: float, dagger_round: int | None) -> dict:
    return {
        "collected_motions": len(records),
        "total_frames": int(sum(r["frames"] for r in records)),
        "average_rollout_length": float(np.mean([r["frames"] for r in records])) if records else 0.0,
        "average_student_label_mse": float(np.mean([r["student_label_mse"] for r in records])) if records else 0.0,
        "obs_student_dim": mapping.student_obs_dim,
        "target_action_dim": mapping.target_action_dim,
        "teacher_action_prob": float(teacher_action_prob),
        "dagger_round": dagger_round,
        "dagger_beta": float(teacher_action_prob),
    }


def main() -> int:
    args = parse_args()
    teacher_action_prob, dagger_round = resolve_teacher_action_prob(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = load_fullbody_config(args.config_name)
    cfg = apply_cli_prosthesis_overrides(cfg, args)
    label_pd_kp, label_pd_kd = reference_pd_from_config(cfg, args)
    output_dir = resolve_output(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.motion_path:
        motions = list(args.motion_path)
    elif args.dataset_group:
        motions = motion_list_from_group(args.dataset_group)
    else:
        motions = list(DEFAULT_MOTIONS)
    motions = filter_motions(motions, args.motion_filter)
    if args.max_motions is not None:
        motions = motions[: int(args.max_motions)]
    pending = [
        m for m in motions if not (output_dir / safe_motion_filename(m)).is_file()
    ] if args.skip_existing else motions

    teacher_env0 = make_env(cfg, env_name="MyoFullBody", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True)
    student_env0 = make_env(
        cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[motions[0]], use_mujoco=True, fixed_start=True
    )
    mapping = build_distill_mapping(teacher_env0, student_env0)
    mapping.save_json(output_dir / "mapping.json")
    teacher_env0.stop()
    student_env0.stop()

    print(
        f"Collecting {len(pending)} split-action DAgger motions -> {output_dir}\n"
        f"student={args.student_checkpoint} teacher_action_prob={teacher_action_prob:.4f} "
        f"dagger_round={dagger_round} label_mode={args.label_mode}"
    )

    records: list[dict] = []
    failures: list[dict] = []
    for idx, motion in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] {motion}", flush=True)
        teacher_env = None
        student_env = None
        try:
            teacher_env = make_env(cfg, env_name="MyoFullBody", motion_paths=[motion], use_mujoco=True, fixed_start=True)
            student_env = make_env(
                cfg, env_name="MyoFullBodyProsthesisEnv", motion_paths=[motion], use_mujoco=True, fixed_start=True
            )
            teacher_policy = PolicyRunner.from_checkpoint(
                args.teacher_checkpoint, teacher_env, deterministic=True, seed=args.seed
            )
            student_policy = load_student_policy(
                resolve_path(args.student_checkpoint).as_posix(), student_env, seed=args.seed + idx
            )
            record = rollout_split_action_dagger(
                teacher_env=teacher_env,
                student_env=student_env,
                teacher_policy=teacher_policy,
                student_policy=student_policy,
                mapping=mapping,
                motion_path=motion,
                output_dir=output_dir,
                n_steps=resolve_steps(args.n_steps_per_motion),
                teacher_action_prob=float(teacher_action_prob),
                dagger_round=dagger_round,
                dagger_beta=float(teacher_action_prob),
                stop_on_done=bool(args.stop_on_done),
                seed=int(args.seed) + idx,
                label_mode=args.label_mode,
                reference_pd_kp=label_pd_kp,
                reference_pd_kd=label_pd_kd,
            )
            records.append(record)
            print(
                f"  frames={record['frames']} return={record['return']:.3f} "
                f"student_label_mse={record['student_label_mse']:.4f} done={record['done']}",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {error}", flush=True)
            failures.append({"motion_path": motion, "error": error})
            if not args.continue_on_error:
                raise
        finally:
            if teacher_env is not None:
                teacher_env.stop()
            if student_env is not None:
                student_env.stop()

    payload = {
        "summary": summarize(records, mapping, teacher_action_prob=teacher_action_prob, dagger_round=dagger_round),
        "records": records,
        "failed_count": len(failures),
        "failures": failures,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_checkpoint": args.student_checkpoint,
        "label_mode": args.label_mode,
        "beta_schedule": args.beta_schedule,
        "dagger_round": dagger_round,
        "dagger_beta": float(teacher_action_prob),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
