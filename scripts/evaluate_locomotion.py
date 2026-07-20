#!/usr/bin/env python
"""Unified locomotion evaluation CLI (official / prosthesis / ProKnee)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.distill.config import repo_root
from musclemimic.evaluation.runner import (
    LocomotionEvalRunner,
    default_eval_force_from_hydra,
    resolve_motion_list,
)
from musclemimic.evaluation.types import EvalConfig

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True ")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="conf_fullbody_gmr_resnet")
    p.add_argument("--eval-config-name", default="conf_eval_locomotion")
    p.add_argument(
        "--env_type",
        required=True,
        choices=["official_fullbody", "masked_full_muscle", "prosthesis", "prosthesis_external", "proknee_hybrid"],
    )
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--motion_path", default=None)
    p.add_argument("--motion_list", default=None, help="Text file with one motion path per line")
    p.add_argument("--dataset_group", default=None)
    p.add_argument("--output_dir", default="outputs/eval")
    p.add_argument("--n_steps", default="auto")
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--use_mujoco", type=_str2bool, default=True)
    p.add_argument("--save_video", type=_str2bool, default=True)
    p.add_argument("--save_plots", type=_str2bool, default=True)
    p.add_argument("--show_ghost", type=_str2bool, default=True)
    p.add_argument("--no_ghost", action="store_true", help="Disable ghost overlay")
    p.add_argument("--no_termination", action="store_true")
    p.add_argument("--prosthesis_control_mode", default="eval_policy")
    p.add_argument("--prosthesis_controller", default="reference_pd", choices=["reference_pd", "fsm_impedance"])
    p.add_argument("--force_reference_path", default=None)
    p.add_argument("--healthy_teacher_path", default=None)
    p.add_argument("--proknee_checkpoint", default=None)
    p.add_argument("--controller_type", default=None, help="Label for outputs; defaults to env_type")
    return p.parse_args()


def load_eval_hydra(config_name: str, eval_config_name: str):
    cfg_dir = str(repo_root() / "fullbody")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        main_cfg = compose(config_name=config_name)
        eval_cfg = compose(config_name=eval_config_name)
    if eval_cfg is not None and "eval" in eval_cfg:
        OmegaConf.set_struct(main_cfg, False)
        main_cfg.eval = eval_cfg.eval
    return main_cfg


def main() -> int:
    args = parse_args()
    hydra_cfg = load_eval_hydra(args.config_name, args.eval_config_name)
    base_out = Path(args.output_dir)
    if not base_out.is_absolute():
        base_out = repo_root() / base_out

    show_ghost = args.show_ghost and not args.no_ghost
    eval_force = default_eval_force_from_hydra(hydra_cfg)

    eval_config = EvalConfig(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        controller_type=args.controller_type or args.env_type,
        env_type=args.env_type,
        output_dir=base_out,
        eval_seed=args.eval_seed,
        n_steps=args.n_steps,
        use_mujoco=args.use_mujoco,
        save_video=args.save_video,
        save_plots=args.save_plots,
        show_ghost=show_ghost,
        no_termination=args.no_termination,
        prosthesis_control_mode=args.prosthesis_control_mode,
        prosthesis_controller=args.prosthesis_controller,
        force_reference_path=args.force_reference_path,
        healthy_teacher_path=args.healthy_teacher_path,
        eval_force=eval_force,
        proknee_checkpoint=args.proknee_checkpoint,
        extra={},
    )

    motions = resolve_motion_list(
        motion_path=args.motion_path,
        motion_list_file=args.motion_list,
        dataset_group=args.dataset_group,
    )
    runner = LocomotionEvalRunner(eval_config, motions, cli_args=vars(args))
    results = runner.run_all()
    n_ok = sum(1 for r in results if r.success and not r.error_message)
    print(f"Eval complete: {n_ok}/{len(results)} succeeded -> {runner.eval_dir}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
