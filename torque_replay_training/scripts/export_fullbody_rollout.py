#!/usr/bin/env python3
"""Export a successful full-body tracker rollout with substep forces."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from torque_replay_training.exporter import export_fullbody_rollout
from torque_replay_training.paths import DEFAULT_CHECKPOINT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, help="AMASS motion path accepted by MuscleMimic")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=0, help="0 exports the complete motion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="For smoke/debug only; production datasets must omit this flag",
    )
    args = parser.parse_args()
    dataset = export_fullbody_rollout(
        checkpoint_path=args.checkpoint,
        motion_path=args.motion,
        output_path=args.output,
        n_steps=args.steps,
        seed=args.seed,
        deterministic=not args.stochastic,
        require_complete=not args.allow_incomplete,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "steps": dataset.n_steps,
                "substeps": dataset.n_substeps,
                "qualified_full_motion": dataset.metadata["qualified_full_motion"],
                "root_height_min": dataset.metadata["root_height_min"],
                "root_up_min": dataset.metadata["root_up_min"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
