#!/usr/bin/env python3
"""Run replay equivalence tests and return nonzero on failure."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.validation import validate_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()
    result = validate_replay(args.dataset, args.checkpoint, steps=args.steps)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
