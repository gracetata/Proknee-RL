#!/usr/bin/env python
"""Collect split-action prosthesis distillation rollouts.

This is an isolated entrypoint around the existing teacher rollout collector.
It keeps the same .npz schema while defaulting to the split-action experiment
data directory.
"""

from __future__ import annotations

import collect_teacher_rollouts as base


base.DEFAULT_OUTPUT_DIR = "data/split_action_prosthesis_distill/KIT_KINESIS_TRAINING_MOTIONS"


if __name__ == "__main__":
    raise SystemExit(base.main())
