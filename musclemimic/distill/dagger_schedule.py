"""Beta schedules for formal DAgger-style prosthesis distillation."""

from __future__ import annotations

import math


def beta_for_round(
    round_idx: int,
    *,
    schedule: str = "inverse_high_first",
    constant: float = 0.3,
    decay: float = 0.5,
    beta_min: float = 0.0,
    max_rounds: int | None = None,
) -> float:
    """Return expert-mix probability beta for a 1-indexed DAgger round.

    During rollout, the expert (teacher split action) is executed with
    probability beta; the student policy is executed otherwise.
    """
    if round_idx < 1:
        raise ValueError(f"DAgger round index must be >= 1, got {round_idx}")
    schedule = str(schedule).strip().lower()
    if schedule == "constant":
        beta = float(constant)
    elif schedule == "inverse":
        beta = 1.0 / (1.0 + float(round_idx))
    elif schedule == "inverse_high_first":
        beta = 1.0 / float(round_idx)
    elif schedule == "exponential":
        beta = math.exp(-float(decay) * float(round_idx - 1))
    elif schedule == "polynomial":
        beta = float(decay) ** float(round_idx - 1)
    elif schedule == "linear":
        if max_rounds is None or int(max_rounds) < 1:
            raise ValueError("linear beta schedule requires max_rounds >= 1")
        span = max(int(max_rounds) - 1, 1)
        beta = 1.0 - (float(round_idx - 1) / float(span))
    else:
        raise ValueError(
            f"Unknown beta schedule {schedule!r}; expected constant, inverse, "
            "inverse_high_first, exponential, polynomial, or linear"
        )
    return float(max(beta_min, min(beta, 1.0)))


def schedule_preview(
    num_rounds: int,
    *,
    schedule: str = "inverse_high_first",
    constant: float = 0.3,
    decay: float = 0.5,
    beta_min: float = 0.0,
) -> list[tuple[int, float]]:
    return [
        (
            round_idx,
            beta_for_round(
                round_idx,
                schedule=schedule,
                constant=constant,
                decay=decay,
                beta_min=beta_min,
                max_rounds=num_rounds,
            ),
        )
        for round_idx in range(1, int(num_rounds) + 1)
    ]
