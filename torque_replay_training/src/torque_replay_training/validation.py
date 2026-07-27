"""Physics-equivalence checks for the torque replay boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .replay_env import ReplayConfig, TorqueReplayEnv
from .schema import TorqueReplayDataset


@dataclass(frozen=True)
class ValidationTolerances:
    all_qpos_max: float = 2e-3
    all_qvel_max: float = 2e-2
    split_qpos_max: float = 2e-3
    split_qvel_max: float = 2e-2


def _rollout(
    env: TorqueReplayEnv,
    steps: int,
    *,
    start_step: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    env.reset(start_step=start_step)
    qpos = [env.data.qpos.copy()]
    qvel = [env.data.qvel.copy()]
    for _ in range(steps):
        _obs, _reward, terminated, truncated, _info = env.step(np.zeros(4))
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        if terminated or truncated:
            break
    return np.asarray(qpos), np.asarray(qvel)


def _errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.asarray(actual) - np.asarray(expected[: actual.shape[0]])
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _reset_window_errors(
    env: TorqueReplayEnv,
    dataset: TorqueReplayDataset,
    starts: list[int],
    *,
    end_step: int,
    window_steps: int,
) -> dict[str, Any]:
    rows = []
    qpos_max = 0.0
    qvel_max = 0.0
    for start in starts:
        steps = min(int(window_steps), int(end_step) - int(start))
        qpos, qvel = _rollout(env, steps, start_step=int(start))
        qpos_error = _errors(qpos, dataset.rollout_qpos[start : start + steps + 1])
        qvel_error = _errors(qvel, dataset.rollout_qvel[start : start + steps + 1])
        qpos_max = max(qpos_max, qpos_error["max_abs"])
        qvel_max = max(qvel_max, qvel_error["max_abs"])
        rows.append(
            {
                "start_step": int(start),
                "steps": int(steps),
                "qpos": qpos_error,
                "qvel": qvel_error,
            }
        )
    return {"qpos_max_abs": qpos_max, "qvel_max_abs": qvel_max, "windows": rows}


def validate_replay(
    dataset_path: str | Path,
    checkpoint_path: str,
    *,
    steps: int = 0,
    tolerances: ValidationTolerances | None = None,
) -> dict[str, Any]:
    """Validate full-vector replacement and the healthy/prosthesis partition."""

    dataset = TorqueReplayDataset.load(dataset_path)
    n_steps = dataset.n_steps if int(steps) <= 0 else min(int(steps), dataset.n_steps)
    tol = tolerances or ValidationTolerances()
    common = dict(
        episode_steps=n_steps,
        random_start=False,
        healthy_kp=0.0,
        healthy_kd=0.0,
        exact_baseline=True,
        fall_height=-1e6,
        fall_up_z=-1e6,
    )
    reset_starts = sorted(
        {
            int(n_steps * fraction)
            for fraction in (0.25, 0.5, 0.75)
            if int(n_steps * fraction) < n_steps
        }
    )
    reset_window_steps = min(128, n_steps)
    with TorqueReplayEnv(
        dataset_path,
        checkpoint_path,
        ReplayConfig(**common, replay_mode="all"),
    ) as all_env:
        all_qpos, all_qvel = _rollout(all_env, n_steps)
        all_reset_errors = _reset_window_errors(
            all_env,
            dataset,
            reset_starts,
            end_step=n_steps,
            window_steps=reset_window_steps,
        )
    with TorqueReplayEnv(
        dataset_path,
        checkpoint_path,
        ReplayConfig(**common, replay_mode="split"),
    ) as split_env:
        split_qpos, split_qvel = _rollout(split_env, n_steps)
        split_reset_errors = _reset_window_errors(
            split_env,
            dataset,
            reset_starts,
            end_step=n_steps,
            window_steps=reset_window_steps,
        )

    expected_qpos = dataset.rollout_qpos[: n_steps + 1]
    expected_qvel = dataset.rollout_qvel[: n_steps + 1]
    all_qpos_error = _errors(all_qpos, expected_qpos)
    all_qvel_error = _errors(all_qvel, expected_qvel)
    split_qpos_error = _errors(split_qpos, expected_qpos)
    split_qvel_error = _errors(split_qvel, expected_qvel)
    partition_qpos_error = _errors(split_qpos, all_qpos)
    partition_qvel_error = _errors(split_qvel, all_qvel)
    checks = {
        "all_qpos": all_qpos_error["max_abs"] <= tol.all_qpos_max,
        "all_qvel": all_qvel_error["max_abs"] <= tol.all_qvel_max,
        "split_qpos": split_qpos_error["max_abs"] <= tol.split_qpos_max,
        "split_qvel": split_qvel_error["max_abs"] <= tol.split_qvel_max,
        "all_reset_qpos": all_reset_errors["qpos_max_abs"] <= tol.all_qpos_max,
        "all_reset_qvel": all_reset_errors["qvel_max_abs"] <= tol.all_qvel_max,
        "split_reset_qpos": split_reset_errors["qpos_max_abs"] <= tol.split_qpos_max,
        "split_reset_qvel": split_reset_errors["qvel_max_abs"] <= tol.split_qvel_max,
    }
    return {
        "passed": bool(all(checks.values())),
        "steps": int(n_steps),
        "checks": checks,
        "tolerances": asdict(tol),
        "all_vs_recording": {"qpos": all_qpos_error, "qvel": all_qvel_error},
        "split_zero_residual_vs_recording": {
            "qpos": split_qpos_error,
            "qvel": split_qvel_error,
        },
        "split_vs_all": {"qpos": partition_qpos_error, "qvel": partition_qvel_error},
        "random_reset_windows": {
            "starts": reset_starts,
            "window_steps": int(reset_window_steps),
            "all": all_reset_errors,
            "split": split_reset_errors,
        },
    }
