#!/usr/bin/env python3
"""Visualize exact full-body generalized-force replay on a local MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import _bootstrap  # noqa: F401

from torque_replay_training.control import root_up_z
from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.replay_env import ReplayConfig, TorqueReplayEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, nargs="+")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="0 replays each motion to its end")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--check-only", action="store_true", help="Run physics without opening a GUI")
    args = parser.parse_args()
    if not args.check_only and not os.environ.get("DISPLAY"):
        raise SystemExit("MuJoCo GUI requires a local DISPLAY; use --check-only on headless machines")
    if args.start_step < 0 or args.steps < 0 or args.repeat <= 0 or args.realtime_factor <= 0:
        raise SystemExit("--start-step/--steps must be non-negative; --repeat/realtime-factor must be positive")

    config = ReplayConfig(
        episode_steps=10_000_000,
        random_start=False,
        healthy_kp=0.0,
        healthy_kd=0.0,
        replay_mode="all",
        exact_baseline=True,
    )
    reports: list[dict[str, object]] = []
    with TorqueReplayEnv(args.dataset, args.checkpoint, config) as env:
        viewer = None
        if not args.check_only:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -15.0

        stop = False
        try:
            for repeat_index in range(args.repeat):
                for dataset_index, dataset in enumerate(env.datasets):
                    if args.start_step >= dataset.n_steps:
                        raise ValueError(
                            f"--start-step={args.start_step} exceeds {dataset.motion_path} "
                            f"with {dataset.n_steps} control steps"
                        )
                    target_steps = dataset.n_steps - args.start_step
                    if args.steps:
                        target_steps = min(target_steps, args.steps)
                    env.reset(start_step=args.start_step, dataset_index=dataset_index)
                    qpos_max = 0.0
                    qvel_max = 0.0
                    min_height = float(env.data.qpos[2])
                    min_up_z = root_up_z(env.data.qpos)
                    completed = 0
                    fell = False
                    for _ in range(target_steps):
                        if viewer is not None and not viewer.is_running():
                            stop = True
                            break
                        started = time.perf_counter()
                        _obs, _reward, terminated, truncated, info = env.step(np.zeros(4))
                        completed += 1
                        reference_index = args.start_step + completed
                        qpos_max = max(
                            qpos_max,
                            float(
                                np.max(
                                    np.abs(
                                        env.data.qpos - dataset.rollout_qpos[reference_index]
                                    )
                                )
                            ),
                        )
                        qvel_max = max(
                            qvel_max,
                            float(
                                np.max(
                                    np.abs(
                                        env.data.qvel - dataset.rollout_qvel[reference_index]
                                    )
                                )
                            ),
                        )
                        min_height = min(min_height, float(env.data.qpos[2]))
                        min_up_z = min(min_up_z, root_up_z(env.data.qpos))
                        if viewer is not None:
                            viewer.cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
                            viewer.sync()
                            delay = (
                                dataset.dt_control / args.realtime_factor
                                - (time.perf_counter() - started)
                            )
                            if delay > 0:
                                time.sleep(delay)
                        if terminated or truncated:
                            fell = bool(info["fell"])
                            break
                    reports.append(
                        {
                            "motion": dataset.motion_path,
                            "dataset": str(env.dataset_paths[dataset_index]),
                            "repeat": repeat_index,
                            "start_step": args.start_step,
                            "steps": completed,
                            "completed_requested": completed == target_steps,
                            "fell": fell,
                            "qpos_max_abs": qpos_max,
                            "qvel_max_abs": qvel_max,
                            "root_height_min": min_height,
                            "root_up_min": min_up_z,
                        }
                    )
                    if stop:
                        break
                if stop:
                    break
        finally:
            if viewer is not None:
                viewer.close()

    passed = all(
        row["completed_requested"]
        and not row["fell"]
        and float(row["qpos_max_abs"]) <= 2e-3
        and float(row["qvel_max_abs"]) <= 2e-2
        for row in reports
    )
    print(json.dumps({"passed": passed, "check_only": args.check_only, "replays": reports}, indent=2))
    if args.check_only and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
