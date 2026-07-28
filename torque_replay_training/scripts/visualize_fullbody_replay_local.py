#!/usr/bin/env python3
"""Interactively visualize exact full-body generalized-force replay in MuJoCo."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401

from torque_replay_training.control import root_up_z
from torque_replay_training.paths import DEFAULT_CHECKPOINT, NEW_PROJECT_ROOT
from torque_replay_training.replay_env import ReplayConfig, TorqueReplayEnv


DEFAULT_MANIFEST = (
    NEW_PROJECT_ROOT / "data" / "fullbody_all_v3" / "validation_manifest.json"
)
SPACE_KEY = 32


def _manifest_datasets(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets: list[Path] = []
    for row in payload.get("rollouts", []):
        if row.get("passed") is not True:
            continue
        candidate = Path(str(row["dataset"])).expanduser()
        if not candidate.is_absolute():
            candidate = NEW_PROJECT_ROOT / candidate
        if not candidate.is_file():
            relocated = path.parent / candidate.name
            if relocated.is_file():
                candidate = relocated
        if not candidate.is_file():
            raise FileNotFoundError(
                f"validated replay is missing: {row['dataset']} "
                f"(also tried {path.parent / candidate.name})"
            )
        datasets.append(candidate.resolve())
    if not datasets:
        raise ValueError(f"manifest contains no passed replay datasets: {path}")
    return datasets


def _argument_datasets(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(os.path.expanduser(value)))
        if not matches:
            matches = [value]
        for match in matches:
            path = Path(match).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"replay dataset does not exist: {path}")
            if path.name.endswith(".rejected.npz"):
                raise ValueError(f"refusing to visualize rejected replay: {path}")
            paths.append(path)
    return list(dict.fromkeys(paths))


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset:
        return _argument_datasets(args.dataset)
    return _manifest_datasets(Path(args.manifest).expanduser().resolve())


def _target_steps(env: TorqueReplayEnv, start_step: int, steps: int) -> int:
    if start_step >= env.dataset.n_steps:
        raise ValueError(
            f"--start-step={start_step} exceeds {env.dataset.motion_path} "
            f"with {env.dataset.n_steps} control steps"
        )
    available = env.dataset.n_steps - start_step
    return min(available, steps) if steps else available


def _replay_once(
    env: TorqueReplayEnv,
    *,
    start_step: int,
    steps: int,
    realtime_factor: float,
    viewer: Any = None,
    next_motion: threading.Event | None = None,
    catalog_position: tuple[int, int] | None = None,
) -> tuple[dict[str, object], str]:
    env.reset(start_step=start_step, dataset_index=0)
    dataset = env.dataset
    target_steps = _target_steps(env, start_step, steps)
    qpos_max = 0.0
    qvel_max = 0.0
    min_height = float(env.data.qpos[2])
    min_up_z = root_up_z(env.data.qpos)
    completed = 0
    fell = False
    outcome = "completed"

    while completed < target_steps:
        if viewer is not None and not viewer.is_running():
            outcome = "closed"
            break
        if next_motion is not None and next_motion.is_set():
            next_motion.clear()
            outcome = "switched"
            break

        started = time.perf_counter()
        _obs, _reward, terminated, truncated, info = env.step(np.zeros(4))
        completed += 1
        reference_index = start_step + completed
        qpos_max = max(
            qpos_max,
            float(
                np.max(
                    np.abs(env.data.qpos - dataset.rollout_qpos[reference_index])
                )
            ),
        )
        qvel_max = max(
            qvel_max,
            float(
                np.max(
                    np.abs(env.data.qvel - dataset.rollout_qvel[reference_index])
                )
            ),
        )
        min_height = min(min_height, float(env.data.qpos[2]))
        min_up_z = min(min_up_z, root_up_z(env.data.qpos))

        if viewer is not None:
            viewer.cam.lookat[:] = np.asarray(env.data.qpos[:3], dtype=np.float64)
            if catalog_position is not None and (completed == 1 or completed % 10 == 0):
                current, total = catalog_position
                viewer.set_texts(
                    (
                        None,
                        None,
                        f"Replay {current}/{total}\n{dataset.motion_path}\nSPACE: next motion",
                        f"step {start_step + completed}/{dataset.n_steps}",
                    )
                )
            viewer.sync()
            delay = (
                dataset.dt_control / realtime_factor
                - (time.perf_counter() - started)
            )
            if delay > 0:
                time.sleep(delay)

        if terminated or truncated:
            fell = bool(info["fell"])
            break

    report: dict[str, object] = {
        "motion": dataset.motion_path,
        "dataset": str(env.dataset_path),
        "start_step": start_step,
        "steps": completed,
        "completed_requested": completed == target_steps,
        "outcome": outcome,
        "fell": fell,
        "qpos_max_abs": qpos_max,
        "qvel_max_abs": qvel_max,
        "root_height_min": min_height,
        "root_up_min": min_up_z,
    }
    return report, outcome


def _check_only(
    paths: list[Path],
    args: argparse.Namespace,
    config: ReplayConfig,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    with TorqueReplayEnv(paths[0], args.checkpoint, config) as env:
        for repeat_index in range(args.repeat):
            for index, path in enumerate(paths):
                if index or repeat_index:
                    env.replace_dataset(path, start_step=args.start_step)
                report, _outcome = _replay_once(
                    env,
                    start_step=args.start_step,
                    steps=args.steps,
                    realtime_factor=args.realtime_factor,
                )
                report["repeat"] = repeat_index
                reports.append(report)
    return reports


def _interactive(
    paths: list[Path],
    args: argparse.Namespace,
    config: ReplayConfig,
) -> None:
    import mujoco.viewer

    next_motion = threading.Event()

    def key_callback(keycode: int) -> None:
        if keycode == SPACE_KEY:
            next_motion.set()

    index = args.start_index % len(paths)
    completed_loops = 0
    switches = 0
    with TorqueReplayEnv(paths[index], args.checkpoint, config) as env:
        viewer = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=key_callback,
        )
        try:
            with viewer.lock():
                viewer.cam.distance = 4.0
                viewer.cam.elevation = -15.0
            while viewer.is_running():
                path = paths[index]
                if path != env.dataset_path:
                    env.replace_dataset(path, start_step=args.start_step)
                print(
                    f"[{index + 1}/{len(paths)}] {env.dataset.motion_path} "
                    f"({env.dataset.n_steps} steps) | SPACE: next",
                    flush=True,
                )
                report, outcome = _replay_once(
                    env,
                    start_step=args.start_step,
                    steps=args.steps,
                    realtime_factor=args.realtime_factor,
                    viewer=viewer,
                    next_motion=next_motion,
                    catalog_position=(index + 1, len(paths)),
                )
                if outcome == "closed":
                    break
                if outcome == "switched":
                    switches += 1
                    index = (index + 1) % len(paths)
                    continue
                completed_loops += 1
                if report["fell"]:
                    print(
                        f"warning: replay fell at step "
                        f"{args.start_step + int(report['steps'])}; restarting",
                        flush=True,
                    )
        finally:
            viewer.close()
    print(
        json.dumps(
            {
                "closed": True,
                "catalog_size": len(paths),
                "space_switches": switches,
                "completed_loops": completed_loops,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Replay NPZ files or glob patterns; overrides --manifest",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Validation manifest used when --dataset is omitted",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="0 replays each motion to its end")
    parser.add_argument("--repeat", type=int, default=1, help="Used by --check-only")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run physics without opening a GUI",
    )
    args = parser.parse_args()
    if not args.check_only and not os.environ.get("DISPLAY"):
        raise SystemExit(
            "MuJoCo GUI requires a local DISPLAY; use --check-only on headless machines"
        )
    if (
        args.start_index < 0
        or args.start_step < 0
        or args.steps < 0
        or args.repeat <= 0
        or args.realtime_factor <= 0
    ):
        raise SystemExit(
            "--start-index/--start-step/--steps must be non-negative; "
            "--repeat/--realtime-factor must be positive"
        )

    paths = _dataset_paths(args)
    config = ReplayConfig(
        episode_steps=10_000_000,
        random_start=False,
        healthy_kp=0.0,
        healthy_kd=0.0,
        replay_mode="all",
        exact_baseline=True,
    )
    if args.check_only:
        reports = _check_only(paths, args, config)
        passed = all(
            row["completed_requested"]
            and not row["fell"]
            and float(row["qpos_max_abs"]) <= 2e-3
            and float(row["qvel_max_abs"]) <= 2e-2
            for row in reports
        )
        print(
            json.dumps(
                {"passed": passed, "check_only": True, "replays": reports},
                indent=2,
            )
        )
        if not passed:
            raise SystemExit(1)
        return

    _interactive(paths, args, config)


if __name__ == "__main__":
    main()
