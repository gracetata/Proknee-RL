#!/usr/bin/env python3
"""Evaluate a Warp-v2 policy in CPU MuJoCo with replay beta forced to zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import mujoco
import numpy as np
import torch

from torque_replay_training.compact_schema import CompactTorqueReplayDataset
from torque_replay_training.control import root_up_z
from torque_replay_training.hora.models import RunningMeanStd
from torque_replay_training.warp_v2.config import load_training_config
from torque_replay_training.warp_v2.observations import (
    ObservationHistory,
    actor_frame,
    quaternion_rotate_inverse,
)
from torque_replay_training.warp_v2.ppo import AsymmetricActorCritic


def _paths(split_path: Path, data_dir: Path, group: str) -> list[Path]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return [data_dir / row["dataset_basename"] for row in split[group]]


def _command(dataset: CompactTorqueReplayDataset, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = torch.from_numpy(dataset.rollout_qpos[step : step + 1]).float()
    qvel = torch.from_numpy(dataset.rollout_qvel[step : step + 1]).float()
    linear = quaternion_rotate_inverse(qpos[:, 3:7], qvel[:, :3])
    angular = quaternion_rotate_inverse(qpos[:, 3:7], qvel[:, 3:6])
    return linear, angular[:, 2]


def _frame(
    data: mujoco.MjData,
    dataset: CompactTorqueReplayDataset,
    step: int,
    prosthesis_qpos: torch.Tensor,
    prosthesis_dofs: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    command_velocity, command_yaw = _command(dataset, step)
    return actor_frame(
        torch.from_numpy(data.qpos[None]).float(),
        torch.from_numpy(data.qvel[None]).float(),
        prosthesis_qpos,
        prosthesis_dofs,
        command_velocity,
        command_yaw,
        previous_action,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--group", choices=("train", "validation"), default="validation")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--full-trajectories", action="store_true")
    args = parser.parse_args()

    config = load_training_config(args.config)
    paths = _paths(
        Path(args.split),
        Path(args.data_dir),
        args.group,
    )
    datasets = [CompactTorqueReplayDataset.load(path) for path in paths]
    model = mujoco.MjModel.from_binary_path(str(Path(args.model).resolve()))
    checkpoint = torch.load(
        Path(args.policy),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("run_metadata", {}).get("behavior_cloning") is not False:
        raise ValueError("checkpoint does not declare behavior_cloning=false")
    if checkpoint.get("run_metadata", {}).get("imitation_reward") is not False:
        raise ValueError("checkpoint does not declare imitation_reward=false")
    if float(checkpoint.get("replay_beta", -1.0)) != 0.0:
        raise ValueError(
            "formal evaluation requires a checkpoint saved with replay_beta=0"
        )

    actor = AsymmetricActorCritic(
        96,
        96 + model.nq + model.nv + 4,
        4,
        config.ppo.actor_units,
        config.ppo.critic_units,
        config.ppo.initial_log_std,
    )
    actor.load_state_dict(checkpoint["model"])
    actor.eval()
    normalizer = RunningMeanStd((96,))
    normalizer.load_state_dict(checkpoint["actor_mean_std"])
    normalizer.eval()
    first = datasets[0]
    prosthesis_dofs_np = np.asarray(
        first.metadata["prosthesis_dof_indices"],
        dtype=np.int32,
    )
    prosthesis_dofs = torch.from_numpy(prosthesis_dofs_np.astype(np.int64))
    prosthesis_qpos = torch.tensor(
        first.metadata["prosthesis_qpos_indices"],
        dtype=torch.long,
    )
    torque_limits = torch.tensor(config.environment.torque_limits)
    rng = np.random.default_rng(args.seed)
    trials: list[tuple[int, int, int]]
    if args.full_trajectories:
        trials = [
            (index, 0, dataset.n_steps)
            for index, dataset in enumerate(datasets)
        ]
    else:
        trials = []
        for _ in range(args.episodes):
            index = int(rng.integers(0, len(datasets)))
            dataset = datasets[index]
            length = min(args.episode_steps, dataset.n_steps)
            maximum = max(0, dataset.n_steps - length)
            start = int(rng.integers(0, maximum + 1)) if maximum else 0
            trials.append((index, start, length))

    falls = 0
    completed = 0
    min_heights: list[float] = []
    min_up_z: list[float] = []
    saturation: list[float] = []
    slip_speeds: list[float] = []
    foot_bodies = {
        side: {
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in (f"talus_{side}", f"calcn_{side}", f"toes_{side}")
        }
        for side in ("l", "r")
    }
    foot_geoms = {
        side: {
            index
            for index, body in enumerate(model.geom_bodyid)
            if int(body) in bodies
        }
        for side, bodies in foot_bodies.items()
    }
    representative_body = {
        side: int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"calcn_{side}",
            )
        )
        for side in ("l", "r")
    }

    for dataset_index, start, length in trials:
        dataset = datasets[dataset_index]
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:] = dataset.rollout_qpos[start]
        data.qvel[:] = dataset.rollout_qvel[start]
        data.qacc_warmstart[:] = dataset.qacc_warmstart[start]
        data.time = start * dataset.dt_control
        mujoco.mj_forward(model, data)
        previous_action = torch.zeros(1, 4)
        history = ObservationHistory(1, device=torch.device("cpu"))
        initial = _frame(
            data,
            dataset,
            start,
            prosthesis_qpos,
            prosthesis_dofs,
            previous_action,
        )
        history.reset(torch.ones(1, dtype=torch.bool), initial)
        episode_fell = False
        episode_min_height = float(data.qpos[2])
        episode_min_up = root_up_z(data.qpos)
        for local_step in range(length):
            step = start + local_step
            with torch.no_grad():
                policy_action = actor.act_inference(
                    normalizer(history.observation())
                )
            # beta is deliberately and unconditionally zero here.
            torque = policy_action[0] * torque_limits
            saturation.append(float((policy_action.abs() > 0.999).float().mean()))
            for substep in range(dataset.n_substeps):
                data.ctrl[:] = 0.0
                if data.act.size:
                    data.act[:] = 0.0
                data.qfrc_applied[:] = dataset.qfrc_actuator[step, substep]
                data.qfrc_applied[prosthesis_dofs_np] = torque.numpy()
                mujoco.mj_step(model, data)
            episode_min_height = min(episode_min_height, float(data.qpos[2]))
            episode_min_up = min(episode_min_up, root_up_z(data.qpos))
            for side in ("l", "r"):
                contact = any(
                    int(row.geom[0]) in foot_geoms[side]
                    or int(row.geom[1]) in foot_geoms[side]
                    for row in data.contact[: data.ncon]
                )
                if contact:
                    slip_speeds.append(
                        float(
                            np.linalg.norm(
                                data.cvel[representative_body[side], 3:5]
                            )
                        )
                    )
            if (
                data.qpos[2] < config.environment.fall_height
                or root_up_z(data.qpos) < config.environment.fall_up_z
                or not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
            ):
                falls += 1
                episode_fell = True
                break
            previous_action = policy_action
            next_step = min(step + 1, dataset.n_steps)
            history.append(
                _frame(
                    data,
                    dataset,
                    next_step,
                    prosthesis_qpos,
                    prosthesis_dofs,
                    previous_action,
                )
            )
        if not episode_fell:
            completed += 1
        min_heights.append(episode_min_height)
        min_up_z.append(episode_min_up)

    report = {
        "backend": "CPU MuJoCo",
        "replay_beta": 0.0,
        "episodes": len(trials),
        "completed": completed,
        "falls": falls,
        "fall_rate": falls / max(1, len(trials)),
        "minimum_root_height": min(min_heights),
        "minimum_root_up_z": min(min_up_z),
        "mean_action_saturation": float(np.mean(saturation)),
        "foot_slip_p95": (
            float(np.percentile(slip_speeds, 95)) if slip_speeds else None
        ),
        "policy": str(Path(args.policy).resolve()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if falls:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
