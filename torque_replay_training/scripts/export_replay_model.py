#!/usr/bin/env python3
"""Export the actuator-disabled MuJoCo model used by HORA vector replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco

import _bootstrap  # noqa: F401

from torque_replay_training.paths import DEFAULT_CHECKPOINT
from torque_replay_training.replay_env import ReplayConfig, TorqueReplayEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--xml-output",
        help="optional portable XML for recompilation by a newer MuJoCo",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = ReplayConfig(
        random_start=False,
        replay_mode="all",
        exact_baseline=True,
    )
    with TorqueReplayEnv(args.dataset, args.checkpoint, config) as env:
        mujoco.mj_saveModel(env.model, str(output))
        xml_output = (
            Path(args.xml_output).resolve() if args.xml_output else None
        )
        if xml_output is not None:
            xml_output.parent.mkdir(parents=True, exist_ok=True)
            spec = env._source_env.mjspec  # noqa: SLF001 - export owns this env
            spec.to_file(str(xml_output))
            if not xml_output.is_file() or xml_output.stat().st_size == 0:
                raise RuntimeError("MuJoCo did not create a portable XML model")
        metadata = {
            "model": str(output),
            "sha256": _sha256(output),
            "nq": env.model.nq,
            "nv": env.model.nv,
            "nu": env.model.nu,
            "njnt": env.model.njnt,
            "timestep": env.model.opt.timestep,
            "source_dataset": str(Path(args.dataset).resolve()),
            "source_motion": env.dataset.motion_path,
            "actuator_gain_max_abs": float(abs(env.model.actuator_gainprm).max()),
            "actuator_bias_max_abs": float(abs(env.model.actuator_biasprm).max()),
            "xml_model": str(xml_output) if xml_output else None,
            "xml_sha256": _sha256(xml_output) if xml_output else None,
        }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
