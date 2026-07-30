#!/usr/bin/env python3
"""Compile the portable replay XML with the active MuJoCo version."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument(
        "--asset-root",
        required=True,
        help="musclemimic_models/model directory containing meshes/",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.xml).resolve()
    asset_root = Path(args.asset_root).resolve()
    output = Path(args.output).resolve()
    if not (asset_root / "meshes").is_dir():
        raise FileNotFoundError(f"missing model mesh directory: {asset_root / 'meshes'}")
    tree = ET.parse(source)
    compiler = tree.getroot().find("compiler")
    if compiler is None:
        raise ValueError("portable XML has no compiler element")
    compiler.set("meshdir", str(asset_root))
    compiler.set("texturedir", str(asset_root))
    xml = ET.tostring(tree.getroot(), encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    leakage = (
        float(np.max(np.abs(data.qfrc_actuator)))
        if data.qfrc_actuator.size
        else 0.0
    )
    if leakage > 1e-10:
        raise RuntimeError(f"disabled actuator leakage is {leakage}")
    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(output))
    metadata = {
        "mujoco_version": mujoco.__version__,
        "source_xml": str(source),
        "source_xml_sha256": _sha256(source),
        "asset_root": str(asset_root),
        "output": str(output),
        "output_sha256": _sha256(output),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "ntendon": model.ntendon,
        "nsite": model.nsite,
        "actuator_gain_max_abs": float(np.max(np.abs(model.actuator_gainprm))),
        "actuator_bias_max_abs": float(np.max(np.abs(model.actuator_biasprm))),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
