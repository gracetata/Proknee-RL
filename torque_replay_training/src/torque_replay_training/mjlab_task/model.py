"""Build the MuscleMimic model as an MJLAB entity."""

from __future__ import annotations

import os
import re
from pathlib import Path

import mujoco


def resolve_model_root(explicit: str | Path | None = None) -> Path:
    """Resolve the directory containing ``meshes/`` for the MuscleMimic model."""

    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MUSCLEMIMIC_MODEL_ROOT"):
        candidates.append(Path(os.environ["MUSCLEMIMIC_MODEL_ROOT"]))
    candidates.extend(
        (
            Path(
                "/home/user/Workspace/musclemimic/musclemimic/.venv/lib/"
                "python3.11/site-packages/musclemimic_models/model"
            ),
            Path(
                "/workspace/Proknee-RL-muscle/.venv/lib/python3.11/"
                "site-packages/musclemimic_models/model"
            ),
        )
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "meshes").is_dir():
            return root
    rendered = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate MuscleMimic model assets. Set "
        "MUSCLEMIMIC_MODEL_ROOT to the directory containing meshes/.\n"
        f"Checked:\n{rendered}"
    )


def load_replay_spec(
    xml_path: str | Path,
    model_root: str | Path | None = None,
) -> mujoco.MjSpec:
    """Load the exported XML and disable every original actuator in the spec."""

    xml_path = Path(xml_path).resolve()
    root = resolve_model_root(model_root)
    text = xml_path.read_text(encoding="utf-8")
    text = text.replace('meshdir="../"', f'meshdir="{root.as_posix()}"')
    text = text.replace('texturedir="../"', f'texturedir="{root.as_posix()}"')
    text = re.sub(
        r'file="[^"]*musclemimic_models/model/([^"]+)"',
        lambda match: f'file="{(root / match.group(1)).as_posix()}"',
        text,
    )
    spec = mujoco.MjSpec.from_string(text)
    for actuator in spec.actuators:
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gainprm[:] = 0.0
        actuator.biasprm[:] = 0.0
        actuator.dynprm[:] = 0.0
    return spec
