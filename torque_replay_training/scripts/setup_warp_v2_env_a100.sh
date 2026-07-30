#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
VENV="${REPO_ROOT}/.venv-warp"
PYTHON="${VENV}/bin/python"
UV="${UV:-/root/.local/bin/uv}"

if [[ ! -x "${UV}" ]]; then
  UV="$(command -v uv)"
fi
if [[ ! -x "${PYTHON}" ]]; then
  "${UV}" venv --python 3.11 "${VENV}"
fi
"${UV}" pip install --python "${PYTHON}" \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r "${ROOT}/configs/warp_v2_requirements.txt"
"${UV}" pip install --python "${PYTHON}" --no-deps -e "${ROOT}"
SOURCE_XML="${ROOT}/data/replay_model/musclemimic_replay.xml"
WARP_MODEL="${ROOT}/data/replay_model/musclemimic_replay_warp_3_5.mjb"
if [[ ! -f "${SOURCE_XML}" ]]; then
  echo "missing portable replay XML: ${SOURCE_XML}" >&2
  exit 1
fi
ASSET_ROOT="$("${REPO_ROOT}/.venv/bin/python" - <<'PY'
from pathlib import Path
from musclemimic_models import get_xml_path

print(Path(get_xml_path("myofullbody")).resolve().parents[1])
PY
)"
"${PYTHON}" "${ROOT}/scripts/compile_warp_replay_model.py" \
  --xml "${SOURCE_XML}" \
  --asset-root "${ASSET_ROOT}" \
  --output "${WARP_MODEL}"
"${PYTHON}" - <<'PY'
import mujoco
import mujoco_warp
import torch
import warp

expected = {
    "mujoco": "3.5.0",
    "mujoco_warp": "3.5.0",
    "warp": "1.12.1",
}
actual = {
    "mujoco": mujoco.__version__,
    "mujoco_warp": mujoco_warp.__version__,
    "warp": warp.__version__,
}
if actual != expected:
    raise SystemExit(f"version mismatch: {actual} != {expected}")
if torch.__version__ != "2.7.1+cu126":
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
print({"versions": actual, "torch": torch.__version__})
PY
