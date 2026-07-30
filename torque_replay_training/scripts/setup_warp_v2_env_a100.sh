#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
VENV="${REPO_ROOT}/.venv-warp"
PYTHON="${VENV}/bin/python"
UV="${UV:-/root/.local/bin/uv}"
PYTHON311="${PYTHON311:-/workspace/.tools/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}"

if [[ ! -x "${UV}" ]]; then
  UV="$(command -v uv || true)"
fi
if [[ -n "${UV}" && -x "${UV}" ]]; then
  if [[ ! -x "${PYTHON}" ]]; then
    "${UV}" venv --python 3.11 "${VENV}"
  fi
  "${UV}" pip install --python "${PYTHON}" \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r "${ROOT}/configs/warp_v2_requirements.txt"
  "${UV}" pip install --python "${PYTHON}" --no-deps -e "${ROOT}"
else
  if [[ ! -x "${PYTHON311}" ]]; then
    echo "missing Python 3.11 runtime: ${PYTHON311}" >&2
    exit 1
  fi
  if [[ ! -x "${PYTHON}" ]]; then
    "${PYTHON311}" -m venv "${VENV}"
  fi
  # The A100 workspace already has the pinned CUDA Torch, Warp and MJWarp
  # wheels in .venv. Reuse those immutable packages through a .pth file and
  # install only the conflicting MuJoCo version locally. This avoids a
  # multi-gigabyte CUDA download on servers without uv.
  SHARED_WARP_SITE="$("${REPO_ROOT}/.venv/bin/python" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
  SHARED_TORCH_SITE="$("${REPO_ROOT}/.venv-hora/bin/python" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
  LOCAL_SITE="$("${PYTHON}" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
  printf '%s\n%s\n' \
    "${SHARED_WARP_SITE}" \
    "${SHARED_TORCH_SITE}" \
    >"${LOCAL_SITE}/proknee_shared_runtime.pth"
  "${PYTHON}" -m pip install --no-deps mujoco==3.5.0
  "${PYTHON}" -m pip install --no-deps -e "${ROOT}"
fi
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
