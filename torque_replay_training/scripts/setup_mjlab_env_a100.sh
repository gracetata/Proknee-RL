#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${PYTHON:-/workspace/.tools/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}"
VENV="${REPO_ROOT}/.venv-mjlab"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing Python 3.11 runtime: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON}" -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
"${VENV}/bin/python" -m pip install \
  -r "${ROOT}/configs/mjlab_requirements.txt"
"${VENV}/bin/python" -m pip install -e "${ROOT}"
"${VENV}/bin/python" - <<'PY'
import mjlab
import mujoco
import mujoco_warp
import torch
import warp
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("mujoco", mujoco.__version__)
print("mujoco_warp", getattr(mujoco_warp, "__version__", "unknown"))
print("warp", warp.config.version)
print("mjlab", getattr(mjlab, "__version__", "1.5.3"))
PY
