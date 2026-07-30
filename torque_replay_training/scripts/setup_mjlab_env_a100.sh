#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${PYTHON:-/workspace/.tools/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}"
VENV="${REPO_ROOT}/.venv-mjlab"
HORA_VENV="${REPO_ROOT}/.venv-hora"

if [[ ! -x "${PYTHON}" && ! -x "${HORA_VENV}/bin/python" ]]; then
  echo "missing Python 3.11 runtime: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  if [[ -x "${HORA_VENV}/bin/python" ]] &&
    "${HORA_VENV}/bin/python" -c \
      'import torch; assert torch.__version__.startswith("2.7.1+")'; then
    echo "reusing the validated A100 PyTorch 2.7.1 environment with copy-on-write"
    cp -a --reflink=auto "${HORA_VENV}" "${VENV}"
  else
    "${PYTHON}" -m venv "${VENV}"
  fi
fi
if ! "${VENV}/bin/python" -c \
  'import torch; assert torch.__version__.startswith("2.7.1+")'; then
  "${VENV}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1
fi
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
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
