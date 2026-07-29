#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BUILD="${ROOT}/runtime/hora_env_build"
BUNDLE="${ROOT}/runtime/hora_torch_py311_cu126_minimal.tar.zst"
UV="${UV:-$(command -v uv)}"

rm -rf "${BUILD}"
"${UV}" venv --python 3.11 "${BUILD}"
"${UV}" pip install \
  --python "${BUILD}/bin/python" \
  --index https://download.pytorch.org/whl/cu126 \
  --default-index https://pypi.org/simple \
  -r "${ROOT}/configs/hora_requirements.txt"
"${BUILD}/bin/python" - <<'PY'
import mujoco
import torch
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"mujoco={mujoco.__version__}"
)
assert torch.version.cuda == "12.6"
PY
SITE="${BUILD}/lib/python3.11/site-packages"
tar --zstd -C "${SITE}" -cf "${BUNDLE}" \
  torch torch-2.7.1+cu126.dist-info functorch torchgen \
  sympy sympy-1.14.0.dist-info mpmath mpmath-1.3.0.dist-info \
  networkx networkx-3.6.1.dist-info \
  jinja2 jinja2-3.1.6.dist-info markupsafe markupsafe-3.0.3.dist-info \
  typing_extensions.py typing_extensions-4.15.0.dist-info
sha256sum "${BUNDLE}" >"${BUNDLE}.sha256"
echo "created ${BUNDLE}"
