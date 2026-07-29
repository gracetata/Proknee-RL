#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BUILD="${ROOT}/runtime/hora_env_build"
BUNDLE="${ROOT}/runtime/hora_site_packages_py311_cu126.tar.zst"
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
tar --zstd -C "${BUILD}/lib/python3.11" \
  -cf "${BUNDLE}" site-packages
sha256sum "${BUNDLE}" >"${BUNDLE}.sha256"
echo "created ${BUNDLE}"
