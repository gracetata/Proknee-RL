#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BUNDLE="${1:-${ROOT}/runtime/hora_site_packages_py311_cu126.tar.zst}"
TARGET="${REPO_ROOT}/.venv-hora"

if [[ ! -f "${BUNDLE}" ]]; then
  echo "missing offline HORA environment bundle: ${BUNDLE}" >&2
  exit 1
fi
if [[ -e "${TARGET}" ]]; then
  echo "refusing to overwrite existing environment: ${TARGET}" >&2
  exit 1
fi
"${REPO_ROOT}/.venv/bin/python" -m venv --without-pip "${TARGET}"
tar --zstd -C "${TARGET}/lib/python3.11" -xf "${BUNDLE}"
"${TARGET}/bin/python" - <<'PY'
import mujoco
import torch
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"cuda_available={torch.cuda.is_available()} mujoco={mujoco.__version__}"
)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
PY
