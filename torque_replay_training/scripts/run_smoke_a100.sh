#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
source "${ROOT}/scripts/a100_env.sh"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing environment: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" - <<'PY'
import os
import jax

visible = os.environ.get("CUDA_VISIBLE_DEVICES")
devices = jax.devices()
print(f"CUDA_VISIBLE_DEVICES={visible}")
print(f"JAX devices={devices}")
if visible != "5":
    raise SystemExit("A100 smoke must currently use physical GPU 5 only")
if not devices or devices[0].platform != "gpu":
    raise SystemExit("JAX CUDA backend is not available")
if len(devices) != 1:
    raise SystemExit(f"expected one visible JAX GPU, got {devices}")
PY

bash "${ROOT}/scripts/run_smoke.sh"
