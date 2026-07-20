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
if visible != "5,6,7":
    raise SystemExit("A100 smoke must use physical GPUs 5-7 only")
if not devices or devices[0].platform != "gpu":
    raise SystemExit("JAX CUDA backend is not available")
PY

bash "${ROOT}/scripts/run_smoke.sh"

