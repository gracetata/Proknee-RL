#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
RUNTIME="${ROOT}/runtime"
PIPELINE_FAILED="${RUNTIME}/a100_pipeline.failed"

mkdir -p "${RUNTIME}"
record_failure() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${status}" >"${PIPELINE_FAILED}"
  fi
}
trap record_failure EXIT

"${PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5 6 7
bash "${ROOT}/scripts/collect_production_a100.sh"
# Collection runs only on physical GPU 5. Recheck all three immediately before training.
"${PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5 6 7
bash "${ROOT}/scripts/run_training_a100.sh"
trap - EXIT
