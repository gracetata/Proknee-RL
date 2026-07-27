#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
RUNTIME="${ROOT}/runtime"
SESSION="proknee-a100-train"

mkdir -p "${RUNTIME}"
date -Is >"${RUNTIME}/watchdog_last_check"

if [[ -f "${RUNTIME}/a100_train_v3.complete" || -f "${RUNTIME}/a100_train_v3.failed" \
  || -f "${RUNTIME}/a100_pipeline.failed" ]]; then
  exit 0
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  exit 0
fi
if ! "${PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5 \
  >"${RUNTIME}/gpu_guard_latest.json" 2>&1; then
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && exec bash '${ROOT}/scripts/run_production_pipeline_a100.sh' >>'${RUNTIME}/pipeline.log' 2>&1"
