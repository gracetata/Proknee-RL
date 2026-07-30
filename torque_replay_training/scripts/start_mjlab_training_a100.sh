#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
SESSION="proknee-a100-mjlab"
LOG="${ROOT}/runtime/a100_mjlab.launch.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "${SESSION} is already running"
  exit 0
fi
"${REPO_ROOT}/.venv/bin/python" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
mkdir -p "${ROOT}/runtime"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && exec bash '${ROOT}/scripts/run_mjlab_training_a100.sh' >>'${LOG}' 2>&1"
echo "started ${SESSION} on physical GPU 5"
