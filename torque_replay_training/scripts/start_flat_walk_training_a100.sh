#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SESSION="proknee-a100-flat-walk"
LOG="${ROOT}/runtime/a100_flat_walk_v1.launch.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "${SESSION} is already running"
  exit 0
fi
"${PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
mkdir -p "${ROOT}/runtime"
tmux new-session -d -s "${SESSION}" \
  "cd '${REPO_ROOT}' && exec bash '${ROOT}/scripts/run_flat_walk_training_a100.sh' >>'${LOG}' 2>&1"
echo "started tmux session ${SESSION} on physical GPU 5"
