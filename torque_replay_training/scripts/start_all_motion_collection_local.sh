#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="proknee-local-all-replays"
PYTHON="${ROOT}/../.venv/bin/python"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 0
fi

"${PYTHON}" "${ROOT}/scripts/local_4090_guard.py"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}/..' && bash '${ROOT}/scripts/collect_all_available_local.sh'"
echo "started tmux session: ${SESSION}"
