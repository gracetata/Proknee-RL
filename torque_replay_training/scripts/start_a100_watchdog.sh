#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="proknee-a100-watchdog"
INTERVAL="${A100_WATCHDOG_INTERVAL_SECONDS:-120}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "watchdog already running in tmux session ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "while true; do bash '${ROOT}/scripts/a100_training_watchdog.sh'; sleep '${INTERVAL}'; done"
echo "started ${SESSION}; interval=${INTERVAL}s"
