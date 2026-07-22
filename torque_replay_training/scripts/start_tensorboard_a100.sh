#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
TENSORBOARD="${REPO_ROOT}/.venv/bin/tensorboard"
LOGDIR="${ROOT}/outputs"
RUNTIME="${ROOT}/runtime"
PORT="${TENSORBOARD_PORT:-6011}"
SESSION="proknee-tensorboard-${PORT}"

mkdir -p "${LOGDIR}" "${RUNTIME}"
if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
  echo "TensorBoard is already available at 127.0.0.1:${PORT}"
  exit 0
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "${SESSION} exists but HTTP is not ready; inspect ${RUNTIME}/tensorboard_${PORT}.log" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "exec '${TENSORBOARD}' --logdir '${LOGDIR}' --host 127.0.0.1 --port '${PORT}' --reload_interval 10 >'${RUNTIME}/tensorboard_${PORT}.log' 2>&1"

for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "TensorBoard started at 127.0.0.1:${PORT}; logdir=${LOGDIR}"
    exit 0
  fi
  sleep 1
done
echo "TensorBoard failed to become ready; inspect ${RUNTIME}/tensorboard_${PORT}.log" >&2
exit 1
