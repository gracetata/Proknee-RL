#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
GUARD_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON="${REPO_ROOT}/.venv-hora/bin/python"
CONFIG="${ROOT}/configs/flat_walk_hora_smoke.yaml"
SPLIT="${ROOT}/configs/flat_walk_split.json"
MANIFEST="${ROOT}/configs/flat_walk_compact_manifest.json"
DATA_DIR="${ROOT}/data/flat_walk_compact_v1"
MODEL="${ROOT}/data/replay_model/musclemimic_replay.mjb"
OUTPUT="${FLAT_WALK_SMOKE_OUTPUT:-/tmp/proknee_flat_walk_smoke_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing HORA environment: ${PYTHON}" >&2
  exit 1
fi

"${GUARD_PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/train_hora_policy.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --manifest "${MANIFEST}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --output "${OUTPUT}/seed_0" \
  --device cuda:0 \
  --limit-datasets 2 \
  --seed 0 | tee "${OUTPUT}.console.log"
LATEST_POLICY="${OUTPUT}/seed_0/stage1_nn/last.pth"
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/evaluate_hora_policy.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --group train \
  --limit-datasets 2 \
  --policy "${LATEST_POLICY}" \
  --device cuda:0 \
  --episodes 4 | tee "${OUTPUT}/evaluation.json"
echo "flat-walk smoke output: ${OUTPUT}"
