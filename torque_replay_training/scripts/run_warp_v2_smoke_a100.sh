#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv-warp/bin/python"
CONFIG="${ROOT}/configs/flat_walk_warp_v2_smoke.yaml"
SPLIT="${ROOT}/configs/flat_walk_split.json"
MANIFEST="${ROOT}/configs/flat_walk_compact_manifest.json"
DATA_DIR="${ROOT}/data/flat_walk_compact_v1"
MODEL="${ROOT}/data/replay_model/musclemimic_replay_warp_3_5.mjb"
OUTPUT="${ROOT}/outputs/a100_flat_walk_warp_v2_smoke"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_AGENT_STEPS="${MAX_AGENT_STEPS:-65536}"

"${REPO_ROOT}/.venv/bin/python" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite smoke output: ${OUTPUT}" >&2
  exit 1
fi
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/train_warp_v2.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --manifest "${MANIFEST}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --num-envs "${NUM_ENVS}" \
  --max-agent-steps "${MAX_AGENT_STEPS}" \
  --limit-datasets 4
