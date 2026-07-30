#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv-mjlab/bin/python"
MODEL_ROOT="${MUSCLEMIMIC_MODEL_ROOT:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/musclemimic_models/model}"

"${REPO_ROOT}/.venv/bin/python" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/train_mjlab_prosthesis.py" \
  --split "${ROOT}/configs/flat_walk_split.json" \
  --data-dir "${ROOT}/data/flat_walk_compact_v1" \
  --model-xml "${ROOT}/data/replay_model/musclemimic_replay.xml" \
  --model-root "${MODEL_ROOT}" \
  --group train --num-envs 4096 --episode-steps 512 \
  --max-iterations 3000 --seed 0 --device cuda:0 \
  --log-root "${ROOT}/outputs/mjlab_a100" --run-name seed_0
