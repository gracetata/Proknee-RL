#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv-mjlab/bin/python"
MODEL_ROOT="${MUSCLEMIMIC_MODEL_ROOT:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/musclemimic_models/model}"

"${REPO_ROOT}/.venv/bin/python" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/audit_mjlab_replay.py" \
  --split "${ROOT}/configs/flat_walk_split.json" \
  --data-dir "${ROOT}/data/flat_walk_compact_v1" \
  --model-xml "${ROOT}/data/replay_model/musclemimic_replay.xml" \
  --model-root "${MODEL_ROOT}" --device cuda:0 \
  --group validation --num-envs 64 --episode-steps 512 --steps 512 \
  --healthy-kp 50 --healthy-kd 5 --healthy-correction-limit 300 \
  --root-position-kp 5000 --root-velocity-kd 1000 --root-force-limit 10000 \
  --root-orientation-kp 1000 --root-angular-velocity-kd 100 \
  --root-torque-limit 1000 --max-fall-rate 0 --max-tracking-error 5 \
  --output "${ROOT}/outputs/mjlab_a100_replay_gate.json"

for _ in $(seq 1 12); do
  if "${REPO_ROOT}/.venv/bin/python" \
    "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5; then
    break
  fi
  echo "GPU 5 still has transient activity after replay gate; waiting 5 seconds"
  sleep 5
done
"${REPO_ROOT}/.venv/bin/python" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/train_mjlab_prosthesis.py" \
  --split "${ROOT}/configs/flat_walk_split.json" \
  --data-dir "${ROOT}/data/flat_walk_compact_v1" \
  --model-xml "${ROOT}/data/replay_model/musclemimic_replay.xml" \
  --model-root "${MODEL_ROOT}" --device cuda:0 \
  --group validation --num-envs "${NUM_ENVS:-4096}" --episode-steps 64 \
  --max-iterations "${MAX_ITERATIONS:-1}" --seed 0 \
  --log-root "${ROOT}/outputs/mjlab_a100_smoke" --run-name smoke
