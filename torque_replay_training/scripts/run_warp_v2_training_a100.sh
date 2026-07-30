#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv-warp/bin/python"
CONFIG="${ROOT}/configs/flat_walk_warp_v2.yaml"
SPLIT="${ROOT}/configs/flat_walk_split.json"
MANIFEST="${ROOT}/configs/flat_walk_compact_manifest.json"
DATA_DIR="${ROOT}/data/flat_walk_compact_v1"
MODEL="${ROOT}/data/replay_model/musclemimic_replay_warp_3_5.mjb"
OUTPUT_ROOT="${ROOT}/outputs/a100_flat_walk_warp_v2"
OUTPUT="${OUTPUT_ROOT}/seed_0"
RUNTIME="${ROOT}/runtime"
RUNNING="${RUNTIME}/a100_flat_walk_warp_v2.running"
COMPLETE="${RUNTIME}/a100_flat_walk_warp_v2.complete"
FAILED="${RUNTIME}/a100_flat_walk_warp_v2.failed"
LOCK="${RUNTIME}/a100_flat_walk_warp_v2.lock"

mkdir -p "${RUNTIME}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another warp-v2 launcher owns ${LOCK}" >&2
  exit 4
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "missing ${PYTHON}; run setup_warp_v2_env_a100.sh" >&2
  exit 1
fi
if [[ "$(git -C "${REPO_ROOT}" branch --show-current)" != "muscle" ]]; then
  echo "formal training must run from branch muscle" >&2
  exit 1
fi
"${REPO_ROOT}/.venv/bin/python" \
  "${ROOT}/scripts/verify_compact_flat_walk_data.py" \
  --manifest "${MANIFEST}" \
  --data-dir "${DATA_DIR}"
"${REPO_ROOT}/.venv/bin/python" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
mkdir -p "${OUTPUT_ROOT}"
printf 'started_at=%s\ngit_head=%s\nphysical_gpu=5\nnum_envs=4096\npid=%s\n' \
  "$(date -Is)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" "$$" >"${RUNNING}"

on_exit() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${status}" >"${FAILED}"
  fi
  unlink "${RUNNING}" 2>/dev/null || true
}
trap on_exit EXIT

"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/train_warp_v2.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --manifest "${MANIFEST}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --seed 0 >"${OUTPUT_ROOT}/train.log" 2>&1

CHECKPOINT="${OUTPUT}/stage1_nn/last.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "training ended without ${CHECKPOINT}" >&2
  exit 1
fi
"${PYTHON}" "${ROOT}/scripts/evaluate_warp_v2_policy.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --policy "${CHECKPOINT}" \
  --group validation \
  --full-trajectories >"${OUTPUT_ROOT}/validation_full_trajectories.json"
"${PYTHON}" "${ROOT}/scripts/evaluate_warp_v2_policy.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --policy "${CHECKPOINT}" \
  --group validation \
  --episodes 1000 >"${OUTPUT_ROOT}/validation_random_1000.json"
printf 'completed_at=%s\ngit_head=%s\nphysical_gpu=5\ncheckpoint=%s\n' \
  "$(date -Is)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
  "${CHECKPOINT}" >"${COMPLETE}"
unlink "${RUNNING}" 2>/dev/null || true
trap - EXIT
echo "warp-v2 training completed: ${CHECKPOINT}"
