#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CONFIG="${ROOT}/configs/train.yaml"
DATA_DIR="${ROOT}/data/fullbody_v1"
MANIFEST="${DATA_DIR}/manifest.json"
OUTPUT_ROOT="${ROOT}/outputs/a100_train_v1"
RUNTIME="${ROOT}/runtime"
LOCK="${RUNTIME}/a100_train.lock"
RUNNING="${RUNTIME}/a100_train_v1.running"
COMPLETE="${RUNTIME}/a100_train_v1.complete"
FAILED="${RUNTIME}/a100_train_v1.failed"

mkdir -p "${RUNTIME}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another Proknee training launcher owns ${LOCK}" >&2
  exit 4
fi

if [[ -f "${COMPLETE}" ]]; then
  echo "training already completed: ${COMPLETE}"
  exit 0
fi
if [[ -f "${FAILED}" ]]; then
  echo "previous run failed; inspect and archive ${FAILED} before retrying" >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite existing incomplete output: ${OUTPUT_ROOT}" >&2
  exit 1
fi

mapfile -t DATASETS < <(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.npz' -print | sort)
if [[ ${#DATASETS[@]} -ne 4 || ! -f "${MANIFEST}" ]]; then
  echo "expected four qualified datasets plus ${MANIFEST}" >&2
  exit 1
fi
"${PYTHON}" - "${MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("production") is not True or manifest.get("all_passed") is not True:
    raise SystemExit("manifest is not a fully qualified production manifest")
PY

"${PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5 6 7
if [[ "$(git -C "${REPO_ROOT}" branch --show-current)" != "muscle" ]]; then
  echo "A100 training must run from the muscle branch" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
rm -f "${RUNNING}"
printf 'started_at=%s\ngit_head=%s\ngpus=5,6,7\n' \
  "$(date -Is)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" >"${RUNNING}"

declare -a PIDS=()
declare -a SEEDS=(0 1 2)
declare -a GPUS=(5 6 7)

on_exit() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    for pid in "${PIDS[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    done
    printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${status}" >"${FAILED}"
  fi
  rm -f "${RUNNING}"
}
terminate_children() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  exit 143
}
trap on_exit EXIT
trap terminate_children INT TERM

for index in 0 1 2; do
  gpu="${GPUS[$index]}"
  seed="${SEEDS[$index]}"
  output="${OUTPUT_ROOT}/seed_${seed}"
  mkdir -p "${output}"
  "${ROOT}/scripts/a100_exec_gpu.sh" "${gpu}" "${PYTHON}" \
    "${ROOT}/scripts/train_policy.py" \
    --config "${CONFIG}" \
    --dataset "${DATASETS[@]}" \
    --output "${output}" \
    --seed "${seed}" >"${output}/train.log" 2>&1 &
  PIDS+=("$!")
  printf 'seed_%s_pid=%s\nseed_%s_gpu=%s\n' "${seed}" "$!" "${seed}" "${gpu}" >>"${RUNNING}"
done

remaining=${#PIDS[@]}
while ((remaining > 0)); do
  if ! wait -n; then
    echo "one training seed failed; stopping only this launcher's remaining seed processes" >&2
    exit 1
  fi
  ((remaining -= 1))
done

printf 'completed_at=%s\ngit_head=%s\ngpus=5,6,7\n' \
  "$(date -Is)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" >"${COMPLETE}"
rm -f "${RUNNING}"
trap - EXIT INT TERM
echo "all three training seeds completed: ${OUTPUT_ROOT}"
