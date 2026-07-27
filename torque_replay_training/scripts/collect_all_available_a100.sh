#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
MOTION_LIST="${ROOT}/configs/all_available_motions.txt"
OUTPUT="${ROOT}/data/fullbody_all_v3"
LOG_DIR="${ROOT}/runtime"
CACHE_ROOT="${HOME}/.musclemimic/caches/AMASS/MyoFullBody/gmr"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing Python environment: ${PYTHON}" >&2
  exit 1
fi

missing=0
while IFS= read -r motion; do
  [[ -z "${motion}" || "${motion}" == \#* ]] && continue
  if [[ ! -f "${CACHE_ROOT}/${motion}.npz" ]]; then
    echo "missing GMR cache: ${CACHE_ROOT}/${motion}.npz" >&2
    missing=$((missing + 1))
  fi
done < "${MOTION_LIST}"
if (( missing > 0 )); then
  echo "refusing to start: ${missing} motion caches are missing" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5

set +e
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" "${ROOT}/scripts/collect_rollouts.py" \
  --motion-file "${MOTION_LIST}" \
  --output-dir "${OUTPUT}" \
  --chunk-size 32 \
  --resume \
  2>&1 | tee -a "${LOG_DIR}/collect_all_available.log"
collect_rc=${PIPESTATUS[0]}
set -e

if [[ ! -f "${OUTPUT}/manifest.json" ]]; then
  echo "collection did not produce ${OUTPUT}/manifest.json" >&2
  exit "${collect_rc}"
fi

"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" "${ROOT}/scripts/validate_rollouts.py" \
  --manifest "${OUTPUT}/manifest.json" \
  --output "${OUTPUT}/validation_manifest.json" \
  --resume \
  2>&1 | tee -a "${LOG_DIR}/validate_all_available.log"

"${PYTHON}" "${ROOT}/scripts/all_motion_replay_status.py"
exit 0
