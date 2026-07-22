#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
OUTPUT="${ROOT}/data/fullbody_v1"
MANIFEST="${OUTPUT}/manifest.json"
CACHE_ROOT="${HOME}/.musclemimic/caches/AMASS/MyoFullBody/gmr"
MOTIONS=(
  KIT/314/walking_medium09_poses
  KIT/425/walking_slow07_poses
  KIT/167/turn_right01_poses
  KIT/167/turn_left01_poses
)

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing Python environment: ${PYTHON}" >&2
  exit 1
fi

for motion in "${MOTIONS[@]}"; do
  if [[ ! -f "${CACHE_ROOT}/${motion}.npz" ]]; then
    echo "missing qualified GMR cache: ${CACHE_ROOT}/${motion}.npz" >&2
    exit 1
  fi
done

if [[ -f "${MANIFEST}" ]] && "${PYTHON}" - "${MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
valid = manifest.get("production") is True and manifest.get("all_passed") is True
valid = valid and len(manifest.get("rollouts", [])) == 4
raise SystemExit(0 if valid else 1)
PY
then
  echo "qualified production dataset already exists: ${MANIFEST}"
  exit 0
fi

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite incomplete production data: ${OUTPUT}" >&2
  exit 1
fi

mkdir -p "${ROOT}/data"
"${PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" "${ROOT}/scripts/collect_rollouts.py" \
  --output-dir "${OUTPUT}" \
  --motion "${MOTIONS[0]}" \
  --motion "${MOTIONS[1]}" \
  --motion "${MOTIONS[2]}" \
  --motion "${MOTIONS[3]}"

for dataset in "${OUTPUT}"/*.npz; do
  "${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
    "${ROOT}/scripts/validate_replay.py" --dataset "${dataset}"
done
