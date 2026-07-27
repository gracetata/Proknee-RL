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

expected=0
missing=0
while IFS= read -r motion; do
  [[ -z "${motion}" || "${motion}" == \#* ]] && continue
  expected=$((expected + 1))
  if [[ ! -f "${CACHE_ROOT}/${motion}.npz" ]]; then
    echo "missing GMR cache: ${CACHE_ROOT}/${motion}.npz" >&2
    missing=$((missing + 1))
  fi
done < "${MOTION_LIST}"
if (( expected != 1089 )); then
  echo "refusing to start: expected 1089 motions, found ${expected}" >&2
  exit 1
fi
if (( missing > 0 )); then
  echo "refusing to start: ${missing} motion caches are missing" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/scripts/local_4090_guard.py"

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

manifest_value() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
if not path.is_file():
    print(-1)
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key == "session_setup_failures":
        print(
            sum(
                str(row.get("error", "")).startswith("session setup failed:")
                for row in payload.get("rollouts", [])
            )
        )
    else:
        print(payload.get(key, -1))
PY
}

collection_attempt=0
previous_pending=1089
collection_chunk_size=32
while true; do
  collection_attempt=$((collection_attempt + 1))
  set +e
  "${PYTHON}" "${ROOT}/scripts/collect_rollouts.py" \
    --motion-file "${MOTION_LIST}" \
    --output-dir "${OUTPUT}" \
    --chunk-size "${collection_chunk_size}" \
    --resume \
    2>&1 | tee -a "${LOG_DIR}/collect_all_available_local.log"
  collect_rc=${PIPESTATUS[0]}
  set -e

  pending="$(manifest_value "${OUTPUT}/manifest.json" pending)"
  if (( pending == 0 )); then
    setup_failures="$(manifest_value "${OUTPUT}/manifest.json" session_setup_failures)"
    if (( setup_failures > 0 && collection_chunk_size > 1 )); then
      echo "${setup_failures} motions failed batch session setup; retrying them individually"
      collection_chunk_size=1
      previous_pending="${setup_failures}"
      continue
    fi
    echo "collection complete; exit code ${collect_rc} (1 is expected when motions are rejected)"
    break
  fi
  if (( pending < 0 || pending >= previous_pending || collection_attempt >= 10 )); then
    echo "collection stopped without recoverable progress: rc=${collect_rc}, pending=${pending}" >&2
    (( collect_rc == 0 )) && exit 1
    exit "${collect_rc}"
  fi
  echo "collection process interrupted after progress: rc=${collect_rc}, pending=${pending}; resuming"
  previous_pending="${pending}"
  "${PYTHON}" "${ROOT}/scripts/local_4090_guard.py"
done

validation_attempt=0
previous_pending="$(manifest_value "${OUTPUT}/validation_manifest.json" pending)"
if (( previous_pending < 0 )); then
  previous_pending="$(manifest_value "${OUTPUT}/manifest.json" completed)"
fi
validation_chunk_size=16
while true; do
  validation_attempt=$((validation_attempt + 1))
  set +e
  "${PYTHON}" "${ROOT}/scripts/validate_rollouts.py" \
    --manifest "${OUTPUT}/manifest.json" \
    --output "${OUTPUT}/validation_manifest.json" \
    --chunk-size "${validation_chunk_size}" \
    --resume \
    2>&1 | tee -a "${LOG_DIR}/validate_all_available_local.log"
  validation_rc=${PIPESTATUS[0]}
  set -e

  pending="$(manifest_value "${OUTPUT}/validation_manifest.json" pending)"
  if (( pending == 0 )); then
    "${PYTHON}" "${ROOT}/scripts/all_motion_replay_status.py"
    exit "${validation_rc}"
  fi
  if (( pending >= 0 && pending >= previous_pending && validation_chunk_size > 1 )); then
    echo "validation batch made no progress; retrying pending datasets individually"
    validation_chunk_size=1
    previous_pending="${pending}"
    continue
  fi
  if (( pending < 0 || pending >= previous_pending || validation_attempt >= 10 )); then
    echo "validation stopped without recoverable progress: rc=${validation_rc}, pending=${pending}" >&2
    (( validation_rc == 0 )) && exit 1
    exit "${validation_rc}"
  fi
  echo "validation process interrupted after progress: rc=${validation_rc}, pending=${pending}; resuming"
  previous_pending="${pending}"
  "${PYTHON}" "${ROOT}/scripts/local_4090_guard.py"
done
