#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
MANIFEST="${ROOT}/configs/flat_walk_compact_manifest.json"
DATA_DIR="${ROOT}/data/flat_walk_compact_v1"
MODEL_DIR="${ROOT}/data/replay_model"
RUNTIME="${ROOT}/runtime"
FILE_LIST="${RUNTIME}/flat_walk_compact_files.txt"
PENDING_LIST="${RUNTIME}/flat_walk_compact_pending.txt"
REMOTE_REPORT="${RUNTIME}/flat_walk_compact_remote_report.json"
REMOTE="root@39.105.12.60"
PORT="6029"
REMOTE_ROOT="/workspace/Proknee-RL-muscle"
REMOTE_DATA="${REMOTE_ROOT}/torque_replay_training/data/flat_walk_compact_v1"

mkdir -p "${RUNTIME}"
"${PYTHON}" - "${MANIFEST}" "${FILE_LIST}" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
names = sorted(row["dataset_basename"] for row in manifest["datasets"])
Path(sys.argv[2]).write_text("\n".join(names) + "\n", encoding="utf-8")
print(f"selected_files={len(names)}")
PY

"${PYTHON}" "${ROOT}/scripts/verify_compact_flat_walk_data.py" \
  --manifest "${MANIFEST}" --data-dir "${DATA_DIR}"
ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" "mkdir -p '${REMOTE_DATA}'"
set +e
ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
  "cd '${REMOTE_ROOT}' && .venv/bin/python \
    torque_replay_training/scripts/verify_compact_flat_walk_data.py \
    --manifest 'torque_replay_training/configs/flat_walk_compact_manifest.json' \
    --data-dir '${REMOTE_DATA}' --skip-schema" >"${REMOTE_REPORT}"
set -e
"${PYTHON}" - "${REMOTE_REPORT}" "${PENDING_LIST}" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pending = set(report["missing"]) | set(report["wrong_size"]) | set(report["wrong_sha256"])
pending |= {row["dataset"] for row in report["invalid_schema"]}
Path(sys.argv[2]).write_text(
    "\n".join(sorted(pending)) + ("\n" if pending else ""),
    encoding="utf-8",
)
print(f"pending_files={len(pending)}")
PY

if [[ -s "${PENDING_LIST}" ]]; then
  SHARD_PREFIX="${RUNTIME}/flat_walk_compact_shard_"
  split -n l/4 -d -a 1 "${PENDING_LIST}" "${SHARD_PREFIX}"
  PIDS=()
  for shard in "${SHARD_PREFIX}"*; do
    [[ -s "${shard}" ]] || continue
    (
      tar -C "${DATA_DIR}" -cf - -T "${shard}" |
        ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
          "tar -C '${REMOTE_DATA}' -xf -"
    ) &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
fi

if [[ ! -f "${MODEL_DIR}/musclemimic_replay.mjb" || \
      ! -f "${MODEL_DIR}/musclemimic_replay.mjb.json" || \
      ! -f "${MODEL_DIR}/musclemimic_replay.xml" ]]; then
  echo "missing local replay MJB, metadata, or portable XML in ${MODEL_DIR}" >&2
  exit 1
fi
tar -C "${MODEL_DIR}" -cf - \
  musclemimic_replay.mjb \
  musclemimic_replay.mjb.json \
  musclemimic_replay.xml |
  ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
    "mkdir -p '${REMOTE_ROOT}/torque_replay_training/data/replay_model' && \
     tar -C '${REMOTE_ROOT}/torque_replay_training/data/replay_model' -xf -"

ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
  "cd '${REMOTE_ROOT}' && .venv/bin/python \
    torque_replay_training/scripts/verify_compact_flat_walk_data.py \
    --manifest 'torque_replay_training/configs/flat_walk_compact_manifest.json' \
    --data-dir '${REMOTE_DATA}'"
