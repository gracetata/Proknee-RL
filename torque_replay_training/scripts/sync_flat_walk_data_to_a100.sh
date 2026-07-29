#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SPLIT="${ROOT}/configs/flat_walk_split.json"
DATA_DIR="${ROOT}/data/fullbody_all_v3"
RUNTIME="${ROOT}/runtime"
FILE_LIST="${RUNTIME}/flat_walk_files.txt"
PENDING_LIST="${RUNTIME}/flat_walk_pending_files.txt"
REMOTE_REPORT="${RUNTIME}/flat_walk_remote_report.json"
REMOTE="root@39.105.12.60"
PORT="6029"
REMOTE_ROOT="/workspace/Proknee-RL-muscle"
REMOTE_DATA="${REMOTE_ROOT}/torque_replay_training/data/fullbody_all_v3"
MODEL_DIR="${ROOT}/data/replay_model"
MODEL_BASENAME="musclemimic_replay.mjb"
MODEL_METADATA="musclemimic_replay.json"

mkdir -p "${RUNTIME}"
"${PYTHON}" - "${SPLIT}" "${FILE_LIST}" <<'PY'
import json
from pathlib import Path
import sys

split = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
names = sorted(
    {row["dataset_basename"] for group in ("train", "validation") for row in split[group]}
)
Path(sys.argv[2]).write_text("\n".join(names) + "\n", encoding="utf-8")
print(f"selected_files={len(names)}")
PY

while IFS= read -r name; do
  if [[ ! -f "${DATA_DIR}/${name}" ]]; then
    echo "missing local flat-walk dataset: ${DATA_DIR}/${name}" >&2
    exit 1
  fi
done <"${FILE_LIST}"

ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" "mkdir -p '${REMOTE_DATA}'"
set +e
ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
  "cd '${REMOTE_ROOT}' && .venv/bin/python \
    torque_replay_training/scripts/verify_flat_walk_data.py \
    --split '${SPLIT#${REPO_ROOT}/}' \
    --data-dir '${REMOTE_DATA}'" >"${REMOTE_REPORT}"
set -e
"${PYTHON}" - "${REMOTE_REPORT}" "${PENDING_LIST}" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pending = sorted(set(report["missing"]) | set(report["wrong_size"]))
Path(sys.argv[2]).write_text(
    "\n".join(pending) + ("\n" if pending else ""),
    encoding="utf-8",
)
print(f"pending_files={len(pending)}")
PY

if [[ -s "${PENDING_LIST}" ]]; then
  tar -C "${DATA_DIR}" -cf - -T "${PENDING_LIST}" |
    ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" "tar -C '${REMOTE_DATA}' -xf -"
fi

ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
  "cd '${REMOTE_ROOT}' && .venv/bin/python \
    torque_replay_training/scripts/verify_flat_walk_data.py \
    --split '${SPLIT#${REPO_ROOT}/}' \
    --data-dir '${REMOTE_DATA}'"

if [[ ! -f "${MODEL_DIR}/${MODEL_BASENAME}" || ! -f "${MODEL_DIR}/${MODEL_METADATA}" ]]; then
  echo "missing local replay MJB or metadata in ${MODEL_DIR}" >&2
  exit 1
fi
tar -C "${MODEL_DIR}" -cf - "${MODEL_BASENAME}" "${MODEL_METADATA}" |
  ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
    "mkdir -p '${REMOTE_ROOT}/torque_replay_training/data/replay_model' && \
     tar -C '${REMOTE_ROOT}/torque_replay_training/data/replay_model' -xf -"
