#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
GUARD_PYTHON="${REPO_ROOT}/.venv/bin/python"
PYTHON="${REPO_ROOT}/.venv-hora/bin/python"
CONFIG="${ROOT}/configs/flat_walk_hora.yaml"
SPLIT="${ROOT}/configs/flat_walk_split.json"
MANIFEST="${ROOT}/configs/flat_walk_compact_manifest.json"
DATA_DIR="${ROOT}/data/flat_walk_compact_v1"
MODEL="${ROOT}/data/replay_model/musclemimic_replay.mjb"
OUTPUT_ROOT="${ROOT}/outputs/a100_flat_walk_v1"
OUTPUT="${OUTPUT_ROOT}/seed_0"
RUNTIME="${ROOT}/runtime"
LOCK="${RUNTIME}/a100_flat_walk_v1.lock"
RUNNING="${RUNTIME}/a100_flat_walk_v1.running"
COMPLETE="${RUNTIME}/a100_flat_walk_v1.complete"
FAILED="${RUNTIME}/a100_flat_walk_v1.failed"

mkdir -p "${RUNTIME}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another flat-walk launcher owns ${LOCK}" >&2
  exit 4
fi
if [[ -f "${COMPLETE}" ]]; then
  echo "flat-walk training already completed: ${COMPLETE}"
  exit 0
fi
if [[ -f "${FAILED}" ]]; then
  echo "previous flat-walk run failed; inspect ${FAILED}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite existing output: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "missing HORA environment: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "missing actuator-disabled replay model: ${MODEL}" >&2
  exit 1
fi
if [[ "$(git -C "${REPO_ROOT}" branch --show-current)" != "muscle" ]]; then
  echo "A100 training must run from the muscle branch" >&2
  exit 1
fi

"${GUARD_PYTHON}" "${ROOT}/scripts/verify_compact_flat_walk_data.py" \
  --manifest "${MANIFEST}" --data-dir "${DATA_DIR}"
mapfile -t TRAIN_DATASETS < <(
  "${PYTHON}" - "${SPLIT}" "${MANIFEST}" "${DATA_DIR}" <<'PY'
import json
from pathlib import Path
import sys
split = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {row["dataset_basename"]: int(row["compact_bytes"]) for row in manifest["datasets"]}
root = Path(sys.argv[3])
for row in split["train"]:
    path = root / row["dataset_basename"]
    if not path.is_file() or path.stat().st_size != expected[path.name]:
        raise SystemExit(f"missing or invalid training dataset: {path}")
    print(path)
PY
)
mapfile -t VALIDATION_DATASETS < <(
  "${PYTHON}" - "${SPLIT}" "${MANIFEST}" "${DATA_DIR}" <<'PY'
import json
from pathlib import Path
import sys
split = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {row["dataset_basename"]: int(row["compact_bytes"]) for row in manifest["datasets"]}
root = Path(sys.argv[3])
for row in split["validation"]:
    path = root / row["dataset_basename"]
    if not path.is_file() or path.stat().st_size != expected[path.name]:
        raise SystemExit(f"missing or invalid validation dataset: {path}")
    print(path)
PY
)
if [[ ${#TRAIN_DATASETS[@]} -ne 229 || ${#VALIDATION_DATASETS[@]} -ne 52 ]]; then
  echo "expected 229 train and 52 validation datasets" >&2
  exit 1
fi

"${GUARD_PYTHON}" "${ROOT}/scripts/a100_gpu_guard.py" --gpus 5
mkdir -p "${OUTPUT_ROOT}"
printf 'started_at=%s\ngit_head=%s\ngpu=5\ntrain_datasets=%s\nvalidation_datasets=%s\npid=%s\n' \
  "$(date -Is)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
  "${#TRAIN_DATASETS[@]}" "${#VALIDATION_DATASETS[@]}" "$$" >"${RUNNING}"

on_exit() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${status}" >"${FAILED}"
  fi
  unlink "${RUNNING}" 2>/dev/null || true
}
trap on_exit EXIT

"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/evaluate_hora_policy.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --group validation \
  --episodes 100 >"${OUTPUT_ROOT}/baseline_validation.json" \
  2>"${OUTPUT_ROOT}/baseline_validation.log"
"${PYTHON}" - "${OUTPUT_ROOT}/baseline_validation.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required_zero = (
    "fall_rate",
    "mean_healthy_pos_rms",
    "mean_healthy_vel_rms",
    "mean_prosthesis_pos_rms",
    "mean_prosthesis_vel_rms",
)
failed = {name: report.get(name) for name in required_zero if report.get(name) != 0.0}
if report.get("episodes") != 100 or failed:
    raise SystemExit(f"healthy baseline gate failed: episodes={report.get('episodes')} {failed}")
print("healthy baseline gate passed: 100 episodes, zero falls and zero replay error")
PY

"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/train_hora_policy.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --manifest "${MANIFEST}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --output "${OUTPUT}" \
  --seed 0 >"${OUTPUT_ROOT}/train.log" 2>&1

LATEST_POLICY="${OUTPUT}/stage1_nn/last.pth"
if [[ ! -f "${LATEST_POLICY}" ]]; then
  echo "training completed without a policy checkpoint" >&2
  exit 1
fi
"${ROOT}/scripts/a100_exec_gpu.sh" 5 "${PYTHON}" \
  "${ROOT}/scripts/evaluate_hora_policy.py" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --data-dir "${DATA_DIR}" \
  --model "${MODEL}" \
  --group validation \
  --policy "${LATEST_POLICY}" \
  --episodes 200 >"${OUTPUT_ROOT}/policy_validation.json" \
  2>"${OUTPUT_ROOT}/policy_validation.log"

printf 'completed_at=%s\ngit_head=%s\ngpu=5\ncheckpoint=%s\n' \
  "$(date -Is)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
  "${LATEST_POLICY}" >"${COMPLETE}"
unlink "${RUNNING}" 2>/dev/null || true
trap - EXIT
echo "flat-walk training completed: ${LATEST_POLICY}"
