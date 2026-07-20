#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
if [[ -x "${REPO_ROOT}/musclemimic/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/musclemimic/.venv/bin/python"
else
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi
DATA="${ROOT}/data/smoke/walking_medium09_8steps.npz"
OUTPUT="${ROOT}/outputs/smoke/ppo"

mkdir -p "${ROOT}/data/smoke" "${ROOT}/outputs/smoke"
"${PYTHON}" "${ROOT}/scripts/export_fullbody_rollout.py" \
  --motion KIT/314/walking_medium09_poses \
  --output "${DATA}" --steps 8 --allow-incomplete
"${PYTHON}" "${ROOT}/scripts/validate_replay.py" --dataset "${DATA}" --steps 8
rm -rf "${OUTPUT}"
"${PYTHON}" "${ROOT}/scripts/train_policy.py" \
  --config "${ROOT}/configs/smoke.yaml" --dataset "${DATA}" --output "${OUTPUT}"
"${PYTHON}" "${ROOT}/scripts/evaluate_policy.py" \
  --config "${ROOT}/configs/smoke.yaml" --dataset "${DATA}" \
  --policy "${OUTPUT}/policy_000000032.msgpack" --episodes 1
