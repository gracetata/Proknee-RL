#!/usr/bin/env bash
# Collect teacher rollouts for prosthesis distillation (CPU MuJoCo sim, optional parallel workers).
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

OUTPUT_DIR="${OUTPUT_DIR:-data/teacher_rollouts/KIT_KINESIS_TRAINING_MOTIONS}"
EXTRA_ARGS=("$@")

if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
  FAIL_LOG="${OUTPUT_DIR}/failed_motions.jsonl"
  export FAIL_LOG
  if [[ ! -f "${FAIL_LOG}" ]]; then
    echo "No failed_motions.jsonl at ${FAIL_LOG}; nothing to retry." >&2
    exit 1
  fi
  mapfile -t FAILED_MOTIONS < <(python3 - <<'PY'
import json
from pathlib import Path
import os
path = Path(os.environ["FAIL_LOG"])
seen = set()
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    motion = json.loads(line)["motion_path"]
    if motion not in seen:
        seen.add(motion)
        print(motion)
PY
)
  if [[ ${#FAILED_MOTIONS[@]} -eq 0 ]]; then
    echo "failed_motions.jsonl is empty; nothing to retry." >&2
    exit 1
  fi
  : > "${FAIL_LOG}"
  EXTRA_ARGS=(--motion_path "${FAILED_MOTIONS[@]}" --skip_existing "${EXTRA_ARGS[@]}")
  echo "Retrying ${#FAILED_MOTIONS[@]} previously failed motions..."
fi

uv run python scripts/collect_teacher_rollouts.py \
  --teacher_checkpoint /home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2 \
  --dataset_group KIT_KINESIS_TRAINING_MOTIONS \
  --output_dir "${OUTPUT_DIR}" \
  --skip_existing \
  --continue_on_error \
  --num_workers "${NUM_WORKERS:-2}" \
  "${EXTRA_ARGS[@]}"
