#!/usr/bin/env bash
# Auto-resume teacher rollout collection until TARGET valid npz files exist.
# Survives worker OOM, BrokenProcessPool, and single-run exits: always --skip_existing.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

OUTPUT_DIR="${OUTPUT_DIR:-data/teacher_rollouts/KIT_KINESIS_TRAINING_MOTIONS}"
TARGET="${TARGET:-972}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SLEEP_SEC="${SLEEP_SEC:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"  # 0 = unlimited
LOG="${LOG:-outputs/prosthesis_distill/teacher_collect_until_done.log}"
mkdir -p "$(dirname "$LOG")"

count_valid_npz() {
  uv run python - <<'PY'
from pathlib import Path
from musclemimic.distill.mapping import rollout_npz_is_valid
out = Path(__import__("os").environ["OUTPUT_DIR"])
print(sum(1 for p in out.glob("*.npz") if rollout_npz_is_valid(p)))
PY
}

round=0
while true; do
  round=$((round + 1))
  done="$(OUTPUT_DIR="$OUTPUT_DIR" count_valid_npz)"
  remaining=$((TARGET - done))
  echo "[round ${round}] progress ${done}/${TARGET} (remaining ${remaining})" | tee -a "$LOG"
  if [[ "$done" -ge "$TARGET" ]]; then
    echo "Done: ${done}/${TARGET} valid npz." | tee -a "$LOG"
    exit 0
  fi
  if [[ "$MAX_ROUNDS" -gt 0 && "$round" -gt "$MAX_ROUNDS" ]]; then
    echo "Stopped after ${MAX_ROUNDS} rounds at ${done}/${TARGET}." | tee -a "$LOG"
    exit 1
  fi

  set +e
  NUM_WORKERS="$NUM_WORKERS" OUTPUT_DIR="$OUTPUT_DIR" \
    bash scripts/prosthesis_collect_teacher_rollouts.sh 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "WARNING: collect round exited with code ${rc}; will resume after ${SLEEP_SEC}s" | tee -a "$LOG"
  fi
  sleep "$SLEEP_SEC"
done
