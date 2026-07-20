#!/usr/bin/env bash
# Auto-resume masked-obs full-muscle rollout collection until TARGET valid npz exist.
# Survives single-motion failures (--continue_on_error) and round exits: always --skip_existing.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

OUTPUT_DIR="${OUTPUT_DIR:-data/full_muscle_masked_obs_rollouts/KIT_KINESIS_TRAINING_MOTIONS}"
TARGET="${TARGET:-972}"
SLEEP_SEC="${SLEEP_SEC:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
LOG="${LOG:-outputs/full_muscle_masked_obs_distill/collect_until_done.log}"
mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"

count_valid_npz() {
  uv run python - <<'PY'
from pathlib import Path
import numpy as np
out = Path(__import__("os").environ["OUTPUT_DIR"])
n = 0
for p in out.glob("*.npz"):
    if p.stat().st_size == 0:
        continue
    try:
        with np.load(p, allow_pickle=True) as d:
            if "obs_student_masked" not in d.files or "target_full_muscle_action" not in d.files:
                continue
            if int(d["obs_student_masked"].shape[0]) <= 0:
                continue
        n += 1
    except Exception:
        pass
print(n)
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
  uv run python scripts/collect_masked_obs_full_muscle_rollouts.py \
    --dataset_group KIT_KINESIS_TRAINING_MOTIONS \
    --output_dir "$OUTPUT_DIR" \
    --n_steps_per_motion auto \
    --skip_existing \
    --continue_on_error 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "WARNING: collect round exited with code ${rc}; will resume after ${SLEEP_SEC}s" | tee -a "$LOG"
  fi
  sleep "$SLEEP_SEC"
done
