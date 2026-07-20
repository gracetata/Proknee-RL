#!/usr/bin/env bash
# Auto-resume DAgger collection in fresh subprocess batches until TARGET valid npz exist.
# This avoids long-lived JAX/XLA compilation caches exhausting GPU memory.
set -uo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

OUTPUT_DIR="${OUTPUT_DIR:-data/full_muscle_masked_obs_dagger/round1}"
TARGET="${TARGET:-972}"
BATCH_SIZE="${BATCH_SIZE:-25}"
SLEEP_SEC="${SLEEP_SEC:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
MOTION_GROUP="${MOTION_GROUP:-KIT_KINESIS_TRAINING_MOTIONS}"
TEACHER_ACTION_PROB="${TEACHER_ACTION_PROB:-0.1}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-outputs/full_muscle_masked_obs_distill/latest/checkpoints/checkpoint_distilled}"
LOG="${LOG:-outputs/full_muscle_masked_obs_dagger/round1_until_done.log}"

mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"

count_valid_npz() {
  OUTPUT_DIR="$OUTPUT_DIR" uv run python - <<'PY'
from pathlib import Path
import os
import numpy as np

out = Path(os.environ["OUTPUT_DIR"])
n = 0
for path in out.glob("*.npz"):
    if path.stat().st_size == 0:
        continue
    try:
        with np.load(path, allow_pickle=True) as data:
            if "obs_student_masked" not in data.files or "target_full_muscle_action" not in data.files:
                continue
            if int(data["obs_student_masked"].shape[0]) <= 0:
                continue
        n += 1
    except Exception:
        pass
print(n)
PY
}

select_missing_batch() {
  OUTPUT_DIR="$OUTPUT_DIR" BATCH_SIZE="$BATCH_SIZE" MOTION_GROUP="$MOTION_GROUP" uv run python - <<'PY'
from pathlib import Path
import os
import numpy as np

from musclemimic.distill.config import motion_list_from_group
from musclemimic.distill.rollout import safe_motion_filename

out = Path(os.environ["OUTPUT_DIR"])
batch_size = int(os.environ["BATCH_SIZE"])
motions = motion_list_from_group(os.environ["MOTION_GROUP"])

def valid_npz(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return (
                "obs_student_masked" in data.files
                and "target_full_muscle_action" in data.files
                and int(data["obs_student_masked"].shape[0]) > 0
            )
    except Exception:
        return False

missing = [motion for motion in motions if not valid_npz(out / safe_motion_filename(motion))]
for motion in missing[:batch_size]:
    print(motion)
PY
}

round=0
while true; do
  round=$((round + 1))
  done_count="$(count_valid_npz)"
  remaining=$((TARGET - done_count))
  echo "[round ${round}] progress ${done_count}/${TARGET} (remaining ${remaining})" | tee -a "$LOG"

  if [[ "$done_count" -ge "$TARGET" ]]; then
    echo "Done: ${done_count}/${TARGET} valid npz." | tee -a "$LOG"
    exit 0
  fi
  if [[ "$MAX_ROUNDS" -gt 0 && "$round" -gt "$MAX_ROUNDS" ]]; then
    echo "Stopped after ${MAX_ROUNDS} rounds at ${done_count}/${TARGET}." | tee -a "$LOG"
    exit 1
  fi

  mapfile -t motions < <(select_missing_batch)
  if [[ "${#motions[@]}" -eq 0 ]]; then
    echo "No missing motions selected, but progress is ${done_count}/${TARGET}." | tee -a "$LOG"
    exit 1
  fi

  echo "[round ${round}] running ${#motions[@]} motions; first=${motions[0]}" | tee -a "$LOG"
  set +e
  uv run python scripts/collect_masked_obs_full_muscle_dagger_rollouts.py \
    --student_checkpoint "$STUDENT_CHECKPOINT" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --motion_path "${motions[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --teacher_action_prob "$TEACHER_ACTION_PROB" \
    --skip_existing \
    --continue_on_error 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  after_count="$(count_valid_npz)"
  echo "[round ${round}] exited rc=${rc}; progress ${after_count}/${TARGET}" | tee -a "$LOG"
  if [[ "$rc" -ne 0 ]]; then
    echo "WARNING: batch exited with code ${rc}; continuing after ${SLEEP_SEC}s." | tee -a "$LOG"
  fi
  sleep "$SLEEP_SEC"
done
