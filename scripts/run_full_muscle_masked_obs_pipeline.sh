#!/usr/bin/env bash
# 1) Collect masked-obs + full-muscle-action rollouts (skip failures, resume)
# 2) Train student: masked obs -> 354 muscle actions
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

OUTPUT_DIR="${OUTPUT_DIR:-data/full_muscle_masked_obs_rollouts/KIT_KINESIS_TRAINING_MOTIONS}"
TARGET="${TARGET:-972}"
DISTILL_OUT="${DISTILL_OUT:-outputs/full_muscle_masked_obs_distill}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LOG="${LOG:-outputs/full_muscle_masked_obs_distill/pipeline.log}"
mkdir -p "$(dirname "$LOG")"

echo "=== Phase 1: collect until ${TARGET} ===" | tee -a "$LOG"
TARGET="$TARGET" OUTPUT_DIR="$OUTPUT_DIR" LOG="$LOG" \
  bash scripts/collect_masked_obs_full_muscle_until_done.sh 2>&1 | tee -a "$LOG"
collect_rc=$?
if [[ "$collect_rc" -ne 0 ]]; then
  echo "Collect phase exited ${collect_rc}; check log. Aborting distill." | tee -a "$LOG"
  exit "$collect_rc"
fi

echo "=== Phase 2: supervised distillation ===" | tee -a "$LOG"
uv run python scripts/train_masked_obs_full_muscle_distillation.py \
  --dataset_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$DISTILL_OUT" 2>&1 | tee -a "$LOG"
echo "Done. Checkpoint: ${DISTILL_OUT}/latest/checkpoints/checkpoint_distilled" | tee -a "$LOG"
