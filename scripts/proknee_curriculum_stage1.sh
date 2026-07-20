#!/usr/bin/env bash
# Stage1 prosthesis-replacement curriculum: warm-start from r002 (or bootstrap), then
# raise DAgger ratio and lower prosthesis_muscle_scale toward motor-only.
#
# Environment:
#   PROKNEE_CURRICULUM_STEPS  (default 200000) — full training steps per phase
#   PROKNEE_CURRICULUM_BATCH  (default 256)
#   PROKNEE_R002              override path to warm-start checkpoint
#
# Usage:
#   bash musclemimic/scripts/proknee_curriculum_stage1.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ="$ROOT/musclemimic"
MM_CKPT="${MM_CKPT:-$ROOT/data/checkpoints/mm-10m-2}"
STEPS="${PROKNEE_CURRICULUM_STEPS:-200000}"
BS="${PROKNEE_CURRICULUM_BATCH:-256}"
R002="${PROKNEE_R002:-$ROOT/outputs/proknee_stage1_qpos_dagger_r002/stage1_best.pt}"

# Short smoke runs need frequent eval or stage1_best.pt is never written.
if (( STEPS < 10000 )); then
  EVAL_IV=$(( STEPS / 2 > 128 ? STEPS / 2 : 128 ))
  EVAL_ST="${PROKNEE_CURRICULUM_EVAL_STEPS:-256}"
  LOG_IV=$(( STEPS / 8 > 64 ? STEPS / 8 : 64 ))
  SAVE_IV="$STEPS"
else
  EVAL_IV="${PROKNEE_CURRICULUM_EVAL_INTERVAL:-10000}"
  EVAL_ST="${PROKNEE_CURRICULUM_EVAL_STEPS:-2048}"
  LOG_IV=500
  SAVE_IV="${PROKNEE_CURRICULUM_SAVE_INTERVAL:-50000}"
fi

TRAIN=(
  env PYTHONUNBUFFERED=1 uv run --project "$PROJ" python -u "$ROOT/musclemimic/scripts/train_proknee_stage1.py"
  --checkpoint "$MM_CKPT"
  --dataset-group KIT_KINESIS_TRAINING_MOTIONS
  --val-dataset-group KIT_KINESIS_TESTING_MOTIONS
  --target-mode qpos
  --steps "$STEPS"
  --batch-size "$BS"
  --history-len 30
  --lr 2e-4
  --dagger-pd-override
  --deterministic-oracle
  --qvel-loss-weight 1e-4
  --smoothness-loss-weight 0.01
  --control-dt 0.01
  --pd-torque-limit 120
  --joint-kp "300,200,200,120"
  --joint-kd "30,20,20,12"
  --log-interval "$LOG_IV"
  --eval-interval "$EVAL_IV"
  --eval-steps "$EVAL_ST"
  --save-interval "$SAVE_IV"
)

resume_from="$R002"
if [[ ! -f "$resume_from" ]]; then
  echo "[proknee_curriculum] Missing warm-start: $resume_from"
  echo "  Train a qpos+DAgger Stage1 checkpoint first, or set PROKNEE_R002 to an existing stage1_best.pt"
  exit 1
fi

run_phase() {
  local name="$1"
  shift
  echo "========== Phase: $name =========="
  "${TRAIN[@]}" --resume "$resume_from" "$@"
  resume_from="$ROOT/outputs/$name/stage1_best.pt"
}

# 1) Wider DAgger @ full boundary muscle support
run_phase proknee_stage1_qpos_dagger_r005_muscle1 \
  --teacher-exec-ratio 0.05 \
  --prosthesis-muscle-scale 1.0 \
  --output-dir "$ROOT/outputs/proknee_stage1_qpos_dagger_r005_muscle1"

# 2) Same DAgger, boundary muscles at 50%
run_phase proknee_stage1_qpos_dagger_r005_muscle05 \
  --teacher-exec-ratio 0.05 \
  --prosthesis-muscle-scale 0.5 \
  --output-dir "$ROOT/outputs/proknee_stage1_qpos_dagger_r005_muscle05"

# 3) Higher DAgger, 25% muscles
run_phase proknee_stage1_qpos_dagger_r010_muscle025 \
  --teacher-exec-ratio 0.10 \
  --prosthesis-muscle-scale 0.25 \
  --output-dir "$ROOT/outputs/proknee_stage1_qpos_dagger_r010_muscle025"

# 4) Motor-only boundary (curriculum endpoint)
run_phase proknee_stage1_qpos_dagger_r010_muscle0 \
  --teacher-exec-ratio 0.10 \
  --prosthesis-muscle-scale 0.0 \
  --output-dir "$ROOT/outputs/proknee_stage1_qpos_dagger_r010_muscle0"

echo "[proknee_curriculum] Final best checkpoint: $resume_from"
