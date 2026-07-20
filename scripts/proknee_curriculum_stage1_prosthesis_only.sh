#!/usr/bin/env bash
# Stage1 prosthesis-only stabilization curriculum.
#
# Starts from a qpos+DAgger checkpoint and keeps prosthesis-boundary muscles at
# zero active control while increasing closed-loop Teacher execution.
#
# Environment:
#   PROKNEE_PO_STEPS        default 100000 per phase
#   PROKNEE_PO_BATCH        default 256
#   PROKNEE_PO_START       warm-start checkpoint
#   PROKNEE_PO_EVAL_STEPS  default 1024 for physical eval
#   PROKNEE_PO_RUN_PREFIX  output prefix, default proknee_stage1_prosthesis_only_forceff
#
# Usage:
#   bash musclemimic/scripts/proknee_curriculum_stage1_prosthesis_only.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ="$ROOT/musclemimic"
MM_CKPT="${MM_CKPT:-$ROOT/data/checkpoints/mm-10m-2}"
STEPS="${PROKNEE_PO_STEPS:-100000}"
BS="${PROKNEE_PO_BATCH:-256}"
START="${PROKNEE_PO_START:-$ROOT/outputs/proknee_stage1_qpos_dagger_r010_muscle0/stage1_best.pt}"
RUN_PREFIX="${PROKNEE_PO_RUN_PREFIX:-proknee_stage1_prosthesis_only_torquehead}"

if (( STEPS < 10000 )); then
  EVAL_IV=$(( STEPS / 2 > 128 ? STEPS / 2 : 128 ))
  LOG_IV=$(( STEPS / 8 > 64 ? STEPS / 8 : 64 ))
  SAVE_IV="$STEPS"
  EVAL_ST="${PROKNEE_PO_EVAL_STEPS:-256}"
else
  EVAL_IV="${PROKNEE_PO_EVAL_INTERVAL:-5000}"
  LOG_IV="${PROKNEE_PO_LOG_INTERVAL:-500}"
  SAVE_IV="${PROKNEE_PO_SAVE_INTERVAL:-25000}"
  EVAL_ST="${PROKNEE_PO_EVAL_STEPS:-1024}"
fi

if [[ ! -f "$START" ]]; then
  echo "[prosthesis_only] Missing warm-start checkpoint: $START"
  exit 1
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
  --lr "${PROKNEE_PO_LR:-1e-4}"
  --dagger-pd-override
  --deterministic-oracle
  --prosthesis-muscle-scale 0.0
  --joint-kp "${PROKNEE_PO_JOINT_KP:-120,45,35,12}"
  --joint-kd "${PROKNEE_PO_JOINT_KD:-35,38,34,18}"
  --pd-torque-limit "${PROKNEE_PO_TORQUE_LIMIT:-55}"
  --pd-torque-limit-list "${PROKNEE_PO_TORQUE_LIMIT_LIST:-110,65,45,22}"
  --pd-torque-slew-limit-list "${PROKNEE_PO_TORQUE_SLEW_LIMIT_LIST:-10,5,3,1.5}"
  --oracle-torque-ff-scale "${PROKNEE_PO_ORACLE_TORQUE_FF_SCALE:-1.0}"
  --oracle-torque-ff-limit-list "${PROKNEE_PO_ORACLE_TORQUE_FF_LIMIT_LIST:-85,45,30,14}"
  --max-prosthesis-qpos-step-list "${PROKNEE_PO_MAX_QPOS_STEP_LIST:-0.008,0.0025,0.002,0.0008}"
  --qvel-loss-weight "${PROKNEE_PO_QVEL_LOSS:-7e-4}"
  --smoothness-loss-weight "${PROKNEE_PO_SMOOTH_LOSS:-0.08}"
  --command-step-loss-weight "${PROKNEE_PO_CMD_STEP_LOSS:-1.5}"
  --command-step-joint-weights "${PROKNEE_PO_CMD_STEP_JOINT_WEIGHTS:-1,3,3,2}"
  --implied-qvel-mag-weight "${PROKNEE_PO_QVEL_MAG_LOSS:-2e-5}"
  --implied-qvel-mag-joint-weights "${PROKNEE_PO_QVEL_MAG_JOINT_WEIGHTS:-1,4,4,2.5}"
  --torque-head
  --torque-loss-weight "${PROKNEE_PO_TORQUE_LOSS:-5e-4}"
  --torque-joint-weights "${PROKNEE_PO_TORQUE_JOINT_WEIGHTS:-3,1.5,1,1}"
  --predicted-torque-ff-scale "${PROKNEE_PO_PRED_TORQUE_FF_SCALE:-1.0}"
  --torque-target-mode "${PROKNEE_PO_TORQUE_TARGET_MODE:-pd_residual}"
  --control-dt 0.01
  --physical-eval
  --physical-eval-steps "$EVAL_ST"
  --physical-qvel-joint-weights "${PROKNEE_PO_PHYSICAL_QVEL_JOINT_WEIGHTS:-1,6,8,16}"
  --physical-toe-z-weight "${PROKNEE_PO_TOE_Z_WEIGHT:-2.0}"
  --physical-root-height-min "${PROKNEE_PO_ROOT_HEIGHT_MIN:-0.85}"
  --physical-root-height-weight "${PROKNEE_PO_ROOT_HEIGHT_WEIGHT:-30.0}"
  --physical-root-up-weight "${PROKNEE_PO_ROOT_UP_WEIGHT:-20.0}"
  --physical-contact-weight "${PROKNEE_PO_CONTACT_WEIGHT:-2.0}"
  --physical-torque-oracle-weight "${PROKNEE_PO_PHYSICAL_TORQUE_ORACLE_WEIGHT:-0.02}"
  --log-interval "$LOG_IV"
  --eval-interval "$EVAL_IV"
  --eval-steps "$EVAL_ST"
  --save-interval "$SAVE_IV"
)

resume_from="$START"

run_phase() {
  local name="$1"
  shift
  echo "========== Prosthesis-only phase: $name =========="
  "${TRAIN[@]}" --resume "$resume_from" "$@"
  resume_from="$ROOT/outputs/$name/stage1_best.pt"
  if [[ ! -f "$resume_from" ]]; then
    echo "[prosthesis_only] Missing phase best checkpoint: $resume_from"
    exit 1
  fi
}

run_phase "${RUN_PREFIX}_r025" \
  --teacher-exec-ratio 0.25 \
  --torque-exec-ratio "${PROKNEE_PO_TORQUE_EXEC_R025:-0.25}" \
  --output-dir "$ROOT/outputs/${RUN_PREFIX}_r025"

run_phase "${RUN_PREFIX}_r050" \
  --teacher-exec-ratio 0.50 \
  --torque-exec-ratio "${PROKNEE_PO_TORQUE_EXEC_R050:-0.50}" \
  --output-dir "$ROOT/outputs/${RUN_PREFIX}_r050"

run_phase "${RUN_PREFIX}_r080" \
  --teacher-exec-ratio 0.80 \
  --torque-exec-ratio "${PROKNEE_PO_TORQUE_EXEC_R080:-0.80}" \
  --output-dir "$ROOT/outputs/${RUN_PREFIX}_r080"

run_phase "${RUN_PREFIX}_r100" \
  --teacher-exec-ratio 1.00 \
  --torque-exec-ratio "${PROKNEE_PO_TORQUE_EXEC_R100:-1.00}" \
  --output-dir "$ROOT/outputs/${RUN_PREFIX}_r100"

echo "[prosthesis_only] Final best checkpoint: $resume_from"
