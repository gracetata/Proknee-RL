#!/usr/bin/env bash
# Closed-loop coupled prosthesis action curriculum.
#
# This script is independent from proknee_curriculum_stage1_prosthesis_only.sh.
# It leaves the qpos+torque-residual rollback path untouched.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ="$ROOT/musclemimic"
MM_CKPT="${MM_CKPT:-$ROOT/data/checkpoints/mm-10m-2}"
START="${PROKNEE_CA_START:-$ROOT/outputs/proknee_stage1_prosthesis_only_torqueclip_r100/stage1_best.pt}"
RUN_PREFIX="${PROKNEE_CA_RUN_PREFIX:-proknee_coupled_action}"
STEPS="${PROKNEE_CA_STEPS:-20000}"
ROLLOUT="${PROKNEE_CA_ROLLOUT:-256}"
EVAL_STEPS="${PROKNEE_CA_EVAL_STEPS:-512}"

if [[ ! -f "$START" ]]; then
  echo "[coupled_action] Missing warm-start checkpoint: $START"
  exit 1
fi

TRAIN=(
  env PYTHONUNBUFFERED=1 uv run --project "$PROJ" python -u "$ROOT/musclemimic/scripts/train_proknee_coupled_action.py"
  --checkpoint "$MM_CKPT"
  --dataset-group KIT_KINESIS_TRAINING_MOTIONS
  --val-dataset-group KIT_KINESIS_TESTING_MOTIONS
  --warm-start "$START"
  --steps "$STEPS"
  --rollout-steps "$ROLLOUT"
  --update-epochs "${PROKNEE_CA_UPDATE_EPOCHS:-4}"
  --lr "${PROKNEE_CA_LR:-3e-4}"
  --discount "${PROKNEE_CA_DISCOUNT:-0.98}"
  --clip-eps "${PROKNEE_CA_CLIP_EPS:-0.2}"
  --entropy-weight "${PROKNEE_CA_ENTROPY:-1e-3}"
  --prosthesis-muscle-scale 0.0
  --action-mode "${PROKNEE_CA_ACTION_MODE:-torque}"
  --action-torque-limit-list "${PROKNEE_CA_TORQUE_LIMIT_LIST:-110,65,45,22}"
  --action-torque-slew-limit-list "${PROKNEE_CA_TORQUE_SLEW_LIMIT_LIST:-10,5,3,1.5}"
  --eval-interval "${PROKNEE_CA_EVAL_INTERVAL:-2048}"
  --eval-steps "$EVAL_STEPS"
  --save-interval "${PROKNEE_CA_SAVE_INTERVAL:-10000}"
)

echo "========== Coupled action phase: prosthesis_only =========="
"${TRAIN[@]}" \
  --output-dir "$ROOT/outputs/${RUN_PREFIX}_prosthesis_only"

if [[ "${PROKNEE_CA_ENABLE_BODY_RESIDUAL:-0}" == "1" ]]; then
  START2="$ROOT/outputs/${RUN_PREFIX}_prosthesis_only/coupled_action_best.pt"
  if [[ ! -f "$START2" ]]; then
    START2="$ROOT/outputs/${RUN_PREFIX}_prosthesis_only/coupled_action_final.pt"
  fi
  echo "========== Coupled action phase: body_residual =========="
  env PYTHONUNBUFFERED=1 uv run --project "$PROJ" python -u "$ROOT/musclemimic/scripts/train_proknee_coupled_action.py" \
    --checkpoint "$MM_CKPT" \
    --dataset-group KIT_KINESIS_TRAINING_MOTIONS \
    --val-dataset-group KIT_KINESIS_TESTING_MOTIONS \
    --warm-start "$START2" \
    --steps "$STEPS" \
    --rollout-steps "$ROLLOUT" \
    --update-epochs "${PROKNEE_CA_UPDATE_EPOCHS:-4}" \
    --lr "${PROKNEE_CA_LR_BODY:-1e-4}" \
    --discount "${PROKNEE_CA_DISCOUNT:-0.98}" \
    --clip-eps "${PROKNEE_CA_CLIP_EPS:-0.2}" \
    --entropy-weight "${PROKNEE_CA_ENTROPY:-1e-3}" \
    --prosthesis-muscle-scale 0.0 \
    --action-mode "${PROKNEE_CA_ACTION_MODE:-torque}" \
    --action-torque-limit-list "${PROKNEE_CA_TORQUE_LIMIT_LIST:-110,65,45,22}" \
    --action-torque-slew-limit-list "${PROKNEE_CA_TORQUE_SLEW_LIMIT_LIST:-10,5,3,1.5}" \
    --body-residual-adapter \
    --body-residual-scale "${PROKNEE_CA_BODY_RESIDUAL_SCALE:-0.05}" \
    --eval-interval "${PROKNEE_CA_EVAL_INTERVAL:-2048}" \
    --eval-steps "$EVAL_STEPS" \
    --save-interval "${PROKNEE_CA_SAVE_INTERVAL:-10000}" \
    --output-dir "$ROOT/outputs/${RUN_PREFIX}_body_residual"
fi

echo "[coupled_action] Done."

