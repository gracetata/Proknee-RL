#!/usr/bin/env bash
# Full-dataset knee15 coupled-action RL (frozen official + trainable 4DOF prosthesis).
#
# Train set: KIT_KINESIS_TRAINING_MOTIONS (~972 motions)
# Val set:   KIT_KINESIS_TESTING_MOTIONS (~108 motions)
#
# Success criteria (manual stop / checkpoint selection):
#   - eval/avg_survival approaches eval_steps with low eval/resets
#   - eval/tracking_mae stays small while surviving full episodes
#   - periodic composite eval on test motions via score_proknee_coupled_action.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ="$ROOT/musclemimic"
VENV="${PROKNEE_CA_VENV:-$PROJ/.venv/bin/python}"
MM_CKPT="${MM_CKPT:-$ROOT/data/checkpoints/mm-10m-2}"
OUT="${PROKNEE_CA_OUTPUT:-$ROOT/musclemimic/outputs/proknee_coupled_knee15_full}"
WARM_START="${PROKNEE_CA_WARM_START:-}"
STEPS="${PROKNEE_CA_STEPS:-500000}"
ROLLOUT="${PROKNEE_CA_ROLLOUT:-256}"
EVAL_INTERVAL="${PROKNEE_CA_EVAL_INTERVAL:-4096}"
EVAL_STEPS="${PROKNEE_CA_EVAL_STEPS:-512}"
SAVE_INTERVAL="${PROKNEE_CA_SAVE_INTERVAL:-50000}"
TB_PORT="${PROKNEE_CA_TB_PORT:-6006}"

mkdir -p "$OUT"

CMD=(
  env PYTHONUNBUFFERED=1 MUJOCO_GL=egl JAX_PLATFORMS=cpu
  "$VENV" -u "$PROJ/scripts/train_proknee_coupled_action.py"
  --checkpoint "$MM_CKPT"
  --dataset-group KIT_KINESIS_TRAINING_MOTIONS
  --val-dataset-group KIT_KINESIS_TESTING_MOTIONS
  --output-dir "$OUT"
  --mask-preset knee15
  --obs-mode easy
  --prosthesis-muscle-scale 0.0
  --action-mode torque
  --action-torque-limit-list 110,65,45,22
  --action-torque-slew-limit-list 10,5,3,1.5
  --steps "$STEPS"
  --rollout-steps "$ROLLOUT"
  --update-epochs "${PROKNEE_CA_UPDATE_EPOCHS:-4}"
  --lr "${PROKNEE_CA_LR:-3e-4}"
  --discount "${PROKNEE_CA_DISCOUNT:-0.98}"
  --clip-eps "${PROKNEE_CA_CLIP_EPS:-0.2}"
  --entropy-weight "${PROKNEE_CA_ENTROPY:-1e-3}"
  --eval-interval "$EVAL_INTERVAL"
  --eval-steps "$EVAL_STEPS"
  --save-interval "$SAVE_INTERVAL"
  --tensorboard-dir "$OUT/tensorboard"
)

if [[ -n "$WARM_START" ]]; then
  CMD+=(--warm-start "$WARM_START" --warm-start-prosthesis-head auto)
fi

echo "[proknee_coupled_knee15_full] output=$OUT"
echo "[proknee_coupled_knee15_full] steps=$STEPS rollout=$ROLLOUT eval_interval=$EVAL_INTERVAL"
echo "[proknee_coupled_knee15_full] tensorboard: $OUT/tensorboard (port $TB_PORT)"
echo "[proknee_coupled_knee15_full] launch tensorboard in another terminal:"
echo "  cd $PROJ && .venv/bin/tensorboard --logdir $OUT/tensorboard --port $TB_PORT --bind_all"
echo

"${CMD[@]}" 2>&1 | tee "$OUT/train.log"
