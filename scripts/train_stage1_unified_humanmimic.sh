#!/usr/bin/env bash
# 一键训练 Stage1 Unified（基于 HumanMimic Stage0 body policy），可选 WandB。
#
# 用法（在仓库根目录）:
#   source scripts/activate_proknee_tc_env.sh
#   bash scripts/train_stage1_unified_humanmimic.sh
#
# 可选参数:
#   bash scripts/train_stage1_unified_humanmimic.sh [MAX_EPOCHS]
#   STAGE1_NUM_ENVS=1024 bash scripts/train_stage1_unified_humanmimic.sh 2000
#   STAGE1_BODY_CKPT=/abs/path/to/stage0.pth bash scripts/train_stage1_unified_humanmimic.sh
#
# WandB:
#   export STAGE1_ENABLE_WANDB=1
#   export WANDB_PROJECT=rlleg-stage1-unified
#   export WANDB_ENTITY=your_team   # 可选
#   export WANDB_RUN_NAME=stage1-unified-local   # 可选

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_PREFIX}/bin/python"

MAX_EPOCHS="${1:-8000}"
NUM_ENVS="${STAGE1_NUM_ENVS:-4096}"

DEFAULT_BODY="$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage0_unified/HumanoidAMPUnifiedHumanMimic_13-14-45-32_16000.pth"
BODY_CKPT="${STAGE1_BODY_CKPT:-$DEFAULT_BODY}"
if [ ! -f "$BODY_CKPT" ]; then
  echo "[!] Stage0 body policy 不存在: $BODY_CKPT" >&2
  echo "    请设置 STAGE1_BODY_CKPT=/abs/path/to/your_stage0.pth" >&2
  exit 1
fi

OUT_DIR="${STAGE1_OUT_DIR:-$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage1_unified}"
mkdir -p "$OUT_DIR"
LOG_FILE="${STAGE1_LOG_FILE:-$OUT_DIR/train.log}"
TB_DIR="$OUT_DIR/tb"

WANDB_ARGS=""
if [ "${STAGE1_ENABLE_WANDB:-1}" = "1" ]; then
  WANDB_ARGS="--wandb --wandb-project ${WANDB_PROJECT:-rlleg-stage1-unified}"
  if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ARGS="$WANDB_ARGS --wandb-entity ${WANDB_ENTITY}"
  fi
  if [ -n "${WANDB_RUN_NAME:-}" ]; then
    WANDB_ARGS="$WANDB_ARGS --wandb-run-name ${WANDB_RUN_NAME}"
  fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Stage1 Unified（HumanMimic Stage0 body）                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ max_epochs:   $MAX_EPOCHS"
echo "║ num_envs:     $NUM_ENVS"
echo "║ body_policy:  $BODY_CKPT"
echo "║ output_dir:   $OUT_DIR"
echo "║ log_file:     $LOG_FILE"
echo "║ tensorboard:  tensorboard --logdir=$TB_DIR --port=${STAGE1_TENSORBOARD_PORT:-6007}"
echo "║ wandb:        ${STAGE1_ENABLE_WANDB:-1}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/train_stage1_unified.py \
  --device "${STAGE1_DEVICE:-cuda:0}" \
  --num-envs "$NUM_ENVS" \
  --max-epochs "$MAX_EPOCHS" \
  --body-policy "$BODY_CKPT" \
  --output-dir "$OUT_DIR" \
  $WANDB_ARGS \
  2>&1 | tee "$LOG_FILE"
