#!/usr/bin/env bash
# 一键训练 Stage2 Unified（基于 HumanMimic Stage0 + Stage1 Unified teacher）。
#
# 用法（在仓库根目录）:
#   source scripts/activate_proknee_tc_env.sh
#   bash scripts/train_stage2_unified_humanmimic.sh
#
# 可选参数:
#   bash scripts/train_stage2_unified_humanmimic.sh [MAX_STEPS]
#   STAGE2_NUM_ENVS=1024 bash scripts/train_stage2_unified_humanmimic.sh 200000000
#
# 机器间迁移只需改路径:
#   export STAGE2_TEACHER_CKPT=/abs/path/to/stage1_best.pth
#   export STAGE2_BODY_CKPT=/abs/path/to/stage0_body.pth

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_PREFIX}/bin/python"

MAX_STEPS="${1:-500000000}"
NUM_ENVS="${STAGE2_NUM_ENVS:-4096}"
DEVICE="${STAGE2_DEVICE:-cuda:0}"

DEFAULT_STAGE1="$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage1_unified/checkpoints/best.pth"
DEFAULT_STAGE0="$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage0_unified/HumanoidAMPUnifiedHumanMimic_13-14-45-32_16000.pth"

TEACHER_CKPT="${STAGE2_TEACHER_CKPT:-$DEFAULT_STAGE1}"
BODY_CKPT="${STAGE2_BODY_CKPT:-$DEFAULT_STAGE0}"
OUT_DIR="${STAGE2_OUT_DIR:-$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage2_unified}"
LOG_FILE="${STAGE2_LOG_FILE:-$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage2_unified_train.log}"

if [ ! -f "$TEACHER_CKPT" ]; then
  echo "[!] Stage1 teacher 不存在: $TEACHER_CKPT" >&2
  echo "    请设置 STAGE2_TEACHER_CKPT=/abs/path/to/stage1_best.pth" >&2
  exit 1
fi
if [ ! -f "$BODY_CKPT" ]; then
  echo "[!] Stage0 body policy 不存在: $BODY_CKPT" >&2
  echo "    请设置 STAGE2_BODY_CKPT=/abs/path/to/stage0_body.pth" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Stage2 Unified（HumanMimic Stage0 + Stage1 teacher）    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ device:       $DEVICE"
echo "║ num_envs:     $NUM_ENVS"
echo "║ max_steps:    $MAX_STEPS"
echo "║ teacher_ckpt: $TEACHER_CKPT"
echo "║ body_ckpt:    $BODY_CKPT"
echo "║ output_dir:   $OUT_DIR"
echo "║ log_file:     $LOG_FILE"
echo "║ tensorboard:  tensorboard --logdir=$OUT_DIR/tb --port=${STAGE2_TENSORBOARD_PORT:-6008}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/train_stage2_unified.py \
  --device "$DEVICE" \
  --num-envs "$NUM_ENVS" \
  --max-steps "$MAX_STEPS" \
  --teacher-ckpt "$TEACHER_CKPT" \
  --body-policy "$BODY_CKPT" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$LOG_FILE"
