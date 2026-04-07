#!/usr/bin/env bash
# 在 **proknee_tc** conda 中，用 HumanMimic Unified Stage0 权重衔接 Stage1 → Stage2（Unified）。
#
# 是否需要 conda：是。本脚本会 source activate_proknee_tc_env.sh 激活环境（等价于先 conda activate proknee_tc）。
# 远程：将 Stage0 文件放到 outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth 后，在项目根执行：
#   bash scripts/train_unified_stage1_stage2_from_humanmimic.sh
#   bash scripts/train_unified_stage1_stage2_from_humanmimic.sh   # 默认后台启动 Stage1
# 或只跑 Stage1：RUN_STAGE1=1 bash ...
#
# 所有训练输出固定写在项目根下：
#   outputs/humanmimic_policy_knee_ankle_vel/
#     （Stage0 权重默认放仓库根）outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth
#     stage0_unified/          # 仅当自行训 Stage0 时：train_dir（HumanoidAMPUnifiedHumanMimic_*/nn/*.pth）
#     stage1_unified/          # Stage1：--output-dir、checkpoints/best.pth、train.log
#     stage2_unified/          # Stage2：--output-dir、checkpoints、tb
#     stage2_train.log         # Stage2 nohup 标准输出
#
# 可选环境变量：
#   STAGE0_BODY_CKPT   覆盖 Stage0。不设时顺序：
#                      outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth →
#                      humanmimic_policy_knee_ankle_vel/stage0_unified/body_policy.pth →
#                      stage0_unified 下最新 nn/*.pth → isaacgymenvs/runs/.../nn/*.pth
#   RUN_STAGE1=0       仅打印命令，不启动 Stage1（默认会启动 Stage1）
#   RUN_STAGE2=1       后台启动 Stage2（需先有 Stage1 的 checkpoints/best.pth）
#   SMOKE_TEST=1       本地冒烟：减小 num-envs / max-epochs（Stage1）；Stage2 减小 max-steps

set -e
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-proknee_tc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"

PYTHON="${CONDA_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "$PROJECT_DIR"

OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel}"
STAGE1_OUT="${STAGE1_OUT:-$OUTPUT_BASE/stage1_unified}"
STAGE2_OUT="${STAGE2_OUT:-$OUTPUT_BASE/stage2_unified}"
STAGE1_BEST="${STAGE1_BEST:-$STAGE1_OUT/checkpoints/best.pth}"
STAGE2_LOG="${STAGE2_LOG:-$OUTPUT_BASE/stage2_train.log}"

STAGE0_DEFAULT="$PROJECT_DIR/outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth"
STAGE0_ALT="$OUTPUT_BASE/stage0_unified/body_policy.pth"
if [ -n "${STAGE0_BODY_CKPT:-}" ]; then
  STAGE0="$STAGE0_BODY_CKPT"
else
  STAGE0="$STAGE0_DEFAULT"
fi
if [ -f "$STAGE0" ]; then
  STAGE0="$(realpath "$STAGE0")"
elif [ -z "${STAGE0_BODY_CKPT:-}" ] && [ -f "$STAGE0_ALT" ]; then
  STAGE0="$(realpath "$STAGE0_ALT")"
  echo "[info] 使用备选 Stage0: $STAGE0"
elif [ -z "${STAGE0_BODY_CKPT:-}" ]; then
  _latest="$(find "$OUTPUT_BASE/stage0_unified" \
    -path '*/HumanoidAMPUnifiedHumanMimic_*/nn/*.pth' -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)"
  if [ -n "$_latest" ] && [ -f "$_latest" ]; then
    STAGE0="$(realpath "$_latest")"
    echo "[info] 使用 outputs/.../stage0_unified 下最新 nn/*.pth:"
    echo "       $STAGE0"
  fi
fi
if [ ! -f "$STAGE0" ] && [ -z "${STAGE0_BODY_CKPT:-}" ]; then
  _latest="$(find "$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs/runs" \
    -path '*/HumanoidAMPUnifiedHumanMimic_*/nn/*.pth' -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)"
  if [ -n "$_latest" ] && [ -f "$_latest" ]; then
    STAGE0="$(realpath "$_latest")"
    echo "[info] 使用 isaacgymenvs/runs 下最新 HumanMimic 检查点:"
    echo "       $STAGE0"
  fi
fi

# 默认训练规模；SMOKE_TEST=1 时缩小以便本地快速验证能起训、写目录
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_EPOCHS="${MAX_EPOCHS:-8000}"
MAX_STEPS="${MAX_STEPS:-500000000}"
if [ "${SMOKE_TEST:-0}" = "1" ]; then
  NUM_ENVS="${NUM_ENVS_SMOKE:-512}"
  MAX_EPOCHS="${MAX_EPOCHS_SMOKE:-5}"
  MAX_STEPS="${MAX_STEPS_SMOKE:-50000}"
  echo "[SMOKE_TEST] num-envs=$NUM_ENVS max-epochs=$MAX_EPOCHS max-steps=$MAX_STEPS"
fi

echo "══════════════════════════════════════════════════════════════"
echo " Unified Stage1 / Stage2（body = HumanMimic Stage0）"
echo "══════════════════════════════════════════════════════════════"
echo "  conda:       $CONDA_PREFIX"
echo "  Python:      $PYTHON"
echo "  输出根目录:  $OUTPUT_BASE"
echo "  Stage0:      $STAGE0"
echo "  Stage1 目录: $STAGE1_OUT"
echo "  Stage2 目录: $STAGE2_OUT"
echo "  Stage1 best: $STAGE1_BEST"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ ! -f "$STAGE0" ]; then
  echo "[!] 未找到 Stage0。请将检查点放到（相对仓库根）:" >&2
  echo "    outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth" >&2
  echo "    或: export STAGE0_BODY_CKPT=/绝对路径/xxx.pth" >&2
  exit 1
fi

mkdir -p "$OUTPUT_BASE"

echo "【Stage 1 Unified】Teacher DAgger"
echo "  mkdir -p \"$STAGE1_OUT\""
echo "  PYTHONUNBUFFERED=1 nohup \"$PYTHON\" scripts/train_stage1_unified.py \\"
echo "    --device cuda:0 --num-envs $NUM_ENVS --max-epochs $MAX_EPOCHS \\"
echo "    --body-policy \"$STAGE0\" \\"
echo "    --output-dir \"$STAGE1_OUT\" \\"
echo "    > \"$STAGE1_OUT/train.log\" 2>&1 &"
echo ""
echo "【Stage 2 Unified】蒸馏（需 $STAGE1_BEST 存在）"
echo "  PYTHONUNBUFFERED=1 nohup \"$PYTHON\" scripts/train_stage2_unified.py \\"
echo "    --device cuda:0 --num-envs $NUM_ENVS \\"
echo "    --body-policy \"$STAGE0\" \\"
echo "    --teacher-ckpt \"$STAGE1_BEST\" \\"
echo "    --output-dir \"$STAGE2_OUT\" \\"
echo "    --max-steps $MAX_STEPS \\"
echo "    > \"$STAGE2_LOG\" 2>&1 &"
echo ""

if [ "${RUN_STAGE1:-1}" = "1" ]; then
  mkdir -p "$STAGE1_OUT"
  PYTHONUNBUFFERED=1 nohup "$PYTHON" scripts/train_stage1_unified.py \
    --device cuda:0 --num-envs "$NUM_ENVS" --max-epochs "$MAX_EPOCHS" \
    --body-policy "$STAGE0" \
    --output-dir "$STAGE1_OUT" \
    > "$STAGE1_OUT/train.log" 2>&1 &
  echo "[ok] Stage1 已在后台启动，日志: $STAGE1_OUT/train.log"
fi

if [ "${RUN_STAGE2:-0}" = "1" ]; then
  if [ ! -f "$STAGE1_BEST" ]; then
    echo "[!] 未找到 Stage1 teacher: $STAGE1_BEST" >&2
    echo "    先完成 Stage1 或设置 STAGE1_BEST=..." >&2
    exit 1
  fi
  mkdir -p "$STAGE2_OUT"
  PYTHONUNBUFFERED=1 nohup "$PYTHON" scripts/train_stage2_unified.py \
    --device cuda:0 --num-envs "$NUM_ENVS" \
    --body-policy "$STAGE0" \
    --teacher-ckpt "$STAGE1_BEST" \
    --output-dir "$STAGE2_OUT" \
    --max-steps "$MAX_STEPS" \
    > "$STAGE2_LOG" 2>&1 &
  echo "[ok] Stage2 已在后台启动，日志: $STAGE2_LOG"
fi
