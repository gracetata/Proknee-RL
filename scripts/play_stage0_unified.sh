#!/bin/bash
# Stage 0 Unified：在 Isaac Gym 窗口中播放 HumanoidAMPUnified 检查点
# 结构与 scripts/play_stage0_amp.sh 一致，仅 task/train、默认 motion 不同。
#
# 用法:
#   conda activate proknee_tc
#   bash scripts/play_stage0_unified.sh
#   bash scripts/play_stage0_unified.sh runs/HumanoidAMPUnified_28-17-26-13/nn/HumanoidAMPUnified_28-17-26-13_3950.pth
#   MOTION_FILE=multi_walk_run.yaml bash scripts/play_stage0_unified.sh
#   bash scripts/play_stage0_unified.sh ../../outputs/checkpoints/stage0/stage0_unified_1800.pth
#
# 若在仓库根目录习惯写 outputs/...，也可:
#   bash scripts/play_stage0_unified.sh outputs/checkpoints/stage0/stage0_unified_1800.pth
#
# 依赖: 与 train_stage0_unified.sh 相同；需本机图形界面（headless=False）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IGE_DIR="$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"

cd "$IGE_DIR"

# ── 与 train_stage0_unified.sh / play_stage0_amp.sh 一致 ─────────────────
CONDA_ENV_ROOT="${CONDA_PREFIX:-/home/cart/miniconda3/envs/proknee_tc}"
export PATH="${CONDA_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_ROOT}/lib:${LD_LIBRARY_PATH}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
PYTHON="${CONDA_ENV_ROOT}/bin/python"

# ── 播放时改这里，或用环境变量覆盖（须与训练该 checkpoint 时一致）──────────
#   MOTION_FILE  默认 multi_walk_run.yaml（见 cfg/task/HumanoidAMPUnified.yaml）
#   NUM_ENVS     可视化并行环境数，显存不足可改为 1
MOTION_FILE="${MOTION_FILE:-multi_walk_run.yaml}"
NUM_ENVS="${NUM_ENVS:-4}"

CHECKPOINT="${1:-}"
# 支持在仓库根目录下的相对路径（如 outputs/checkpoints/...）
if [ -n "$CHECKPOINT" ] && [[ "$CHECKPOINT" != /* ]] && [ -f "$PROJECT_DIR/$CHECKPOINT" ]; then
  CHECKPOINT="../../$CHECKPOINT"
fi

if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT=$(ls -t runs/HumanoidAMPUnified_*/nn/*.pth 2>/dev/null | head -1 || true)
  if [ -z "$CHECKPOINT" ]; then
    if [ -f "$PROJECT_DIR/outputs/checkpoints/stage0/stage0_unified_1800.pth" ]; then
      CHECKPOINT="../../outputs/checkpoints/stage0/stage0_unified_1800.pth"
    else
      _fb=$(ls -t "$PROJECT_DIR"/outputs/checkpoints/stage0/stage0_unified*.pth 2>/dev/null | head -1 || true)
      if [ -n "$_fb" ]; then
        CHECKPOINT="../../outputs/checkpoints/stage0/$(basename "$_fb")"
      fi
    fi
  fi
  if [ -z "$CHECKPOINT" ]; then
    echo "[!] 未找到 checkpoint。请显式传入，例如:"
    echo "    bash scripts/play_stage0_unified.sh runs/HumanoidAMPUnified_<时间>/nn/<xxx>.pth"
    echo "    bash scripts/play_stage0_unified.sh outputs/checkpoints/stage0/stage0_unified_1800.pth"
    exit 1
  fi
  echo "[i] 使用自动解析的 checkpoint: $CHECKPOINT"
else
  shift
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Stage 0 Unified 播放 (test=True, headless=False)       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  task:        HumanoidAMPUnified                         ║"
echo "║  train:       HumanoidAMPUnifiedPPO                      ║"
echo "║  motion_file: $MOTION_FILE"
echo "║  num_envs:    $NUM_ENVS"
echo "║  checkpoint:  $CHECKPOINT"
echo "║  工作目录:    $(pwd)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

$PYTHON train.py \
  task=HumanoidAMPUnified \
  train=HumanoidAMPUnifiedPPO \
  task.env.motion_file="$MOTION_FILE" \
  test=True \
  "num_envs=$NUM_ENVS" \
  headless=False \
  "checkpoint=$CHECKPOINT" \
  "$@"
