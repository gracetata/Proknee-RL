#!/bin/bash
# Stage 0：在 Isaac Gym 窗口中播放 HumanoidAMP 检查点（USE.md「Stage 0 (全身策略)」）
#
# 用法:
#   bash scripts/play_stage0_amp.sh
#   bash scripts/play_stage0_amp.sh runs/HumanoidAMP_28-16-59-36/nn/HumanoidAMP_28-16-59-37_1150.pth
#   MOTION_FILE=amp_humanoid_run.npy bash scripts/play_stage0_amp.sh
#
# 依赖: conda activate proknee_tc；需本机图形界面（headless=False）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IGE_DIR="$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"

cd "$IGE_DIR"

# ── 与 train_stage0_official.sh 一致 ─────────────────
CONDA_ENV_ROOT="${CONDA_PREFIX:-/home/cart/miniconda3/envs/proknee_tc}"
export PATH="${CONDA_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_ROOT}/lib:${LD_LIBRARY_PATH}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
PYTHON="${CONDA_ENV_ROOT}/bin/python"

# ── 播放时改这里，或用环境变量覆盖 ─────────────────
#   MOTION_FILE  必须与训练该 checkpoint 时一致（默认行走）
#   NUM_ENVS     可视化并行环境数，显存不足可改为 1
MOTION_FILE="${MOTION_FILE:-amp_humanoid_walk.npy}"
NUM_ENVS="${NUM_ENVS:-4}"

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT=$(ls -t runs/HumanoidAMP_*/nn/*.pth 2>/dev/null | head -1 || true)
  if [ -z "$CHECKPOINT" ]; then
    echo "[!] 未找到 runs/HumanoidAMP_*/nn/*.pth，请显式传入 checkpoint 路径，例如:"
    echo "    bash scripts/play_stage0_amp.sh runs/HumanoidAMP_<时间>/nn/<xxx>.pth"
    exit 1
  fi
  echo "[i] 使用最新 checkpoint: $CHECKPOINT"
else
  shift
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Stage 0 AMP 播放 (test=True, headless=False)           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  motion_file: $MOTION_FILE"
echo "║  num_envs:    $NUM_ENVS"
echo "║  checkpoint:  $CHECKPOINT"
echo "║  工作目录:    $(pwd)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

$PYTHON train.py \
  task=HumanoidAMP \
  train=HumanoidAMPPPO \
  task.env.motion_file="$MOTION_FILE" \
  test=True \
  "num_envs=$NUM_ENVS" \
  headless=False \
  "checkpoint=$CHECKPOINT" \
  "$@"
