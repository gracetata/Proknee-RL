#!/bin/bash
# Stage 0: Train full body AMP walking policy using OFFICIAL IsaacGymEnvs
# Uses HumanoidAMP task (dt=0.00556 to match ProKnee-Simulator)
#
# Usage:
#   bash scripts/train_stage0_official.sh [MAX_ITERS] [CHECKPOINT] [EXTRA_ARGS...]
#   bash scripts/train_stage0_official.sh 10000
#   bash scripts/train_stage0_official.sh 15000 runs/.../nn/xxx_750.pth

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

cd "$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"

# Activate environment
export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python

MAX_ITERS=${1:-10000}
CHECKPOINT=${2:-""}
shift 2 2>/dev/null || shift $# 2>/dev/null || true

CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARG="checkpoint=$CHECKPOINT"
fi

MOTION_FILE=${MOTION_FILE:-amp_humanoid_walk.npy}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Stage 0: Full Body AMP Walking                 ║"
echo "║          Official IsaacGymEnvs HumanoidAMP              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  开始时间:    $TIMESTAMP"
echo "║  最大迭代:    $MAX_ITERS"
echo "║  工作目录:    $(pwd)"
echo "║  dt:          0.00556 (180Hz sim, 90Hz control)"
echo "║  motion_file: $MOTION_FILE"
echo "║  num_envs:    4096"
if [ -n "$CHECKPOINT" ]; then
echo "║  恢复训练:    $CHECKPOINT"
fi
echo "║  额外参数:    $@"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  TensorBoard: tensorboard --logdir=$(pwd)/runs/ --port=6006"
echo "║  中断训练:    kill \$\$ (PID: $$) 或 Ctrl-C"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date '+%H:%M:%S')] >>> 训练开始 <<<"
echo ""

$PYTHON train.py \
    task=HumanoidAMP \
    train=HumanoidAMPPPO \
    task.env.motion_file=$MOTION_FILE \
    num_envs=4096 \
    max_iterations=$MAX_ITERS \
    headless=True \
    $CHECKPOINT_ARG \
    "$@"

EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          ✅ Stage 0 训练完成!                            ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  结束时间: $END_TIME"
    echo "║  Checkpoints: $(pwd)/runs/"
    echo "╚══════════════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          ❌ Stage 0 训练异常退出 (code: $EXIT_CODE)      ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  结束时间: $END_TIME"
    echo "║  请检查日志和最近的 checkpoint 以恢复训练               ║"
    echo "╚══════════════════════════════════════════════════════════╝"
fi
