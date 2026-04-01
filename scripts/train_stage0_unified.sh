#!/bin/bash
# Stage 0 Unified: Train velocity-conditioned AMP body policy
# Single policy handles walk (1.0 m/s), run (2.5 m/s), stand (0.0 m/s)
#
# Usage:
#   bash scripts/train_stage0_unified.sh [MAX_ITERS] [CHECKPOINT]
#   bash scripts/train_stage0_unified.sh 10000
#   bash scripts/train_stage0_unified.sh 15000 runs/.../nn/xxx_750.pth
#
# 默认 num_envs=4096（RTX 4090 24GB 左右）；显存不够: export STAGE0_NUM_ENVS=2048 或 1024

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_rlleg_env.sh"

cd "$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"
PYTHON="${CONDA_PREFIX}/bin/python"
NUM_ENVS="${STAGE0_NUM_ENVS:-4096}"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/stage0_batch_hydra.sh"

MAX_ITERS=${1:-10000}
CHECKPOINT=${2:-""}
shift 2 2>/dev/null || shift $# 2>/dev/null || true

CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARG="checkpoint=$CHECKPOINT"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║    Stage 0 Unified: Velocity-Conditioned AMP            ║"
echo "║    Walk(1.0) / Run(2.5) / Stand(0.0) — Single Policy   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  开始时间:    $TIMESTAMP"
echo "║  最大迭代:    $MAX_ITERS"
echo "║  motion_file: multi_walk_run_stand.yaml"
echo "║  obs:         106D (105D + 1D velocity cmd)"
echo "║  task_reward:  0.5 (velocity tracking)"
echo "║  disc_reward:  0.5 (AMP style)"
echo "║  num_envs:    $NUM_ENVS (STAGE0_NUM_ENVS)"
if [ -n "$CHECKPOINT" ]; then
echo "║  恢复训练:    $CHECKPOINT"
fi
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  TensorBoard: tensorboard --logdir=$(pwd)/runs/ --port=6006"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

$PYTHON train.py \
    task=HumanoidAMPUnified \
    train=HumanoidAMPUnifiedPPO \
    num_envs=$NUM_ENVS \
    train.params.config.minibatch_size=$STAGE0_MB \
    train.params.config.amp_minibatch_size=$STAGE0_AMP_MB \
    max_iterations=$MAX_ITERS \
    headless=True \
    $CHECKPOINT_ARG \
    "$@"

EXIT_CODE=$?
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Stage 0 Unified 训练完成! ($END_TIME)"
    echo "   Checkpoints: $(pwd)/runs/"
else
    echo "❌ Stage 0 Unified 训练异常退出 (code: $EXIT_CODE, $END_TIME)"
fi
