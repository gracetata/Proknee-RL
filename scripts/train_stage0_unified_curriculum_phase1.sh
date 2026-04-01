#!/bin/bash
# Stage 0 Unified Curriculum — Phase 1 (no mid-episode velocity switch).
# Uses train_curriculum_unified.py to register HumanoidAMPUnifiedCurriculum without editing __init__.py
#
# Usage:
#   bash scripts/train_stage0_unified_curriculum_phase1.sh [MAX_ITERS] [CHECKPOINT]
#
# Defaults match HumanoidAMPUnified YAML: num_envs=4096, minibatch 32768 / amp 4096 (from HumanoidAMPUnifiedCurriculumPPO.yaml).
# For low VRAM, override e.g.: num_envs=24 train.params.config.minibatch_size=384 train.params.config.amp_minibatch_size=384

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_rlleg_env.sh"

cd "$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"
PYTHON="${CONDA_PREFIX}/bin/python"
# 默认 4096（RTX 4090 等）；显存不足: export STAGE0_NUM_ENVS=1024 或 2048
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
echo "║  Stage0 Unified Curriculum — Phase 1                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  开始时间:    $TIMESTAMP"
echo "║  最大迭代:    $MAX_ITERS"
echo "║  task:        HumanoidAMPUnifiedCurriculum_phase1"
echo "║  train:       HumanoidAMPUnifiedCurriculumPPO"
echo "║  num_envs:    $NUM_ENVS  minibatch=$STAGE0_MB  amp_mb=$STAGE0_AMP_MB"
echo "║  entry:       train_curriculum_unified.py"
if [ -n "$CHECKPOINT" ]; then
echo "║  恢复训练:    $CHECKPOINT"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

$PYTHON train_curriculum_unified.py \
  task=HumanoidAMPUnifiedCurriculum_phase1 \
  train=HumanoidAMPUnifiedCurriculumPPO \
  num_envs=$NUM_ENVS \
  train.params.config.minibatch_size=$STAGE0_MB \
  train.params.config.amp_minibatch_size=$STAGE0_AMP_MB \
  max_iterations=$MAX_ITERS \
  headless=True \
  $CHECKPOINT_ARG \
  "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Phase 1 完成. Checkpoints: $(pwd)/runs/"
else
  echo "❌ Phase 1 退出码: $EXIT_CODE"
fi
exit $EXIT_CODE
