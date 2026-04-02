#!/usr/bin/env bash
# HumanMimic-style Stage0: velocity grid 0~3 m/s, dual AMP heads (imitation + style), gap velocity blending.
# Uses train_humanmimic_unified.py (parallel stack). Backup: train_curriculum_unified.py + HumanoidAMPUnifiedCurriculum_*.
#
# Usage:
#   bash scripts/train_stage0_unified_humanmimic_phase1.sh [MAX_ITERS] [CHECKPOINT]
#
# 默认 num_envs=4096（RTX 4090 24GB 左右）；显存不够: export STAGE0_NUM_ENVS=2048 或 1024
# 需 conda 环境 proknee_tc（见 docs/ENVIRONMENT_RLLEG.md）；脚本内已 source activate_proknee_tc_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"

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
echo "HumanoidAMPUnifiedHumanMimic (phase1) — train_humanmimic_unified.py"
echo "Started: $TIMESTAMP  max_iterations=$MAX_ITERS  num_envs=$NUM_ENVS  mb=$STAGE0_MB"
echo ""

$PYTHON train_humanmimic_unified.py \
  task=HumanoidAMPUnifiedHumanMimic_phase1 \
  train=HumanoidAMPUnifiedHumanMimicPPO \
  num_envs=$NUM_ENVS \
  train.params.config.minibatch_size=$STAGE0_MB \
  train.params.config.amp_minibatch_size=$STAGE0_AMP_MB \
  max_iterations=$MAX_ITERS \
  headless=True \
  $CHECKPOINT_ARG \
  "$@"
