#!/bin/bash
# Play / visualize HumanoidAMPUnifiedCurriculum checkpoint (uses train_curriculum_unified.py for task registration).
#
# Usage:
#   bash scripts/play_stage0_unified_curriculum.sh [CHECKPOINT]
#   TASK_PHASE=HumanoidAMPUnifiedCurriculum_phase2 bash scripts/play_stage0_unified_curriculum.sh runs/.../nn/xxx.pth
#
# Env:
#   TASK_PHASE  default HumanoidAMPUnifiedCurriculum_phase3
#   NUM_ENVS    default 4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IGE_DIR="$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"

cd "$IGE_DIR"

CONDA_ENV_ROOT="${CONDA_PREFIX:-/home/cart/miniconda3/envs/proknee_tc}"
export PATH="${CONDA_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_ROOT}/lib:${LD_LIBRARY_PATH}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
PYTHON="${CONDA_ENV_ROOT}/bin/python"

TASK_PHASE="${TASK_PHASE:-HumanoidAMPUnifiedCurriculum_phase3}"
NUM_ENVS="${NUM_ENVS:-4}"

CHECKPOINT="${1:-}"
if [ -n "$CHECKPOINT" ] && [[ "$CHECKPOINT" != /* ]] && [ -f "$PROJECT_DIR/$CHECKPOINT" ]; then
  CHECKPOINT="../../$CHECKPOINT"
fi

if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT=$(ls -t runs/HumanoidAMPUnifiedCurriculum_*/nn/*.pth 2>/dev/null | head -1 || true)
  if [ -z "$CHECKPOINT" ]; then
    echo "[!] 请传入 checkpoint 或确保 runs/HumanoidAMPUnifiedCurriculum_*/nn/*.pth 存在"
    exit 1
  fi
  echo "[i] 使用: $CHECKPOINT"
fi

echo "task=$TASK_PHASE  num_envs=$NUM_ENVS  checkpoint=$CHECKPOINT"

$PYTHON train_curriculum_unified.py \
  task="$TASK_PHASE" \
  train=HumanoidAMPUnifiedCurriculumPPO \
  test=True \
  "num_envs=$NUM_ENVS" \
  headless=False \
  "checkpoint=$CHECKPOINT" \
  "$@"
