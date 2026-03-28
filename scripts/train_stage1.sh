#!/bin/bash
# ================================================================
#  Stage 1: Teacher Policy Training (DAgger, GPU)
#
#  DAgger behavioral cloning from Stage 0 body policy.
#  Teacher rolls out its own actions to avoid distribution shift.
#
#  Usage:
#    bash scripts/train_stage1.sh                    # default args
#    bash scripts/train_stage1.sh --max-epochs 8000  # override
#
#  Interrupt: Ctrl+C or kill <PID>
#  Monitor:   tensorboard --logdir outputs/stage1_dagger_*/tb --port 6007
# ================================================================

set -e
cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python

echo "========================================"
echo "  Stage 1: Teacher Policy Training"
echo "  Start time: $(date)"
echo "========================================"

$PYTHON scripts/train_stage1_dagger.py \
    --device cuda:0 \
    --num-envs 4096 \
    --max-epochs 5000 \
    "$@"

echo "========================================"
echo "  Stage 1 training finished"
echo "  End time: $(date)"
echo "========================================"
