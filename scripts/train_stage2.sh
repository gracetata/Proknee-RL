#!/bin/bash
# ================================================================
#  Stage 2: Student Policy Training (ProprioAdapt, GPU)
#
#  Trains adapt_tconv to predict latent from proprio_hist.
#  Everything else frozen from Stage 1.
#
#  Usage:
#    bash scripts/train_stage2.sh                    # default args
#    bash scripts/train_stage2.sh --max-steps 1e9    # override
#
#  Interrupt: Ctrl+C or kill <PID>
#  Monitor:   tensorboard --logdir outputs/stage2_padapt_*/tb --port 6008
# ================================================================

set -e
cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python

echo "========================================"
echo "  Stage 2: Student Policy Training"
echo "  Start time: $(date)"
echo "========================================"

$PYTHON scripts/train_stage2.py \
    --device cuda:0 \
    --num-envs 4096 \
    "$@"

echo "========================================"
echo "  Stage 2 training finished"
echo "  End time: $(date)"
echo "========================================"
