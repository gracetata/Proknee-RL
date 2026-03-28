#!/bin/bash
# ================================================================
#  Stage 1: Teacher Policy Training (GPU)
#
#  Train prosthetic knee teacher with privileged information.
#  Frozen Stage 0 body policy controls 24 DOFs.
#  Teacher controls left knee (1 DOF). Ankle is passive (locked).
#
#  Usage:
#    # Direct run (blocks terminal):
#    bash scripts/train_stage1.sh
#
#    # Background run with tmux:
#    tmux new -s stage1
#    bash scripts/train_stage1.sh
#    # Ctrl+B, D to detach
#
#    # Background run with nohup:
#    nohup bash scripts/train_stage1.sh > stage1_train.log 2>&1 &
#
#  Interrupt:
#    Ctrl+C (direct/tmux) or kill <PID> (nohup)
#
#  Monitor:
#    tensorboard --logdir outputs/stage1_teacher_*/tb --port 6007
# ================================================================

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python
export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH

echo "========================================"
echo "  Stage 1: Teacher Policy Training"
echo "  Start time: $(date)"
echo "========================================"

$PYTHON scripts/train_stage1.py \
    --device cuda:0 \
    --num-envs 4096 \
    --max-epochs 3000 \
    --lr 1e-3 \
    --horizon 16 \
    --minibatch-size 8192 \
    --mini-epochs 5 \
    --gamma 0.99 \
    --tau 0.95 \
    --e-clip 0.2 \
    --entropy-coef 0.01 \
    --save-interval 100 \
    --log-interval 10 \
    --body-policy outputs/checkpoints/stage0/stage0_amp_walk_5050.pth \
    "$@"

echo "========================================"
echo "  Stage 1 training finished"
echo "  End time: $(date)"
echo "========================================"
