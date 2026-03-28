#!/bin/bash
# Stage 0: Train full body AMP walking policy
# Uses ProKnee-Simulator rl_games AMP pipeline (with discriminator)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/ProKnee-Simulator"

# Activate environment
export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python

MAX_ITERS=${1:-5000}
shift 2>/dev/null || true

echo "=========================================="
echo "Stage 0: Full Body AMP Walking Training"
echo "  Max iterations: $MAX_ITERS"
echo "  Working dir: $(pwd)"
echo "=========================================="

# Launch TensorBoard in background
echo "Starting TensorBoard on port 6006..."
tensorboard --logdir="$PROJECT_DIR/ProKnee-Simulator/runs/" --port=6006 &
TB_PID=$!
echo "TensorBoard PID: $TB_PID (http://localhost:6006)"

$PYTHON train.py \
    task=HumanoidAMP \
    train=HumanoidAMPPPO \
    motion=walk \
    num_envs=4096 \
    max_iterations=$MAX_ITERS \
    headless=True \
    "$@"

# Cleanup TensorBoard
kill $TB_PID 2>/dev/null || true
echo "Stage 0 training complete!"
