#!/bin/bash
# Stage 0: Train standing body policy using AMP
#
# Requires: amp_humanoid_stand.npy reference motion
# Generate it first: python scripts/create_stand_motion.py
#
# Usage:
#   bash scripts/train_stage0_stand.sh [MAX_ITERS]
#   bash scripts/train_stage0_stand.sh 8000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# First, ensure standing motion exists
STAND_MOTION="$PROJECT_DIR/IsaacGymEnvs/assets/amp/motions/amp_humanoid_stand.npy"
if [ ! -f "$STAND_MOTION" ]; then
    echo "[!] Standing motion not found. Generating..."
    export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
    /home/user/anaconda3/envs/proknee_tc/bin/python "$PROJECT_DIR/scripts/create_stand_motion.py"
fi

# Train using the standard Stage 0 script with standing motion
MOTION_FILE=amp_humanoid_stand.npy \
    bash "$PROJECT_DIR/scripts/train_stage0_official.sh" "${1:-8000}"
