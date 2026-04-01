#!/usr/bin/env bash
# 根据 NUM_ENVS 与 horizon_length=16 设置 minibatch，使 batch_size 可被 minibatch 整除。
# 先 export NUM_ENVS，再: source "$(dirname "$0")/stage0_batch_hydra.sh"
# 导出: STAGE0_MB, STAGE0_AMP_MB

HORIZON="${STAGE0_HORIZON:-16}"
BATCH=$((HORIZON * NUM_ENVS))

if [ "$BATCH" -ge 32768 ]; then
  STAGE0_MB=32768
elif [ "$BATCH" -ge 16384 ]; then
  STAGE0_MB=16384
elif [ "$BATCH" -ge 8192 ]; then
  STAGE0_MB=8192
elif [ "$BATCH" -ge 4096 ]; then
  STAGE0_MB=4096
else
  STAGE0_MB=$BATCH
fi

STAGE0_AMP_MB=4096
if [ "$STAGE0_MB" -lt "$STAGE0_AMP_MB" ]; then
  STAGE0_AMP_MB=$STAGE0_MB
fi
