#!/usr/bin/env bash
# Supervised prosthesis distillation on GPU (4090).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl
unset JAX_PLATFORMS

MASK_PRESET="${MASK_PRESET:-strict19}"
PROSTHESIS_ACTION_TYPE="${PROSTHESIS_ACTION_TYPE:-torque}"
if [[ "${MASK_PRESET}" == "knee15" ]]; then
  DEFAULT_CONFIG_NAME="conf_fullbody_prosthesis_gmr_resnet_knee15"
  DEFAULT_DATASET_DIR="data/teacher_rollouts_knee15/KIT_KINESIS_TRAINING_MOTIONS"
else
  DEFAULT_CONFIG_NAME="conf_fullbody_prosthesis_gmr_resnet"
  DEFAULT_DATASET_DIR="data/teacher_rollouts/KIT_KINESIS_TRAINING_MOTIONS"
fi
CONFIG_NAME="${CONFIG_NAME:-${DEFAULT_CONFIG_NAME}}"
DATASET_DIR="${DATASET_DIR:-${DEFAULT_DATASET_DIR}}"

uv run python -c "import jax; print('JAX devices:', jax.devices())"

uv run python scripts/train_prosthesis_distillation.py \
  --dataset_dir "${DATASET_DIR}" \
  --config-name "${CONFIG_NAME}" \
  --mask-preset "${MASK_PRESET}" \
  --prosthesis_action_type "${PROSTHESIS_ACTION_TYPE}" \
  --init official_trunk_init \
  --teacher_checkpoint /home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2 \
  --epochs "${EPOCHS:-50}" \
  --batch_size "${BATCH_SIZE:-4096}" \
  --lambda_prosthesis 10.0 \
  --wandb "${WANDB:-disabled}" \
  "$@"
