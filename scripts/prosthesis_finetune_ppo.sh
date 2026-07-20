#!/usr/bin/env bash
# PPO fine-tune from distilled prosthesis checkpoint on GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset JAX_PLATFORMS

DISTILLED_CKPT="${1:-${ROOT}/outputs/prosthesis_distill/latest/checkpoints/checkpoint_distilled}"
cd "${ROOT}/fullbody"

uv run python -c "import jax; print('JAX devices:', jax.devices())"

uv run python experiment.py --config-name=conf_fullbody_prosthesis_finetune \
  experiment.resume_from_distilled="${DISTILLED_CKPT}" \
  experiment.auto_resume=false \
  "${@:2}"
