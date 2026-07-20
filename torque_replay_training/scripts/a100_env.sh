#!/usr/bin/env bash
# Shared A100 environment. Source this file or use a100_exec.sh.

export CUDA_VISIBLE_DEVICES="5,6,7"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export MUJOCO_GL="egl"
export PYTHONUNBUFFERED="1"

