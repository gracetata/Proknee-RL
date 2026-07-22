#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU COMMAND [ARG ...]" >&2
  exit 2
fi

GPU="$1"
shift
if [[ "${GPU}" != "5" ]]; then
  echo "the current training phase permits physical GPU 5 only" >&2
  exit 2
fi

source "${ROOT}/scripts/a100_env.sh"
export CUDA_VISIBLE_DEVICES="${GPU}"
exec "$@"
