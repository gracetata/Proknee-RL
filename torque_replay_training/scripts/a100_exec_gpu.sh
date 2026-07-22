#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU COMMAND [ARG ...]" >&2
  exit 2
fi

GPU="$1"
shift
if [[ ! "${GPU}" =~ ^[567]$ ]]; then
  echo "only physical GPU 5, 6, or 7 is permitted" >&2
  exit 2
fi

source "${ROOT}/scripts/a100_env.sh"
export CUDA_VISIBLE_DEVICES="${GPU}"
exec "$@"
