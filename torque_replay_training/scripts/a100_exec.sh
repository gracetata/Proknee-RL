#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/a100_env.sh"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

exec "$@"

