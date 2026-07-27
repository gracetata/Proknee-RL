#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION_LIST="${ROOT}/configs/all_available_motions.txt"
CACHE_ROOT="${HOME}/.musclemimic/caches/AMASS/MyoFullBody/gmr"
REMOTE="${PROKNEE_A100_SSH:-root@39.105.12.60}"
PORT="${PROKNEE_A100_PORT:-6029}"
REMOTE_CACHE_ROOT="/root/.musclemimic/caches/AMASS/MyoFullBody/gmr"

missing=0
while IFS= read -r motion; do
  [[ -z "${motion}" || "${motion}" == \#* ]] && continue
  if [[ ! -f "${CACHE_ROOT}/${motion}.npz" ]]; then
    echo "missing local GMR cache: ${CACHE_ROOT}/${motion}.npz" >&2
    missing=$((missing + 1))
  fi
done < "${MOTION_LIST}"
if (( missing > 0 )); then
  echo "refusing to sync: ${missing} local caches are missing" >&2
  exit 1
fi

while IFS= read -r motion; do
  [[ -z "${motion}" || "${motion}" == \#* ]] && continue
  printf '%s.npz\0' "${motion}"
done < "${MOTION_LIST}" |
  tar -C "${CACHE_ROOT}" --null -T - -cf - |
  ssh -p "${PORT}" -o BatchMode=yes "${REMOTE}" \
    "mkdir -p '${REMOTE_CACHE_ROOT}' && tar -C '${REMOTE_CACHE_ROOT}' -xf -"

local_count="$(wc -l < "${MOTION_LIST}")"
remote_count="$(
  ssh -p "${PORT}" -o BatchMode=yes "${REMOTE}" \
    "find '${REMOTE_CACHE_ROOT}' -type f -name '*.npz' | wc -l"
)"
echo "motion list: ${local_count}; remote cache files: ${remote_count}"
