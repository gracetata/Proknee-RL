#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BUNDLE="${ROOT}/runtime/hora_torch_py311_cu126_minimal.tar.zst"
PREFIX="${ROOT}/runtime/hora_torch_chunk_"
REMOTE="root@39.105.12.60"
PORT="6029"
REMOTE_RUNTIME="/workspace/Proknee-RL-muscle/torque_replay_training/runtime"
REMOTE_BUNDLE="${REMOTE_RUNTIME}/$(basename "${BUNDLE}")"

if [[ ! -f "${BUNDLE}" ]]; then
  echo "missing local HORA bundle: ${BUNDLE}" >&2
  exit 1
fi
EXPECTED="$(sha256sum "${BUNDLE}" | cut -d' ' -f1)"
if ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
  "test -f '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}' | cut -d' ' -f1)\" = '${EXPECTED}'"; then
  echo "remote HORA bundle already matches SHA-256 ${EXPECTED}"
  exit 0
fi

find "${ROOT}/runtime" -maxdepth 1 -type f -name 'hora_torch_chunk_*' -delete
split -b 80M -d -a 2 "${BUNDLE}" "${PREFIX}"
ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
  "mkdir -p '${REMOTE_RUNTIME}' && \
   find '${REMOTE_RUNTIME}' -maxdepth 1 -type f -name 'hora_torch_chunk_*' -delete"

PIDS=()
for chunk in "${PREFIX}"*; do
  scp -q -P "${PORT}" "${chunk}" "${REMOTE}:${REMOTE_RUNTIME}/" &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

ssh -o BatchMode=yes -p "${PORT}" "${REMOTE}" \
  "cd '/workspace/Proknee-RL-muscle' && .venv/bin/python - \
    '${REMOTE_RUNTIME}' '${REMOTE_BUNDLE}' '${EXPECTED}'" <<'PY'
import hashlib
from pathlib import Path
import sys

runtime = Path(sys.argv[1])
target = Path(sys.argv[2])
expected = sys.argv[3]
temporary = target.with_suffix(target.suffix + ".tmp")
chunks = sorted(runtime.glob("hora_torch_chunk_*"))
if not chunks:
    raise SystemExit("no HORA chunks were uploaded")
with temporary.open("wb") as output:
    for chunk in chunks:
        with chunk.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                output.write(block)
hasher = hashlib.sha256()
with temporary.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        hasher.update(block)
digest = hasher.hexdigest()
if digest != expected:
    raise SystemExit(f"HORA bundle SHA mismatch: {digest} != {expected}")
temporary.replace(target)
for chunk in chunks:
    chunk.unlink()
print(f"HORA bundle ready: {target} sha256={digest}")
PY
