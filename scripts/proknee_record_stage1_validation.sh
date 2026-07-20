#!/usr/bin/env bash
# Record visual audit (render overlay) + physical PD deployment for one Stage1 checkpoint.
# Produces main .mp4 and *_web.mp4 (unless --no-web-encode on the Python side).
#
# Usage:
#   bash musclemimic/scripts/proknee_record_stage1_validation.sh /path/to/stage1_best.pt tag_name
#
# Env:
#   MUJOCO_GL=egl  (recommended headless)
#   PROKNEE_RECORD_STEPS (default 400)
#   PROKNEE_RECORD_PD_EXTRA — optional extra args for physical pass only, e.g.:
#     export PROKNEE_RECORD_PD_EXTRA="--max-qpos-target-step 0.02 --pd-gain-scale 0.7 --residual-filter 0.92"
#   PROKNEE_RECORD_KIND — stage1 (default) or stage2 (passes --kind stage2 to record script)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJ="$ROOT/musclemimic"
MM="${MM_CKPT:-$ROOT/data/checkpoints/mm-10m-2}"
POLICY="${1:?policy .pt path}"
TAG="${2:-round}"
STEPS="${PROKNEE_RECORD_STEPS:-400}"
OUT="$ROOT/videos/proknee_${TAG}"
mkdir -p "$OUT"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

KIND="${PROKNEE_RECORD_KIND:-stage1}"
REC=(env PYTHONUNBUFFERED=1 uv run --project "$PROJ" python -u "$ROOT/musclemimic/scripts/record_stage1_testset_visual.py"
  --checkpoint "$MM"
  --policy "$POLICY"
  --kind "$KIND"
  --num-motions 3
  --steps-per-motion "$STEPS"
  --prosthesis-muscle-scale "${PROKNEE_RECORD_MUSCLE_SCALE:-1.0}"
)

if [[ -n "${PROKNEE_RECORD_PD_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  PD_EXTRA=(${PROKNEE_RECORD_PD_EXTRA})
else
  PD_EXTRA=()
fi

echo "[record] visual audit -> $OUT/${TAG}_audit.mp4"
"${REC[@]}" --record-path "$OUT/${TAG}_audit.mp4"

echo "[record] physical PD -> $OUT/${TAG}_physical.mp4"
"${REC[@]}" "${PD_EXTRA[@]}" --pd-override --record-path "$OUT/${TAG}_physical.mp4"

echo "[record] done under $OUT"
