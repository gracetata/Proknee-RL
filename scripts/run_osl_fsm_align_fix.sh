#!/usr/bin/env bash
# Collect OSL-aligned data, fine-tune from DAgger1 anchor, replay replay_four.
set -euo pipefail

ROOT="/home/user/Workspace/musclemimic/musclemimic"
LOG_DIR="$ROOT/outputs/_nonformal_runs/osl_fsm_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
cd "$ROOT"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

MOTIONS=(
  "KIT/3/walk_6m_straight_line04_poses"
  "KIT/425/walking_slow07_poses"
  "KIT/359/walking_run04_poses"
  "KIT/9/WalkingStraightForwards07_poses"
)
ANCHOR_CKPT="outputs/full_muscle_masked_obs_dagger1_focused_round3_distill/latest/checkpoints/checkpoint_distilled"
TEACHER_DATA="data/full_muscle_masked_obs_rollouts/osl_align_replay_four_teacher"
DAGGER_DATA="data/full_muscle_masked_obs_dagger/osl_align_replay_four_dagger"
TRAIN_OUT="outputs/full_muscle_masked_obs_osl_align_30ep"
EVAL_OUT="outputs/_nonformal_runs/osl_fsm_fix/replay_four_eval"
BASELINE_EVAL="outputs/eval_masked_dagger_compare/replay_four/dagger1_focused_round3_osl_fsm"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/pipeline.log"; }

build_motion_flags() {
  MOTION_FLAGS=()
  for m in "${MOTIONS[@]}"; do
    MOTION_FLAGS+=(--motion-path "$m")
  done
}

log "=== Collect teacher rollouts (scale=0, OSL FSM) ==="
uv run python scripts/collect_masked_obs_full_muscle_rollouts.py \
  --output_dir "$TEACHER_DATA" \
  --motion_path "${MOTIONS[@]}" \
  --disabled-muscle-scale 0.0 \
  --use-osl-fsm \
  2>&1 | tee "$LOG_DIR/collect_teacher.log"

log "=== Collect DAgger rollouts (anchor student, scale=0, OSL FSM) ==="
uv run python scripts/collect_masked_obs_full_muscle_dagger_rollouts.py \
  --output_dir "$DAGGER_DATA" \
  --student_checkpoint "$ANCHOR_CKPT" \
  --motion_path "${MOTIONS[@]}" \
  --teacher_action_prob 0.3 \
  --disabled-muscle-scale 0.0 \
  --use-osl-fsm \
  2>&1 | tee "$LOG_DIR/collect_dagger.log"

log "=== Fine-tune from DAgger1 anchor ==="
uv run python scripts/train_masked_obs_full_muscle_distillation.py \
  --dataset_dir "$TEACHER_DATA" \
  --extra_dataset_dir "$DAGGER_DATA" \
  --init_checkpoint "$ANCHOR_CKPT" \
  --epochs 30 \
  --lr 5e-5 \
  --output_dir "$TRAIN_OUT" \
  2>&1 | tee "$LOG_DIR/train.log"

CKPT=$(uv run python - <<PY
from pathlib import Path
latest = Path("$TRAIN_OUT/latest")
run_dir = latest.resolve() if latest.is_symlink() else latest
print((run_dir / "checkpoint_path.txt").read_text().strip())
PY
)

log "=== Replay four motions (scale=0) ==="
build_motion_flags
for m in "${MOTIONS[@]}"; do
  uv run python scripts/record_masked_obs_full_muscle_osl_fsm_replay.py \
    --checkpoint "$CKPT" \
    --motion-path "$m" \
    --disabled-muscle-scale 0.0 \
    --output-dir "$EVAL_OUT" \
    2>&1 | tee -a "$LOG_DIR/replay.log"
done

log "=== Summary table ==="
uv run python scripts/summarize_osl_fsm_replay_four.py \
  --eval-root "$EVAL_OUT" \
  --baseline-root "$BASELINE_EVAL" \
  --output "$LOG_DIR/replay_four_comparison.md" \
  2>&1 | tee "$LOG_DIR/summary.log"

log "DONE log=$LOG_DIR/pipeline.log ckpt=$CKPT"
