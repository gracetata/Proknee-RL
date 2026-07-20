#!/usr/bin/env bash
# Resume split-action experiment sweep from the first incomplete step.
set -euo pipefail

ROOT="/home/user/Workspace/musclemimic/musclemimic"
LOG_DIR="$ROOT/outputs/_nonformal_runs/split_action_sweep_resume_$(date +%Y%m%d_%H%M%S)"
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
REFPD_DATA="data/split_action_prosthesis_distill/replay_four_reference_pd_full"
REFPD_CKPT="outputs/split_action_prosthesis_distill/replay_four_refpd_60ep/latest/checkpoints/checkpoint_distilled"
EVAL_ROOT="outputs/eval_split_action_prosthesis/replay_four"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/pipeline.log"; }

build_motion_flags() {
  MOTION_FLAGS=()
  for m in "${MOTIONS[@]}"; do
    MOTION_FLAGS+=(--motion-path "$m")
  done
}

latest_ckpt_for() {
  local name="$1"
  uv run python - <<PY
from pathlib import Path
latest = Path("outputs/split_action_prosthesis_distill/${name}/latest")
run_dir = latest.resolve() if latest.is_symlink() else latest
txt = run_dir / "checkpoint_path.txt"
print(txt.read_text().strip() if txt.exists() else "")
PY
}

replay_if_missing() {
  local name="$1"
  local eval_name="split_action_${name}"
  if [[ -f "$EVAL_ROOT/$eval_name/summary.json" ]]; then
    log "SKIP replay $name (summary exists)"
    return 0
  fi
  local ckpt
  ckpt=$(latest_ckpt_for "$name")
  log "REPLAY $name -> $ckpt"
  build_motion_flags
  uv run python scripts/record_split_action_prosthesis_replay.py \
    --checkpoint "$ckpt" \
    "${MOTION_FLAGS[@]}" \
    --run-name "$eval_name" \
    --output-dir "$EVAL_ROOT" \
    2>&1 | tee "$LOG_DIR/replay_${name}.log"
}

train_if_missing() {
  local name="$1"; shift
  if [[ -f "outputs/split_action_prosthesis_distill/${name}/latest/checkpoint_path.txt" ]] \
     || [[ -L "outputs/split_action_prosthesis_distill/${name}/latest" ]]; then
    log "SKIP train $name (checkpoint exists)"
    replay_if_missing "$name"
    return 0
  fi
  log "TRAIN $name"
  uv run python scripts/train_split_action_prosthesis_distillation.py "$@" \
    --output_dir "outputs/split_action_prosthesis_distill/${name}" \
    2>&1 | tee "$LOG_DIR/train_${name}.log"
  replay_if_missing "$name"
}

collect_if_missing() {
  local out_dir="$1"; shift
  local log_name="$1"; shift
  if [[ -f "$out_dir/summary.json" ]]; then
    log "SKIP collect $log_name"
    return 0
  fi
  log "COLLECT $log_name"
  "$@" 2>&1 | tee "$LOG_DIR/collect_${log_name}.log"
}

log "=== Resume sweep ==="
replay_if_missing "replay_four_refpd_lambda1_60ep"

train_if_missing "replay_four_refpd_lambda25_60ep" \
  --dataset_dir "$REFPD_DATA" --epochs 60 --lambda_prosthesis 25.0

collect_if_missing "data/split_action_prosthesis_distill/replay_four_reference_pd_pinfalse" pinfalse \
  uv run python scripts/collect_teacher_rollouts.py \
    --output_dir data/split_action_prosthesis_distill/replay_four_reference_pd_pinfalse \
    --motion_path "${MOTIONS[@]}" \
    --no-pin_student_state \
    --label_mode reference_pd

train_if_missing "replay_four_refpd_pinfalse_60ep" \
  --dataset_dir data/split_action_prosthesis_distill/replay_four_reference_pd_pinfalse \
  --epochs 60 --lambda_prosthesis 10.0

collect_if_missing "data/split_action_prosthesis_distill/dagger_replay_four_refpd" dagger \
  uv run python scripts/collect_split_action_prosthesis_dagger_rollouts.py \
    --output_dir data/split_action_prosthesis_distill/dagger_replay_four_refpd \
    --student_checkpoint "$REFPD_CKPT" \
    --motion_path "${MOTIONS[@]}" \
    --teacher_action_prob 0.3 \
    --label_mode reference_pd

train_if_missing "replay_four_refpd_dagger30ep" \
  --dataset_dir "$REFPD_DATA" \
  --extra_dataset_dir data/split_action_prosthesis_distill/dagger_replay_four_refpd \
  --init checkpoint \
  --init_checkpoint "$REFPD_CKPT" \
  --epochs 30 --lr 1e-4 --lambda_prosthesis 10.0

uv run python scripts/summarize_split_action_replay_four.py \
  --eval_root "$EVAL_ROOT" \
  --output "$LOG_DIR/replay_four_comparison.md" \
  2>&1 | tee "$LOG_DIR/summary.log"

log "DONE resume log=$LOG_DIR/pipeline.log"
