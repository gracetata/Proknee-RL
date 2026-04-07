#!/usr/bin/env bash
# HumanMimic-style Stage0: velocity grid 0~3 m/s, dual AMP heads (imitation + style), gap velocity blending.
# Uses train_humanmimic_unified.py.
#
# Usage:
#   bash scripts/train_stage0_unified_humanmimic_phase1.sh [MAX_ITERS] [CHECKPOINT]
# TensorBoard：训练时 rl_games 已写 runs/<run>/summaries/；另开终端执行脚本打印的 tensorboard 命令即可。
# 若提示无 tensorboard：pip install tensorboard
# 可选：export STAGE0_LAUNCH_TENSORBOARD=1  训练同时后台打开 TensorBoard
#
# 默认 num_envs=4096（RTX 4090 24GB 左右）；显存不够: export STAGE0_NUM_ENVS=2048 或 1024
# 需 conda 环境 proknee_tc（见 docs/ENVIRONMENT_RLLEG.md）；脚本内已 source activate_proknee_tc_env.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
# 与 Stage1/Stage2 同一 outputs 树：rl_games 写入 <STAGE0_TRAIN_DIR>/HumanoidAMPUnifiedHumanMimic_*/nn/*.pth
STAGE0_TRAIN_DIR="${STAGE0_TRAIN_DIR:-$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage0_unified}"
mkdir -p "$STAGE0_TRAIN_DIR"
STAGE0_TRAIN_DIR_ABS="$(realpath "$STAGE0_TRAIN_DIR")"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"

cd "$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"
PYTHON="${CONDA_PREFIX}/bin/python"
NUM_ENVS="${STAGE0_NUM_ENVS:-4096}"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/stage0_batch_hydra.sh"

MAX_ITERS=${1:-10000}
CHECKPOINT=${2:-""}
shift 2 2>/dev/null || shift $# 2>/dev/null || true

CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ]; then
  CHECKPOINT_ARG="checkpoint=$CHECKPOINT"
fi

# TensorBoard：rl_games 默认写入 runs/<实验目录>/summaries/（无需额外 Python 配置）
# 另开终端观测：tensorboard --logdir=.../runs --port=6006
# 可选后台：export STAGE0_LAUNCH_TENSORBOARD=1  （端口 STAGE0_TENSORBOARD_PORT，默认 6006）
TB_PORT="${STAGE0_TENSORBOARD_PORT:-6006}"
TB_CMD="${CONDA_PREFIX}/bin/tensorboard"
if [ ! -x "$TB_CMD" ]; then
  TB_CMD="tensorboard"
fi
if [ "${STAGE0_LAUNCH_TENSORBOARD:-0}" = "1" ]; then
  "$TB_CMD" --logdir="$STAGE0_TRAIN_DIR_ABS" --port="$TB_PORT" --bind_all >/dev/null 2>&1 &
  echo "[TensorBoard] 后台已启动: http://127.0.0.1:${TB_PORT}/  logdir=$STAGE0_TRAIN_DIR_ABS"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Stage0 HumanMimic — HumanoidAMPUnifiedHumanMimic phase1 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  开始时间:    $TIMESTAMP"
echo "║  max_iterations: $MAX_ITERS   num_envs: $NUM_ENVS"
echo "║  minibatch: $STAGE0_MB   amp_minibatch: $STAGE0_AMP_MB"
if [ -n "$CHECKPOINT" ]; then
echo "║  恢复:        $CHECKPOINT"
fi
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  train_dir（检查点根）: $STAGE0_TRAIN_DIR_ABS"
echo "║  TensorBoard 日志: <train_dir>/<run_name>/summaries/"
echo "║  观测（另开终端，工作目录同下）:"
echo "║    $TB_CMD --logdir=$STAGE0_TRAIN_DIR_ABS --port=$TB_PORT"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

$PYTHON train_humanmimic_unified.py \
  task=HumanoidAMPUnifiedHumanMimic_phase1 \
  train=HumanoidAMPUnifiedHumanMimicPPO \
  num_envs=$NUM_ENVS \
  train.params.config.minibatch_size=$STAGE0_MB \
  train.params.config.amp_minibatch_size=$STAGE0_AMP_MB \
  train.params.config.train_dir="$STAGE0_TRAIN_DIR_ABS" \
  max_iterations=$MAX_ITERS \
  headless=True \
  $CHECKPOINT_ARG \
  "$@"
