#!/usr/bin/env bash
# 从「不换速」阶段保存的 HumanMimic Stage0 检查点续训，强制使用「回合内换速」配置（fine-tune）。
#
# 自动检测：在 IsaacGymEnvs/isaacgymenvs/runs/HumanoidAMPUnifiedHumanMimic_*/nn/ 下按修改时间取最新 *.pth
# （通常为远程默认产出路径；若已有更新的换速 run，请显式传入旧 checkpoint 路径）。
#
# 用法（仓库根目录）:
#   bash scripts/train_stage0_unified_humanmimic_finetune_switch.sh
#   bash scripts/train_stage0_unified_humanmimic_finetune_switch.sh 16000
#   bash scripts/train_stage0_unified_humanmimic_finetune_switch.sh 16000 runs/HumanoidAMPUnifiedHumanMimic_xxx/nn/xxx.pth
#   HUMANMIMIC_BASE_CKPT=/绝对路径/xxx.pth bash scripts/train_stage0_unified_humanmimic_finetune_switch.sh 16000
#
# 重要：续训时 max_iterations（= max_epochs）必须大于检查点里已训练完的 epoch，否则会立刻打印 MAX EPOCHS NUM! 退出。
#   例如基座已训满 8000，请设 12000、16000 等（默认 16000）。
#
# 可选：本机固定默认基座（无第二参数、未设 HUMANMIMIC_BASE_CKPT 时优先于「runs 最新」）
#   export HUMANMIMIC_FINETUNE_DEFAULT_CKPT=IsaacGymEnvs/isaacgymenvs/runs/.../nn/xxx.pth
#
# 显存: export STAGE0_NUM_ENVS=2048 或 1024
# 环境: source 同 train_stage0_unified_humanmimic.sh（本脚本已 activate_proknee_tc_env.sh）
#
# 与 train_stage0_unified_humanmimic.sh 一样在前台看实时输出（勿用 nohup/重定向到文件）:
#   cd <仓库根> && PYTHONUNBUFFERED=1 bash scripts/train_stage0_unified_humanmimic_finetune_switch.sh 16000 <checkpoint路径>
# 同时落盘:  ... 2>&1 | tee outputs/stage0_humanmimic_finetune_switch.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IGE_DIR="$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"
STAGE0_TRAIN_DIR="${STAGE0_TRAIN_DIR:-$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage0_unified}"
mkdir -p "$STAGE0_TRAIN_DIR"
STAGE0_TRAIN_DIR_ABS="$(realpath "$STAGE0_TRAIN_DIR")"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"

cd "$IGE_DIR"
PYTHON="${CONDA_PREFIX}/bin/python"
NUM_ENVS="${STAGE0_NUM_ENVS:-4096}"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/stage0_batch_hydra.sh"

MAX_ITERS=${1:-16000}
MANUAL_CKPT="${2:-}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

resolve_ckpt() {
  local c="$1"
  if [ -z "$c" ]; then
    return 1
  fi
  if [[ "$c" == /* ]] && [ -f "$c" ]; then
    echo "$c"
    return 0
  fi
  if [ -f "$PROJECT_DIR/$c" ]; then
    realpath "$PROJECT_DIR/$c"
    return 0
  fi
  if [ -f "$IGE_DIR/$c" ]; then
    realpath "$IGE_DIR/$c"
    return 0
  fi
  return 1
}

if [ -n "${HUMANMIMIC_BASE_CKPT:-}" ]; then
  CHECKPOINT=$(resolve_ckpt "$HUMANMIMIC_BASE_CKPT") || {
    echo "[!] HUMANMIMIC_BASE_CKPT 指向的文件不存在: $HUMANMIMIC_BASE_CKPT"
    exit 1
  }
elif [ -n "$MANUAL_CKPT" ]; then
  CHECKPOINT=$(resolve_ckpt "$MANUAL_CKPT") || {
    echo "[!] 找不到检查点: $MANUAL_CKPT（相对仓库根或 isaacgymenvs）"
    exit 1
  }
elif [ -n "${HUMANMIMIC_FINETUNE_DEFAULT_CKPT:-}" ]; then
  CHECKPOINT=$(resolve_ckpt "$HUMANMIMIC_FINETUNE_DEFAULT_CKPT") || {
    echo "[!] HUMANMIMIC_FINETUNE_DEFAULT_CKPT 指向的文件不存在: $HUMANMIMIC_FINETUNE_DEFAULT_CKPT"
    exit 1
  }
else
  CHECKPOINT=$(ls -t "$STAGE0_TRAIN_DIR_ABS"/HumanoidAMPUnifiedHumanMimic_*/nn/*.pth 2>/dev/null | head -1 || true)
  if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t "$IGE_DIR"/runs/HumanoidAMPUnifiedHumanMimic_*/nn/*.pth 2>/dev/null | head -1 || true)
  fi
  if [ -z "$CHECKPOINT" ]; then
    echo "[!] 未找到 HumanMimic Stage0 .pth（已查: $STAGE0_TRAIN_DIR_ABS 与 $IGE_DIR/runs）。"
    echo "    请拷贝检查点到 outputs/humanmimic_policy_knee_ankle_vel/stage0_unified/.../nn/ 或:"
    echo "    bash $0 $MAX_ITERS <runs/.../nn/xxx.pth>"
    echo "    或: HUMANMIMIC_BASE_CKPT=/path/to/xxx.pth $0 $MAX_ITERS"
    exit 1
  fi
  CHECKPOINT=$(realpath "$CHECKPOINT")
fi

# 与 HumanoidAMPUnifiedHumanMimic_phase1.yaml 中换速版一致；显式覆盖以便远程未 pull YAML 时仍生效
SWITCH_OVERRIDES=(
  task.env.enableCmdSwitch=True
  task.env.cmdSwitchProb=0.02
  task.env.cmdSwitchInterval=50
)

TB_PORT="${STAGE0_TENSORBOARD_PORT:-6006}"
TB_CMD="${CONDA_PREFIX}/bin/tensorboard"
if [ ! -x "$TB_CMD" ]; then
  TB_CMD="tensorboard"
fi
if [ "${STAGE0_LAUNCH_TENSORBOARD:-0}" = "1" ]; then
  "$TB_CMD" --logdir="$STAGE0_TRAIN_DIR_ABS" --port="$TB_PORT" --bind_all >/dev/null 2>&1 &
  echo "[TensorBoard] 后台: http://127.0.0.1:${TB_PORT}/  logdir=$STAGE0_TRAIN_DIR_ABS"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Stage0 HumanMimic — Fine-tune（换速开关 + 续训）          ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  开始时间:       $TIMESTAMP"
echo "║  max_iterations: $MAX_ITERS   num_envs: $NUM_ENVS"
echo "║  minibatch:      $STAGE0_MB   amp_minibatch: $STAGE0_AMP_MB"
echo "║  基座检查点:     $CHECKPOINT"
echo "║  Hydra 换速:     ${SWITCH_OVERRIDES[*]}"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  train_dir:    $STAGE0_TRAIN_DIR_ABS"
echo "║  TensorBoard: $TB_CMD --logdir=$STAGE0_TRAIN_DIR_ABS --port=$TB_PORT"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

exec "$PYTHON" train_humanmimic_unified.py \
  task=HumanoidAMPUnifiedHumanMimic_phase1 \
  train=HumanoidAMPUnifiedHumanMimicPPO \
  num_envs=$NUM_ENVS \
  train.params.config.minibatch_size=$STAGE0_MB \
  train.params.config.amp_minibatch_size=$STAGE0_AMP_MB \
  train.params.config.train_dir="$STAGE0_TRAIN_DIR_ABS" \
  max_iterations=$MAX_ITERS \
  headless=True \
  "checkpoint=$CHECKPOINT" \
  "${SWITCH_OVERRIDES[@]}" \
  "$@"
