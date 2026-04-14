#!/usr/bin/env bash
# Unified Stage2 交互播放（键盘调速）。依赖：本机图形界面、conda proknee_tc、在仓库根执行。
#
# 用法（在仓库根 RLleg/）:
#   bash scripts/play_stage2_unified.sh
#   STAGE2_CKPT=/path/to/best.pth STAGE0_BODY=/path/to/stage0.pth bash scripts/play_stage2_unified.sh
#
# 默认 Stage2: outputs/humanmimic_policy_knee_ankle_vel/stage2_unified/checkpoints/best.pth
# 默认 Stage0: outputs/humanmimic_policy_knee_ankle_vel/stage0_unified/HumanoidAMPUnifiedHumanMimic_13-14-45-32_16000.pth
# 若默认不存在，会依次尝试 outputs/checkpoints/...（与旧文档一致）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "$PROJECT_DIR"
PYTHON="${CONDA_PREFIX}/bin/python"

DEF_S2="$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage2_unified/checkpoints/best.pth"
ALT_S2="$PROJECT_DIR/outputs/checkpoints/stage2_unified/best.pth"
DEF_S0="$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage0_unified/HumanoidAMPUnifiedHumanMimic_13-14-45-32_16000.pth"
ALT_S0C="$PROJECT_DIR/outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth"
ALT_S0="$PROJECT_DIR/outputs/humanmimic_policy_knee_ankle_vel/stage0_unified/body_policy.pth"
ALT_S0B="$PROJECT_DIR/outputs/checkpoints/stage0/stage0_unified_1800.pth"

STAGE2_CKPT="${STAGE2_CKPT:-}"
if [ -z "$STAGE2_CKPT" ]; then
  if [ -f "$DEF_S2" ]; then STAGE2_CKPT="$DEF_S2"
  elif [ -f "$ALT_S2" ]; then STAGE2_CKPT="$ALT_S2"
  else STAGE2_CKPT="$DEF_S2"; fi
fi
STAGE0_BODY="${STAGE0_BODY:-}"
if [ -z "$STAGE0_BODY" ]; then
  if [ -f "$DEF_S0" ]; then STAGE0_BODY="$DEF_S0"
  elif [ -f "$ALT_S0" ]; then STAGE0_BODY="$ALT_S0"
  elif [ -f "$ALT_S0B" ]; then STAGE0_BODY="$ALT_S0B"
  elif [ -f "$ALT_S0C" ]; then STAGE0_BODY="$ALT_S0C"
  else STAGE0_BODY="$DEF_S0"; fi
fi

STAGE2_CKPT="$(realpath "$STAGE2_CKPT" 2>/dev/null || echo "$STAGE2_CKPT")"
STAGE0_BODY="$(realpath "$STAGE0_BODY" 2>/dev/null || echo "$STAGE0_BODY")"

if [ ! -f "$STAGE2_CKPT" ]; then
  echo "[!] 未找到 Stage2 检查点: $STAGE2_CKPT" >&2
  echo "    请设置 STAGE2_CKPT=... 或训练后确认 outputs/.../stage2_unified/checkpoints/best.pth" >&2
  exit 1
fi
if [ ! -f "$STAGE0_BODY" ]; then
  echo "[!] 未找到 Stage0 全身策略: $STAGE0_BODY" >&2
  echo "    请设置 STAGE0_BODY=...（须与训练 Stage1/2 时用的 body 一致，rl_games 导出的 .pth）" >&2
  exit 1
fi

echo "Stage2: $STAGE2_CKPT"
echo "Stage0: $STAGE0_BODY"
exec "$PYTHON" "$PROJECT_DIR/scripts/interactive_unified.py" --device "${DEVICE:-cuda:0}" \
  --checkpoint "$STAGE2_CKPT" \
  --body-policy "$STAGE0_BODY" \
  "$@"
