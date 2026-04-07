#!/usr/bin/env bash
# Stage0 HumanMimic：在 Isaac Gym 窗口中播放检查点（train_humanmimic_unified.py, test=True）。
#
# 用法（在仓库根目录）:
#   source scripts/activate_proknee_tc_env.sh
#   bash scripts/play_stage0_unified_humanmimic.sh
#   bash scripts/play_stage0_unified_humanmimic.sh runs/HumanoidAMPUnifiedHumanMimic_xxx/nn/xxx.pth
#
# 未传 checkpoint 时：若存在下方「默认权重」则优先用它，否则取 runs/ 下最新 HumanMimic .pth。
# 覆盖默认：export HUMANMIMIC_PLAY_CHECKPOINT=/绝对路径/xxx.pth
#
# 换速版权重：与训练时一致，由 HumanoidAMPUnifiedHumanMimic_phase1.yaml 的 enableCmdSwitch 等控制；
# 一般无需改脚本；若需与训练时换速概率不一致，可在命令后追加 Hydra，例如:
#   task.env.cmdSwitchProb=0.02 task.env.cmdSwitchInterval=50
#
# 可选：NUM_ENVS=1 bash scripts/play_stage0_unified_humanmimic.sh   # 显存紧
#
# 依赖：与 train_stage0_unified_humanmimic.sh 相同；需本机图形界面（headless=False）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IGE_DIR="$PROJECT_DIR/IsaacGymEnvs/isaacgymenvs"
# 默认播放（换速 fine-tune 后权重，可 export HUMANMIMIC_PLAY_CHECKPOINT 覆盖）
_DEFAULT_REL="runs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth"
DEFAULT_HUMANMIMIC_CKPT="${HUMANMIMIC_PLAY_CHECKPOINT:-$IGE_DIR/$_DEFAULT_REL}"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/activate_proknee_tc_env.sh"

cd "$IGE_DIR"
PYTHON="${CONDA_PREFIX}/bin/python"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

NUM_ENVS="${NUM_ENVS:-4}"

CHECKPOINT="${1:-}"
if [ -n "$CHECKPOINT" ] && [[ "$CHECKPOINT" != /* ]]; then
  if [ -f "$PROJECT_DIR/$CHECKPOINT" ]; then
    CHECKPOINT="$(realpath "$PROJECT_DIR/$CHECKPOINT")"
  elif [ -f "$IGE_DIR/$CHECKPOINT" ]; then
    CHECKPOINT="$(realpath "$IGE_DIR/$CHECKPOINT")"
  fi
fi

if [ -z "$CHECKPOINT" ]; then
  if [ -f "$DEFAULT_HUMANMIMIC_CKPT" ]; then
    CHECKPOINT="$(realpath "$DEFAULT_HUMANMIMIC_CKPT")"
    echo "[i] 使用默认 HumanMimic checkpoint: $CHECKPOINT"
  else
    CHECKPOINT=$(ls -t runs/HumanoidAMPUnifiedHumanMimic_*/nn/*.pth 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT" ]; then
      CHECKPOINT=$(ls -t runs/HumanoidAMPUnifiedHumanMimic*.pth 2>/dev/null | head -1 || true)
    fi
    if [ -z "$CHECKPOINT" ]; then
      echo "[!] 未找到 checkpoint。请传入，或设置 HUMANMIMIC_PLAY_CHECKPOINT，例如:"
      echo "    bash scripts/play_stage0_unified_humanmimic.sh IsaacGymEnvs/isaacgymenvs/runs/.../nn/xxx.pth"
      exit 1
    fi
    echo "[i] 使用 runs/ 下最新 checkpoint: $CHECKPOINT"
  fi
else
  shift
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Stage0 HumanMimic 播放 (test=True, headless=False)    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  task:   HumanoidAMPUnifiedHumanMimic_phase1            ║"
echo "║  train:  HumanoidAMPUnifiedHumanMimicPPO               ║"
echo "║  num_envs: $NUM_ENVS"
echo "║  checkpoint: $CHECKPOINT"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

$PYTHON train_humanmimic_unified.py \
  task=HumanoidAMPUnifiedHumanMimic_phase1 \
  train=HumanoidAMPUnifiedHumanMimicPPO \
  test=True \
  headless=False \
  "num_envs=$NUM_ENVS" \
  "checkpoint=$CHECKPOINT" \
  +train.params.config.torch_compile=False \
  "$@"
