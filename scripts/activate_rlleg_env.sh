#!/usr/bin/env bash
# 激活用于 Isaac Gym + IsaacGymEnvs 训练的 conda 环境（rlleg：Python 3.8 + PyTorch cu124 + 可编辑安装的 isaacgym）。
# 必须 source 使用，否则 conda activate 不会作用于当前 shell。
#
#   source /path/to/RLleg/scripts/activate_rlleg_env.sh
#
# 可选：export MINICONDA=/path/to/miniconda3

_RL_SCR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_RLLEG_ROOT="$(dirname "$_RL_SCR")"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

if [ -f "$MINICONDA/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$MINICONDA/etc/profile.d/conda.sh"
  conda activate rlleg
else
  echo "activate_rlleg_env.sh: 未找到 $MINICONDA/etc/profile.d/conda.sh，请先安装 Miniconda 或设置 MINICONDA" >&2
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# 使用 pip install -e IsaacGymEnvs 后一般无需 PYTHONPATH；若设置可能导致 Hydra 将 cfg 解析为包路径失败
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
