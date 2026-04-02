#!/usr/bin/env bash
# 激活用于 Isaac Gym + IsaacGymEnvs 训练的 conda 环境（Stage0 Unified / HumanMimic 等）。
# 默认：conda 环境名 proknee_tc（远程与本项目约定）。
# 若需使用其它环境：export CONDA_ENV_NAME=你的环境名
# 必须 source 使用，否则 conda activate 不会作用于当前 shell。
#
#   source /path/to/RLleg/scripts/activate_proknee_tc_env.sh
#
# 可选：export MINICONDA=/path/to/miniconda3

_RL_SCR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

if [ -f "$MINICONDA/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$MINICONDA/etc/profile.d/conda.sh"
  _CE="${CONDA_ENV_NAME:-proknee_tc}"
  conda activate "$_CE"
else
  echo "activate_proknee_tc_env.sh: 未找到 $MINICONDA/etc/profile.d/conda.sh，请先安装 Miniconda 或设置 MINICONDA" >&2
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
