#!/usr/bin/env bash
# 激活用于 Isaac Gym + IsaacGymEnvs 训练的 conda 环境（Stage0 Unified / HumanMimic 等）。
# 默认：conda 环境名 proknee_tc（远程与本项目约定）。
# 若需使用其它环境：export CONDA_ENV_NAME=你的环境名
# 必须 source 使用，否则 conda activate 不会作用于当前 shell。
#
#   source /path/to/RLleg/scripts/activate_proknee_tc_env.sh
#
# 可选：export MINICONDA=/path/to/miniconda3  或  export CONDA_ROOT=/path/to/anaconda3

_RL_SCR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_CONDA_SH=""
if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  _CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -n "${MINICONDA:-}" ] && [ -f "${MINICONDA}/etc/profile.d/conda.sh" ]; then
  _CONDA_SH="${MINICONDA}/etc/profile.d/conda.sh"
else
  for _root in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
    if [ -f "$_root/etc/profile.d/conda.sh" ]; then
      _CONDA_SH="$_root/etc/profile.d/conda.sh"
      break
    fi
  done
fi

if [ -n "$_CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$_CONDA_SH"
  _CE="${CONDA_ENV_NAME:-proknee_tc}"
  conda activate "$_CE"
else
  echo "activate_proknee_tc_env.sh: 未找到 conda.sh（试过 CONDA_ROOT、MINICONDA、~/anaconda3、~/miniconda3）。请安装 Conda 或: export CONDA_ROOT=/你的/anaconda3" >&2
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
