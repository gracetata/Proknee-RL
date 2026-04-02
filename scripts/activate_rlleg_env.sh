#!/usr/bin/env bash
# 兼容旧名：等同于 activate_proknee_tc_env.sh（默认 proknee_tc）。
# shellcheck source=/dev/null
_REL="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$_REL/activate_proknee_tc_env.sh"
