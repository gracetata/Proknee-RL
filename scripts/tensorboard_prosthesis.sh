#!/usr/bin/env bash
# MuscleMimic 假肢训练 TensorBoard 启动脚本
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="${ROOT}/outputs/2026-05-18/19-16-20/tensorboard"
PORT="${TENSORBOARD_PORT:-6006}"

if [[ ! -d "$LOGDIR" ]]; then
  echo "找不到日志目录: $LOGDIR"
  echo "请把 outputs/.../tensorboard 路径改成本次 run 的实际目录。"
  exit 1
fi

cd "$ROOT"
echo "TensorBoard logdir: $LOGDIR"
echo "端口: $PORT"
echo ""
echo "在 Cursor 里：打开底部 Ports → Forwarded Ports → 访问 http://127.0.0.1:$PORT"
echo "在本机浏览器（非 Cursor）：先 SSH 转发，见 scripts/ssh_tensorboard_local.sh"
exec uv run tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
