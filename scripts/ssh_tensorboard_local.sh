#!/usr/bin/env bash
# 在你自己的笔记本电脑上运行（不是训练服务器上）
# 作用：把远程 6006 端口转发到本机，便于浏览器打开 TensorBoard
#
# 用法:
#   bash ssh_tensorboard_local.sh
#   bash ssh_tensorboard_local.sh user@192.168.50.55
#
# 然后在浏览器打开: http://127.0.0.1:6006

REMOTE="${1:-user@$(hostname -I 2>/dev/null | awk '{print $1}')}"
PORT="${TENSORBOARD_PORT:-6006}"

echo "本机执行 SSH 端口转发:"
echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${REMOTE}"
echo ""
echo "转发成功后，浏览器打开: http://127.0.0.1:${PORT}"
echo "（保持该终端窗口不要关）"
exec ssh -N -L "${PORT}:127.0.0.1:${PORT}" "$REMOTE"
