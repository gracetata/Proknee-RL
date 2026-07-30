# A100：MJLAB 正式训练

当前远端路径 `/workspace/Proknee-RL-muscle`，GitHub 分支 `muscle`。只允许物理
GPU 5；禁止 GPU 0–4、6、7。每次 GPU 命令前都必须重新执行 guard；占用时等待，不得
杀进程或抢卡。

## 同步与安装

```bash
ssh -p 6029 root@39.105.12.60
cd /workspace/Proknee-RL-muscle
git fetch origin muscle
git switch muscle
git pull --ff-only origin muscle
bash torque_replay_training/scripts/setup_mjlab_env_a100.sh
```

安装脚本以写时复制复用服务器现有、已验证的 PyTorch `2.7.1+cu126`，然后在独立
`.venv-mjlab` 中安装 MJLAB/MuJoCo/MJWarp 固定版本；不会修改 `.venv-hora`。

## 启动

```bash
cd /workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5
bash torque_replay_training/scripts/run_mjlab_smoke_a100.sh

.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5
bash torque_replay_training/scripts/start_tensorboard_a100.sh
bash torque_replay_training/scripts/start_mjlab_training_a100.sh
```

smoke 先完成 assisted-replay gate，再用 4096 个环境跑一个官方 PPO iteration。正式
训练为 4096 environments、3000 iterations、seed 0。

## 状态

```bash
.venv-mjlab/bin/python torque_replay_training/scripts/mjlab_training_status.py
nvidia-smi -i 5
tmux capture-pane -pt proknee-a100-mjlab -S -100
tail -f torque_replay_training/runtime/a100_mjlab.launch.log
curl -I http://127.0.0.1:6011/
```

本机访问 TensorBoard：

```bash
ssh -N -L 6011:127.0.0.1:6011 -p 6029 root@39.105.12.60
```

浏览器打开 `http://127.0.0.1:6011`。

完整测试、checkpoint 评估和可视化命令见 [COMMANDS.md](COMMANDS.md)。
