# A100 GPU-5 完整训练、监控与 TensorBoard

## 1. 固定服务器、分支和路径

```sshconfig
Host A100
  HostName 39.105.12.60
  User root
  Port 6029
```

- 远端项目：`/workspace/Proknee-RL-muscle`
- 本机项目：`/home/user/Workspace/Proknee-RL-muscle`
- GitHub：`gracetata/Proknee-RL`
- 共同分支：`muscle`
- 远端环境：`/workspace/Proknee-RL-muscle/.venv`
- tracker checkpoint：`data/checkpoints/mm-10m-2`
- 正式数据：`torque_replay_training/data/fullbody_v1`
- 正式训练：`torque_replay_training/outputs/a100_train_v1`

代码通过 Git 同步；checkpoint、GMR cache、导出数据和训练输出是大文件，不提交 Git，
使用校验过的 `rsync/scp` 单独同步。

## 2. GPU 安全规则

当前阶段只使用物理 GPU 5，不访问 GPU 0–4、6、7。正式训练为一个 seed：

| 训练 | 物理 GPU | 进程内逻辑 GPU |
|---|---:|---:|
| seed 0 | 5 | 0 |

每次启动前必须执行：

```bash
cd /workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5
```

检查同时覆盖计算进程、显存和利用率。默认显存不超过 1024 MiB、利用率不超过 10%，
且不能存在计算进程。任一条件不满足时返回非零，训练只等待，不杀进程、不抢占、不改用
其他卡。真正启动前会在文件锁内再次检查，避免重复启动。

当前训练入口只接受物理 GPU 5：

```bash
torque_replay_training/scripts/a100_exec_gpu.sh 5 COMMAND [ARG ...]
```

## 3. 环境

当前部署使用 Python 3.11.15、JAX 0.7.2 CUDA、MuJoCo 3.4.0：

```bash
cd /workspace/Proknee-RL-muscle
PYTHON=/workspace/.tools/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
"${PYTHON}" -m venv .venv

PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  .venv/bin/python -m pip install -e '.[dev]'

PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  .venv/bin/python -m pip install 'jax[cuda12]==0.7.2'
```

不要使用系统 Python 3.10。镜像缺少 `tfp-nightly` 时从官方 PyPI 补充，不能删除依赖。

## 4. 正式数据与完整训练

正式数据固定包含直行、慢走、右转、左转四条完整 motion：

```text
KIT/314/walking_medium09_poses
KIT/425/walking_slow07_poses
KIT/348/turn_right03_poses
KIT/167/turn_left05_poses
```

安全生产流水线会：检查 GPU 5 → 在 GPU 5 导出并验证四条完整轨迹 → 再次检查 GPU 5 →
在 GPU 5 训练 seed 0。正式数据不允许 `--allow-incomplete`。

手动前台执行：

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/run_production_pipeline_a100.sh
```

推荐使用排队守护任务。它每 120 秒检查一次，GPU 5 忙时只记录状态；空闲后自动启动：

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/start_a100_watchdog.sh
tmux ls
```

相关 tmux 会话：

- `proknee-a100-watchdog`：周期检查任务；
- `proknee-a100-train`：数据生产和三卡训练流水线；
- `proknee-tensorboard-6011`：TensorBoard。

训练配置为 `configs/train.yaml`：seed 0 训练 200,000 environment steps：

```text
outputs/a100_train_v1/seed_0
```

## 5. 持续监控

查看一次结构化状态：

```bash
cd /workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/a100_training_status.py
```

查看流水线和各 seed 日志：

```bash
tail -f torque_replay_training/runtime/pipeline.log
tail -f torque_replay_training/outputs/a100_train_v1/seed_0/train.log
```

状态必须综合检查：GPU 5、唯一训练 PID、seed 0 的 `metrics.jsonl`、
最新 `policy_*.msgpack`、tmux 会话和 TensorBoard HTTP 状态，不能只看终端是否有输出。

## 6. TensorBoard

远端 TensorBoard 使用独立端口 6011，避免影响服务器已有的 6006 服务：

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/start_tensorboard_a100.sh
curl -I http://127.0.0.1:6011/
```

在本机建立 SSH 隧道：

```bash
ssh -N -L 6011:127.0.0.1:6011 -p 6029 root@39.105.12.60
```

然后只在本机浏览器打开 `http://127.0.0.1:6011`。A100 上不启动 MuJoCo GUI。

## 7. 代码同步规范

本机开发完成后：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
git status --short
git add <明确文件>
git commit -m "..."
git push origin muscle
```

A100 必须快进到同一提交后才能开始新训练：

```bash
cd /workspace/Proknee-RL-muscle
git fetch proknee muscle
git merge --ff-only proknee/muscle
git rev-parse HEAD
```

比较本机、GitHub、A100 三个 SHA，必须完全一致。远端 GitHub DNS 不可用时，从本机
生成 Git bundle 传到服务器并快进，不能只复制文件而让 Git 历史不一致。

## 8. Smoke 与边界

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/run_smoke_a100.sh
```

Smoke 验证数据导出、广义力回放、短 PPO、TensorBoard event、保存和重载链路；它不是
完整行走/转弯训练的性能结论。

## 9. 2026-07-22 部署状态

- 功能实现与测试基线提交为 `ed3a102407537fa9e78eabf2b8c67161935058f5`；
- 本机和 A100 单元测试均为 `6 passed`；
- 本机完整 smoke、TensorBoard event 读取和可视化 `--check-only` 均通过；
- 四条正式 GMR cache 已传到 A100，并逐文件与本机 SHA-256 一致；
- `proknee-a100-watchdog` 已启动，每 120 秒检查一次；
- `proknee-tensorboard-6011` 已启动，HTTP 状态为 200；
- Codex 线程监控任务 `monitor-proknee-a100-training` 每 10 分钟只读检查一次；
- 2026-07-22 用户将当前训练安排改为只使用 GPU 5；GPU 6、7 不再影响启动条件，也不会
  被本项目访问。

正式 cache 校验值：

```text
8f320295504ffc0a7759ba76ee454dd2b1e3f1feb7244d476b2373100235cc9b  KIT/314/walking_medium09_poses.npz
29346e54820c2fc6fbe65a47da77e9974f1fe671798d5eb31196649daa615530  KIT/425/walking_slow07_poses.npz
99255a4e15866ce81b3a5fe6d7b9a5de9138905697170241bebccdb1bb84399d  KIT/348/turn_right03_poses.npz
c874464595c76e124152a0d603db14e4dc5fd1430cb3e353b8b5b4f545f7e831  KIT/167/turn_left05_poses.npz
```
