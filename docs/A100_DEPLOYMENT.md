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
- 正式数据：`torque_replay_training/data/fullbody_v3`
- 正式训练：`torque_replay_training/outputs/a100_train_v3`

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

### nvitop GPU 监控

`nvitop` 使用独立环境 `/opt/nvitop`，不安装进项目 `.venv`。全局入口为
`/usr/local/bin/nvitop`，当前验证版本是 1.7.1：

```bash
PYTHON=/workspace/.tools/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
"${PYTHON}" -m venv /opt/nvitop
/opt/nvitop/bin/python -m pip install --upgrade nvitop
ln -sfn /opt/nvitop/bin/nvitop /usr/local/bin/nvitop
ln -sfn /opt/nvitop/bin/nvisel /usr/local/bin/nvisel
nvitop --version
```

共享服务器默认使用只读模式，避免误触 `T/K` 向他人的进程发送信号。只监控当前允许使用
的物理 GPU 5：

```bash
nvitop --readonly -o 5
```

只打印一次状态后退出：

```bash
nvitop -1 --readonly -o 5
```

## 4. 正式数据与完整训练

正式数据固定包含直行、慢走、右转、左转四条完整 motion：

```text
KIT/314/walking_medium09_poses
KIT/425/walking_slow07_poses
KIT/167/turn_right01_poses
KIT/167/turn_left01_poses
```

GMR trajectory 的长度是状态帧数；合法控制转移数是 `trajectory_length - 1`。生产资格检查
允许最后一个合法转移产生 `done`，但仍拒绝更早的终止、吸收状态、NaN/Inf 或少步数据。

安全生产流水线会：检查 GPU 5 → 在 GPU 5 导出并验证四条完整轨迹 → 再次检查 GPU 5 →
在 GPU 5 训练 seed 0。正式数据不允许 `--allow-incomplete`。

`fullbody_v3` 的物理边界与旧数据不同：

- 不在物理子步前插入额外 `mj_forward`，每次 `mj_step` 后记录该转移实际使用的力；
- `qfrc_actuator` 和 `rollout_qacc` 必须以 float64 保存；
- 任意帧 reset 都恢复前一物理子步的 `qacc_warmstart`；
- split 模式回放所有非假肢 DOF，包括数值上接近零的 free-root actuator 广义力；
- manifest 必须包含 `schema_version: 3`；
- 全长 all/split 和三个随机起点窗口必须全部通过。

旧 `fullbody_v1`、`fullbody_v2` 会因 schema 版本不符被拒绝，不能用于训练。

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
- `proknee-a100-train`：GPU 5 数据生产和单 seed 训练流水线；
- `proknee-tensorboard-6011`：TensorBoard。

训练配置为 `configs/train.yaml`：seed 0 训练 200,000 environment steps：

```text
outputs/a100_train_v3/seed_0
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
tail -f torque_replay_training/outputs/a100_train_v3/seed_0/train.log
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

## 9. 2026-07-23 全身回放修复状态

- 旧 v1 数据将关键广义力保存为 float32，四条轨迹在约 108–143 步后开始明显发散；
- v2 修复 float64 物理数据、随机 reset warmstart 和 split root DOF 覆盖；
- 本机单元测试为 `11 passed`，完整 smoke 通过；
- 本机 v2 四条完整轨迹的 all、split、三个随机起点窗口均为零误差；
- 本机 MuJoCo GUI 已实际启动并完成中速行走 100 步回放；
- A100 重新导出 v2 和恢复训练前仍必须检查物理 GPU 5；禁止使用 GPU 0–4、6、7；
- 旧失败标记和 v1 数据只作为审计记录，不得直接改名为 v2 或绕过验证。

正式 cache 校验值：

```text
8f320295504ffc0a7759ba76ee454dd2b1e3f1feb7244d476b2373100235cc9b  KIT/314/walking_medium09_poses.npz
29346e54820c2fc6fbe65a47da77e9974f1fe671798d5eb31196649daa615530  KIT/425/walking_slow07_poses.npz
72da69268dc658cf8bebeb7a480f3ed9b727538b8f7322f0b42231cfe4d7b6f2  KIT/167/turn_right01_poses.npz
dc19c93e75ce1158c3724dd4e590d52b703209038028c97b3de9448257abf476  KIT/167/turn_left01_poses.npz
```
