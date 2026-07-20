# A100 部署与训练

## 1. 固定服务器与路径

```sshconfig
Host A100
  HostName 39.105.12.60
  User root
  Port 6029
```

- 部署目录：`/workspace/Proknee-RL-muscle`
- Git 仓库：`gracetata/Proknee-RL`
- 分支：`muscle`
- Python 3.11：`/workspace/.tools/cpython-3.11.15-linux-x86_64-gnu`
- Python 环境：`/workspace/Proknee-RL-muscle/.venv`
- checkpoint：`/workspace/Proknee-RL-muscle/data/checkpoints/mm-10m-2`
- GMR cache：`/root/.musclemimic/caches/AMASS/MyoFullBody/gmr`

## 2. GPU 约束

这个部署只能使用物理 GPU 5、6、7，禁止使用 GPU 0–4。所有训练和 smoke 命令必须通过：

```bash
torque_replay_training/scripts/a100_exec.sh COMMAND [ARG ...]
```

该入口会无条件设置：

```bash
CUDA_VISIBLE_DEVICES=5,6,7
XLA_PYTHON_CLIENT_PREALLOCATE=false
MUJOCO_GL=egl
```

因此程序内部看到的逻辑设备与物理设备关系是：

| JAX/CUDA 逻辑编号 | 物理 GPU |
|---:|---:|
| 0 | 5 |
| 1 | 6 |
| 2 | 7 |

当前 torque-replay PPO 是单环境顺序采样，默认 JAX 设备为逻辑 0，即物理 GPU 5。GPU 6、7 保留为允许设备，但启用多卡训练前仍应先用 `nvidia-smi -i 5,6,7` 检查占用。

## 3. 环境安装

项目要求 Python 3.11。远端系统 Python 3.10 不可直接使用。本次部署使用独立的
Python 3.11.15；不要使用系统 Python，也不要依赖容器中来源不明的 `uv` 二进制。

当前环境已经配置完成。若需要在同一台服务器重建 `.venv`，执行：

```bash
cd /workspace/Proknee-RL-muscle
PYTHON=/workspace/.tools/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
"${PYTHON}" -m venv .venv

PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  .venv/bin/python -m pip install -e . pytest

PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  .venv/bin/python -m pip install 'jax[cuda12]==0.7.2'
```

第二条安装命令是必要步骤：它安装与 JAX 0.7.2 匹配的 CUDA 12、cuDNN、NCCL
运行库，避免直接加载容器系统中不匹配的 cuDNN。`tfp-nightly` 在镜像站缺失时会从
官方 PyPI 补充，不能因为镜像缺包而删除该依赖。

验证版本和 CUDA 后端：

```bash
torque_replay_training/scripts/a100_exec.sh .venv/bin/python - <<'PY'
import jax, mujoco
print("JAX:", jax.__version__)
print("MuJoCo:", mujoco.__version__)
print("devices:", jax.devices())
assert jax.devices()[0].platform == "gpu"
assert len(jax.devices()) == 3
PY
```

## 4. Smoke 测试

一键运行数据导出、力矩等价性验证、32 步 PPO、checkpoint 重载和确定性评估：

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/run_smoke_a100.sh
```

该脚本会先检查 `CUDA_VISIBLE_DEVICES=5,6,7` 和 JAX GPU 后端，检查失败时不会开始训练。

## 5. 正式数据收集

完整人体 MuJoCo rollout 主要在 CPU 物理引擎执行，但 policy 推理仍通过相同 GPU 限制入口运行：

```bash
cd /workspace/Proknee-RL-muscle
torque_replay_training/scripts/a100_exec.sh .venv/bin/python \
  torque_replay_training/scripts/collect_rollouts.py \
  --output-dir torque_replay_training/data/fullbody \
  --motion KIT/314/walking_medium09_poses \
  --motion KIT/425/walking_slow07_poses \
  --motion KIT/348/turn_right03_poses \
  --motion KIT/167/turn_left05_poses
```

不要给正式数据收集命令添加 `--allow-incomplete`。

## 6. 正式训练

```bash
cd /workspace/Proknee-RL-muscle
torque_replay_training/scripts/a100_exec.sh .venv/bin/python \
  torque_replay_training/scripts/train_policy.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody/*.npz \
  --output torque_replay_training/outputs/a100_train_v1
```

查看当前允许 GPU 的占用：

```bash
nvidia-smi -i 5,6,7
```

## 7. 更新代码

当远端 DNS 和 GitHub 网络可用时：

```bash
cd /workspace/Proknee-RL-muscle
git pull --ff-only proknee muscle
```

如果容器无法解析 GitHub，应从可信本地 checkout 同步代码，不要修改为其他 GPU 或改用 0–4 号卡规避问题。

## 8. 本次部署与验证记录

验证日期：2026-07-20。

- 远端分支：`muscle`
- smoke 测试时基线提交：`2c180a1`
- Python：3.11.15
- JAX：0.7.2，CUDA 后端
- MuJoCo：3.4.0
- JAX 可见设备：`CudaDevice(id=0..2)`，对应物理 GPU 5–7
- 单元测试：`5 passed`
- smoke 数据：`walking_medium09_8steps.npz`，8 control steps，5 physics substeps
- 回放验证：通过；最大误差 `qpos=4.995e-9`、`qvel=8.602e-8`
- PPO smoke：32 steps，产生 `policy_000000032.msgpack`
- 确定性评估：1 episode，`falls=0`，`mean_length=8`，`mean_return=7.9972`
- 测试结束后物理 GPU 5、6、7 均为 4 MiB；未在 GPU 0–4 启动本项目进程

部署所用 checkpoint 和 GMR cache 在解压前已做 SHA-256 校验：

```text
f226275a1ccc7d3ed0fba58b70cff2ba1b82715e01eed458094d46a5d7134a29  mm-10m-2-checkpoint.tar.gz
46f350968b0bacaf1b98b553c90303edec413343a13b2390bce38c8ad6c56ab5  musclemimic-kit314-smoke-cache.tar.gz
```

以上 smoke 只验证数据导出、广义力拆分回放、训练、保存和重载链路可运行，不能替代
完整行走/转弯数据集上的长时稳定性和假肢性能验收。
