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

项目要求 Python 3.11。远端系统 Python 3.10 不可直接使用。部署时使用独立的 Python 3.11 和 `uv`：

```bash
cd /workspace/Proknee-RL-muscle
uv sync --python 3.11 --extra cuda --extra dev
```

验证版本和 CUDA 后端：

```bash
torque_replay_training/scripts/a100_exec.sh .venv/bin/python - <<'PY'
import jax, mujoco
print("JAX:", jax.__version__)
print("MuJoCo:", mujoco.__version__)
print("devices:", jax.devices())
assert jax.devices()[0].platform == "gpu"
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

