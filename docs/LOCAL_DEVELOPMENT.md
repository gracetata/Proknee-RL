# 本机开发、测试与可视化

## 1. 独立路径和环境

本机只在受 Git 管理的独立 checkout 中开发：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
git branch --show-current   # 必须是 muscle
git status --short
```

当前可复用已验证的 Python 3.11 环境：

```bash
ln -s /home/user/Workspace/musclemimic/musclemimic/.venv \
  /home/user/Workspace/Proknee-RL-muscle/.venv
mkdir -p /home/user/Workspace/Proknee-RL-muscle/data/checkpoints
ln -s /home/user/Workspace/musclemimic/data/checkpoints/mm-10m-2 \
  /home/user/Workspace/Proknee-RL-muscle/data/checkpoints/mm-10m-2
```

如果不复用环境，可在项目根目录新建 `.venv` 后安装：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,cuda]'
```

GMR cache 位于用户级路径：
`/home/user/.musclemimic/caches/AMASS/MyoFullBody/gmr`。

## 2. 本机测试链路

```bash
cd /home/user/Workspace/Proknee-RL-muscle
PYTHONPATH=torque_replay_training/src:. \
  .venv/bin/python -m pytest torque_replay_training/tests
bash torque_replay_training/scripts/run_smoke.sh
```

不打开窗口的可视化链路检查：

```bash
.venv/bin/python torque_replay_training/scripts/visualize_policy_local.py \
  --config torque_replay_training/configs/smoke.yaml \
  --dataset torque_replay_training/data/smoke/walking_medium09_8steps.npz \
  --policy torque_replay_training/outputs/smoke/ppo/policy_000000032.msgpack \
  --steps 8 --check-only
```

## 3. 本机 MuJoCo 可视化

以下命令只能在有桌面 `DISPLAY` 的本机运行，不能在 A100 运行。

可视化健康 baseline：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/visualize_policy_local.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody_v1/KIT_314_walking_medium09_poses.npz \
  --steps 1000
```

可视化训练后的确定性策略：

```bash
.venv/bin/python torque_replay_training/scripts/visualize_policy_local.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody_v1/*.npz \
  --policy torque_replay_training/outputs/a100_train_v1/seed_0/policy_000200000.msgpack \
  --dataset-index 0 --steps 1000
```

窗口会跟随人体 root；省略 `--policy` 表示零残差健康力矩 baseline。

## 4. 从 A100 同步数据和训练结果

代码只通过 Git 同步。生成数据和输出通过 `rsync`：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
rsync -av --partial -e 'ssh -p 6029' \
  root@39.105.12.60:/workspace/Proknee-RL-muscle/torque_replay_training/data/fullbody_v1/ \
  torque_replay_training/data/fullbody_v1/

rsync -av --partial -e 'ssh -p 6029' \
  root@39.105.12.60:/workspace/Proknee-RL-muscle/torque_replay_training/outputs/a100_train_v1/ \
  torque_replay_training/outputs/a100_train_v1/
```

同步后先运行 `--check-only`，再打开 MuJoCo GUI。

## 5. TensorBoard

直接查看同步到本机的 event：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/tensorboard \
  --logdir torque_replay_training/outputs/a100_train_v1 \
  --port 6012
```

或使用 A100 的实时 TensorBoard SSH 隧道，见
[A100 文档](A100_DEPLOYMENT.md#6-tensorboard)。浏览器操作都在本机完成。

## 6. 提交前同步检查

```bash
cd /home/user/Workspace/Proknee-RL-muscle
git diff --check
git status --short
git rev-parse HEAD
git ls-remote origin refs/heads/muscle
ssh -p 6029 root@39.105.12.60 \
  'git -C /workspace/Proknee-RL-muscle rev-parse HEAD'
```

三个 SHA 不一致时，不启动新的 A100 训练。
