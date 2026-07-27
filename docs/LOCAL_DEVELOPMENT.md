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

## 3. 生成和验证完整人体回放数据

schema v1 将关键物理量保存成 float32，长时间接触回放会发散，禁止继续使用。
本机和 A100 的新数据目录统一为 `fullbody_v2`。一次生成四条完整轨迹：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python torque_replay_training/scripts/collect_rollouts.py \
  --output-dir torque_replay_training/data/fullbody_v2 \
  --motion KIT/314/walking_medium09_poses \
  --motion KIT/425/walking_slow07_poses \
  --motion KIT/167/turn_right01_poses \
  --motion KIT/167/turn_left01_poses
```

逐文件执行全长 `all`、零残差 `split` 和三个随机起点窗口验证：

```bash
for dataset in torque_replay_training/data/fullbody_v2/*.npz; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python torque_replay_training/scripts/validate_replay.py \
    --dataset "${dataset}"
done
```

不打开窗口的完整可视化链路检查：

```bash
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset torque_replay_training/data/fullbody_v2/*.npz \
  --check-only
```

成功结果必须同时满足：四条轨迹走到末尾、`fell=false`，并且 `qpos_max_abs`、
`qvel_max_abs` 在门限内。当前本机重新导出的四条 v2 轨迹全部为零误差。

## 4. 本机 MuJoCo 可视化

以下命令只能在有桌面 `DISPLAY` 的本机运行，不能在 A100 运行。
桌面可视化使用本机默认 GLFW/OpenGL 后端，不要强制设置 `MUJOCO_GL=egl`；EGL 只用于
明确需要离屏渲染的独立进程。

依次可视化中速行走、慢速行走、左右转弯的精确全身力矩回放：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset \
    torque_replay_training/data/fullbody_v2/KIT_314_walking_medium09_poses.npz \
    torque_replay_training/data/fullbody_v2/KIT_425_walking_slow07_poses.npz \
    torque_replay_training/data/fullbody_v2/KIT_167_turn_right01_poses.npz \
    torque_replay_training/data/fullbody_v2/KIT_167_turn_left01_poses.npz
```

只看中速行走并循环两次：

```bash
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset torque_replay_training/data/fullbody_v2/KIT_314_walking_medium09_poses.npz \
  --repeat 2
```

窗口会跟随 pelvis/root；`--realtime-factor 0.5` 是半速，`2` 是两倍速，
`--start-step N` 可从任意合法帧开始，并恢复对应 MuJoCo warmstart。

在全身回放验证通过之后，才使用下面的工具查看训练后的假肢策略：

```bash
.venv/bin/python torque_replay_training/scripts/visualize_policy_local.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody_v2/*.npz \
  --policy torque_replay_training/outputs/a100_train_v2/seed_0/policy_000200000.msgpack \
  --dataset-index 0 --steps 1000
```

## 5. 从 A100 同步数据和训练结果

代码只通过 Git 同步。服务器没有安装 `rsync` 时使用 `scp`：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
scp -P 6029 -r \
  root@39.105.12.60:/workspace/Proknee-RL-muscle/torque_replay_training/data/fullbody_v2 \
  torque_replay_training/data/

scp -P 6029 -r \
  root@39.105.12.60:/workspace/Proknee-RL-muscle/torque_replay_training/outputs/a100_train_v2 \
  torque_replay_training/outputs/
```

同步后先运行 `--check-only`，再打开 MuJoCo GUI。

## 6. TensorBoard

直接查看同步到本机的 event：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/tensorboard \
  --logdir torque_replay_training/outputs/a100_train_v2 \
  --port 6012
```

或使用 A100 的实时 TensorBoard SSH 隧道，见
[A100 文档](A100_DEPLOYMENT.md#6-tensorboard)。浏览器操作都在本机完成。

## 7. 提交前同步检查

```bash
cd /home/user/Workspace/Proknee-RL-muscle
git diff --check
git status --short
git rev-parse HEAD
git ls-remote origin refs/heads/muscle
ssh -p 6029 root@39.105.12.60 \
  'git -C /workspace/Proknee-RL-muscle rev-parse HEAD'
```

三个 SHA 不一致时，不启动新的 A100 训练。当前 A100 正式训练只使用物理 GPU 5。
