# Torque Replay Prosthesis Training

这是与旧假肢训练、蒸馏和 DAgger 路线隔离的新实现。它完成四件事：

1. 用官方 MuscleMimic checkpoint 跟踪完整人体动作；
2. 在 500 Hz MuJoCo 物理子步导出 `qfrc_actuator` 和状态；
3. 关闭原 actuator，在非假肢 DOF 回放健康力矩；
4. 在左膝 1 DOF 和左踝—足 3 DOF 上训练 `healthy baseline + PPO residual`。

方法解释和生产训练流程见 [假肢训练文档](../docs/PROSTHESIS_TRAINING_PLAN.md)。

## 固定环境

在 workspace 根目录执行：

```bash
cd /home/user/Workspace/musclemimic
if [[ -x musclemimic/.venv/bin/python ]]; then
  PYTHON="$PWD/musclemimic/.venv/bin/python"  # 当前嵌套 checkout
else
  PYTHON="$PWD/.venv/bin/python"              # 标准 Git clone
fi
```

本次验证环境：Python 3.11.15、MuJoCo 3.4.0、JAX 0.7.2、Flax 0.12.0、Optax 0.2.8。代码复用上游 checkout 的环境构建和 checkpoint loader，不修改上游旧假肢模块。

## 一键 smoke

```bash
bash torque_replay_training/scripts/run_smoke.sh
```

A100 服务器必须使用物理 GPU 5–7，并运行：

```bash
bash torque_replay_training/scripts/run_smoke_a100.sh
```

完整远端部署说明见 [A100_DEPLOYMENT.md](../docs/A100_DEPLOYMENT.md)。

它依次执行 8 步 tracker 数据导出、两种回放等价性验证、32 步 PPO 更新、checkpoint 重新加载和确定性评估。`--allow-incomplete` 只在这个 smoke 中使用，不能生成正式训练集。

## 生产命令

批量导出完整动作，命令只有当每条动作从头到尾合格时才返回 0：

```bash
$PYTHON torque_replay_training/scripts/collect_rollouts.py \
  --output-dir torque_replay_training/data/fullbody \
  --motion KIT/314/walking_medium09_poses \
  --motion KIT/425/walking_slow07_poses \
  --motion KIT/348/turn_right03_poses \
  --motion KIT/167/turn_left05_poses
```

对每个正式 `.npz` 做力矩回放等价性检查：

```bash
$PYTHON torque_replay_training/scripts/validate_replay.py \
  --dataset torque_replay_training/data/fullbody/KIT_314_walking_medium09_poses.npz
```

用多条动作训练；每次 episode reset 会随机选择一条完整 motion 数据：

```bash
$PYTHON torque_replay_training/scripts/train_policy.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody/*.npz \
  --output torque_replay_training/outputs/train_v1
```

加载 checkpoint 并确定性评估：

```bash
$PYTHON torque_replay_training/scripts/evaluate_policy.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody/*.npz \
  --policy torque_replay_training/outputs/train_v1/policy_000200000.msgpack \
  --episodes 20
```

## 测试

```bash
PYTHONPATH=torque_replay_training/src:musclemimic \
  $PYTHON -m pytest torque_replay_training/tests
```

输出目录下的 `metrics.jsonl` 保存每轮 PPO 指标，`policy_*.msgpack` 保存参数树、数据来源和完整训练配置。
