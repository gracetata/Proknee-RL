# Torque Replay Prosthesis Training

这是与旧假肢训练、蒸馏和 DAgger 路线隔离的新实现。它完成四件事：

1. 用官方 MuscleMimic checkpoint 跟踪完整人体动作；
2. 在 500 Hz MuJoCo 物理子步导出 `qfrc_actuator` 和状态；
3. 关闭原 actuator，在非假肢 DOF 回放健康力矩；
4. 在左膝 1 DOF 和左踝—足 3 DOF 上训练 `healthy baseline + PPO residual`。

当前全身回放的物理边界、schema v3、验证和同步规则见
[全身广义力回放文档](../docs/FULLBODY_TORQUE_REPLAY.md)。假肢训练流程见
[假肢训练文档](../docs/PROSTHESIS_TRAINING_PLAN.md)。全部 1,089 条候选动作的清点和
本机 RTX 4090 headless 批处理见
[全部轨迹回放文档](../docs/ALL_MOTION_REPLAY.md)。
平地行走的受试者隔离数据集、baseline/residual 配置、A100 命令和验收门槛见
[平地行走训练文档](../docs/FLAT_WALK_TRAINING.md)。

## 固定环境

在受 Git 管理的本机独立 checkout 执行：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
PYTHON="$PWD/.venv/bin/python"
```

本次验证环境：Python 3.11.15、MuJoCo 3.4.0、JAX 0.7.2、Flax 0.12.0、Optax 0.2.8。代码复用上游 checkout 的环境构建和 checkpoint loader，不修改上游旧假肢模块。

## 一键 smoke

```bash
bash torque_replay_training/scripts/run_smoke.sh
```

A100 当前只使用物理 GPU 5，并运行：

```bash
bash torque_replay_training/scripts/run_smoke_a100.sh
```

完整远端部署说明见 [A100_DEPLOYMENT.md](../docs/A100_DEPLOYMENT.md)。
本机测试和 MuJoCo GUI 说明见 [LOCAL_DEVELOPMENT.md](../docs/LOCAL_DEVELOPMENT.md)。

它依次执行 8 步 tracker 数据导出、两种回放等价性验证、32 步 PPO 更新、checkpoint 重新加载和确定性评估。`--allow-incomplete` 只在这个 smoke 中使用，不能生成正式训练集。

## 生产命令

批量导出完整动作，命令只有当每条动作从头到尾合格时才返回 0：

```bash
$PYTHON torque_replay_training/scripts/collect_rollouts.py \
  --output-dir torque_replay_training/data/fullbody_v3 \
  --motion KIT/314/walking_medium09_poses \
  --motion KIT/425/walking_slow07_poses \
  --motion KIT/167/turn_right01_poses \
  --motion KIT/167/turn_left01_poses
```

对每个正式 `.npz` 做力矩回放等价性检查：

```bash
for dataset in torque_replay_training/data/fullbody_v3/*.npz; do
  $PYTHON torque_replay_training/scripts/validate_replay.py --dataset "${dataset}"
done
```

schema v3 严格遵循官方 `env.step` 顺序，在每个 `mj_step` 后记录实际使用的
float64 `qfrc_actuator/rollout_qacc`，恢复随机起点的 MuJoCo warmstart，并在 split
模式回放所有非假肢 DOF。旧 v1、v2 数据不兼容，不能用于训练。

本机交互查看全部已验证的精确全身力矩回放：

```bash
$PYTHON torque_replay_training/scripts/visualize_fullbody_replay_local.py
```

轨迹播完后自动循环当前条；在 MuJoCo 窗口按空格切到下一条，关闭窗口退出。脚本根据
`data/fullbody_all_v3/validation_manifest.json` 逐条按需加载，不会把全部 12 GB 数据
同时放入内存。无窗口验证使用：

```bash
$PYTHON torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset torque_replay_training/data/fullbody_v3/*.npz --check-only
```

用多条动作训练；每次 episode reset 会随机选择一条完整 motion 数据：

```bash
$PYTHON torque_replay_training/scripts/train_policy.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody_v3/*.npz \
  --output torque_replay_training/outputs/train_v1
```

加载 checkpoint 并确定性评估：

```bash
$PYTHON torque_replay_training/scripts/evaluate_policy.py \
  --config torque_replay_training/configs/train.yaml \
  --dataset torque_replay_training/data/fullbody_v3/*.npz \
  --policy torque_replay_training/outputs/train_v1/policy_000200000.msgpack \
  --episodes 20
```

## 测试

```bash
PYTHONPATH=torque_replay_training/src:musclemimic \
  $PYTHON -m pytest torque_replay_training/tests
```

输出目录下的 `metrics.jsonl` 保存每轮 PPO 指标，`policy_*.msgpack` 保存参数树、数据来源和完整训练配置。
每个训练目录的 `tensorboard/` 保存同一批指标对应的 TensorBoard event。
