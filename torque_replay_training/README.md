# Torque Replay Prosthesis Training

当前正式训练使用官方 `MJLAB 1.5.3 + MuJoCo Warp + RSL-RL PPO`：

- 4096 个 GPU 并行环境；
- 非假肢 84 DOF 回放逐物理子步记录的广义力；
- 左膝 1 DOF、左踝/距下/MTP 3 DOF 由 policy 接管；
- replay 只作探索初值，不使用 BC、AMP 或模仿 reward；
- A100 只允许物理 GPU 5。

详细架构、reward 和 PPO loss：
[FLAT_WALK_TRAINING.md](../docs/FLAT_WALK_TRAINING.md)。

本机可视化、回放审计、smoke、checkpoint 测试、A100 启动和同步：
[COMMANDS.md](../docs/COMMANDS.md)。

## 入口

```text
src/torque_replay_training/mjlab_task/config.py   环境与官方 PPO 配置
src/torque_replay_training/mjlab_task/action.py   回放力与四维假肢动作
src/torque_replay_training/mjlab_task/mdp.py      观测、reward、done、metrics
scripts/audit_mjlab_replay.py                     MJLAB 回放 gate
scripts/train_mjlab_prosthesis.py                 官方 RSL-RL 训练
scripts/evaluate_mjlab_checkpoint.py              beta=0 独立评估
scripts/run_mjlab_smoke_a100.sh                    A100 smoke
scripts/start_mjlab_training_a100.sh               A100 正式训练
scripts/mjlab_training_status.py                   只读状态
```

## 测试

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python -m pytest torque_replay_training/tests -q
```

数据生成、schema-v3 精确回放和全部动作浏览仍见：

- [FULLBODY_TORQUE_REPLAY.md](../docs/FULLBODY_TORQUE_REPLAY.md)
- [ALL_MOTION_REPLAY.md](../docs/ALL_MOTION_REPLAY.md)

旧 JAX、HORA 和自写 Warp PPO 文件仅保留作历史对照，不再是正式训练入口。
