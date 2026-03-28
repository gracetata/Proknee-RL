# Outputs 目录结构

## 概览

所有训练产物、检查点和评估结果存放在 `outputs/` 目录下。

```
outputs/
├── checkpoints/                    # 📌 规范检查点（最终模型）
│   ├── stage0/
│   │   ├── stage0_amp_walk_5050.pth     # 单动作: walk
│   │   ├── stage0_amp_run_XXXX.pth      # 单动作: run
│   │   ├── stage0_amp_dance_XXXX.pth    # 单动作: dance
│   │   └── stage0_amp_multi_XXXX.pth    # 多动作: walk+run+dance (用于 Stage 1/2)
│   ├── stage1/
│   │   └── best.pth                      # DAgger教师策略 (4-DOF假肢)
│   └── stage2/
│       └── best.pth                      # ProprioAdapt学生策略 (可部署)
│
├── stage1_dagger_final/            # Stage 1 walk-only 训练运行记录
│   ├── checkpoints/
│   └── tb/                         # TensorBoard日志
│
├── stage2_padapt_final/            # Stage 2 walk-only 训练运行记录
│   ├── checkpoints/
│   └── tb/                         # TensorBoard日志
│
├── eval_stage2/                    # Stage 2 评估结果
│   ├── gait_cycle_analysis.png     # 步态周期分析图
│   └── joint_angles.csv            # 原始关节角度数据
│
└── _archive/                       # 已归档的旧训练运行（可删除）
```

## 检查点说明

### `checkpoints/` — 规范模型（脚本默认使用这些）

**Stage 0 (Body Policy):**

| 文件 | 动作 | 说明 |
|------|------|------|
| `stage0_amp_walk_5050.pth` | walk | 5050 iters, ep_len=287 |
| `stage0_amp_run_XXXX.pth` | run | 待训练 |
| `stage0_amp_dance_XXXX.pth` | dance | 待训练 |
| `stage0_amp_multi_XXXX.pth` | walk+run+dance | 多动作统一策略 |

**Stage 1/2:**

| 文件 | 说明 |
|------|------|
| `stage1/best.pth` | DAgger教师 (当前: walk-only, 后续: multi) |
| `stage2/best.pth` | ProprioAdapt学生 (当前: walk-only, 后续: multi) |

### 训练脚本保存逻辑

- **Stage 0**: 保存在 `IsaacGymEnvs/isaacgymenvs/runs/` → 手动复制到 `checkpoints/stage0/`
- **Stage 1** (`train_stage1_dagger.py`): 自动复制 best → `checkpoints/stage1/best.pth`
- **Stage 2** (`train_stage2.py`): 自动复制 best → `checkpoints/stage2/best.pth`

### 多动作 YAML 配置

```
IsaacGymEnvs/assets/amp/motions/multi_locomotion.yaml
  → walk(1.0) + run(1.0) + dance(0.5)
```

## TensorBoard 查看

```bash
tensorboard --logdir IsaacGymEnvs/isaacgymenvs/runs/ --port 6006   # Stage 0 (all runs)
tensorboard --logdir outputs/stage1_dagger_*/tb --port 6007          # Stage 1
tensorboard --logdir outputs/stage2_padapt_*/tb --port 6008          # Stage 2
```

## 归档说明

`_archive/` 包含旧的训练尝试（PPO方法、freeze_body bug修复前的运行等）。
可以安全删除以节省磁盘空间：`rm -rf outputs/_archive/`
