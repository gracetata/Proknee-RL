# ProKnee Multi-Motion Expansion — 进度跟踪

> 本文档跟踪多动作扩展项目的进度。单动作平地行走部分已完成，详见 `MEMORY.md`。

---

## 项目目标

在 proknee_hora 中构建多动作 Teacher-Student 架构，使 Stage 2 假肢策略能够根据指令实现多种动作适应：
- 站立 (Stand)
- 行走 (Walk) ✅ 已完成
- 奔跑 (Run)
- 跳舞 (Dance)
- 动作组合序列 (走-停-走, 走-跑-走-停-跑)

---

## 架构概述

```
Per-Motion Stage 0 Body Policies (28-DOF)
├── walk_policy.pth   ✅ ep_len=287.2
├── run_policy.pth    ✅ ep_len=287.5
├── dance_policy.pth  ✅ ep_len=295.3
└── stand_policy.pth  ✅ ep_len=299.0

Multi-Motion Environment (ProKneeMultiMotionEnv)
├── 继承 ProKneeBase，不修改原类
├── Per-env motion type 随机分配
├── 两种观测模式: Realistic (obs=16D) / Legacy (obs=20D)
└── 按 motion_id 切换 body policy + smooth_transition / soft_reset

Multi-Motion Stage 1 Teacher (DAgger)
├── Realistic: obs(16D) + priv_info(117D) → 4D action  ★ 推荐
├── Legacy:    obs(20D) + priv_info(113D) → 4D action
└── 多 body policy 提供 ground truth

Multi-Motion Stage 2 Student (Hora蒸馏)
├── Realistic: proprio_hist(30×16D) → adapt_tconv → 32D latent  ★ 推荐
├── Legacy:    proprio_hist(30×20D) → adapt_tconv → 32D latent
└── 纯 MSE 蒸馏，与 walk-only 相同架构
```

---

## Phase 进度

### Phase 0: 基础设施 & 验证

| 任务 | 状态 | 说明 |
|------|------|------|
| 创建 PROGRESS.md | ✅ 完成 | 本文档 |
| 验证 Run Stage 0 (stage0_amp_run_1700.pth) | ✅ 完成 | ep_len=287.5 |
| 验证 Dance Stage 0 (stage0_amp_dance_8000.pth) | ✅ 完成 | ep_len=295.3 |
| 创建站立参考动作 (amp_humanoid_stand.npy) | ✅ 完成 | walk frame116 对称化, root_z=0.842 |
| 训练 Stand Stage 0 | ✅ 完成 | ep_len=299.0 (stage0_amp_stand_900.pth, best@iter900) |

### Phase 1: Multi-Motion 环境

| 任务 | 状态 | 说明 |
|------|------|------|
| constants_multi.py | ✅ 完成 | 多动作常量定义 |
| proknee_multi_motion.py | ✅ 完成 | 多动作环境类 |

### Phase 2: Multi-Motion Stage 1 Teacher

| 任务 | 状态 | 说明 |
|------|------|------|
| train_stage1_multi.py | ✅ 完成 | 多动作 DAgger 训练脚本 |
| 训练多动作 Stage 1 | ✅ 完成 | best ep_len=261.0, loss=0.002, 2.21h |
| 验收: 各动作 ep_len ≥ 250 | ✅ 通过 | 整体 ep_len=261.0 (目标250) |

### Phase 3: Multi-Motion Stage 2 Student

| 任务 | 状态 | 说明 |
|------|------|------|
| train_stage2_multi.py | ✅ 完成 | 多动作蒸馏训练脚本 |
| 训练多动作 Stage 2 | ✅ 完成 | 300M steps, best_reward=726.08 |
| 验收: per-motion ep_len ≥ 240 | ✅ 通过 | walk=282.9, run=280.7, dance=275.9, stand=276.4 |

### Phase 4: 动作切换 & 序列

| 任务 | 状态 | 说明 |
|------|------|------|
| Episode 内动作切换机制 | ✅ 完成 | switch_motion_in_episode + transition blending |
| 走-停-走 序列 | ✅ 完成 | survival=10.7% (blend=15), 需 retrain w/ switching 提升 |
| 走-跑-走-停-跑 序列 | ✅ 完成 | survival=0.0%, 长序列挑战大 |
| 转换平滑度评估 | ✅ 完成 | action_diff < 0.06 (很平滑), 瓶颈在 body policy 突变 |
| evaluate_motion_sequences.py | ✅ 完成 | 4种序列 + 4种单动作评估 |

**序列评估总结**: per-motion 表现优秀 (87-91% survival)，但 mid-episode 切换的 survival 较低。
根本原因: Stage 2 训练时未启用 intra-episode switching，body policy 突变导致不稳定。
改进路径: 以 `switch_motion_in_episode=True` 重训 Stage 1 & 2。

### Phase 5: 文档 & 清理

| 任务 | 状态 | 说明 |
|------|------|------|
| 更新 MEMORY.md | ✅ 完成 | §14 多动作训练命令 |
| 更新 METHOD.md | ✅ 完成 | 多动作架构说明 |
| 更新 PROGRESS.md | ✅ 完成 | 全部评估结果已记录 |

---

## Stage 2 Multi-Motion 评估结果

### Per-Motion 独立评估 (blend=15, 256 envs, 50 episodes)

| 动作 | Avg Ep Len | Std | Survival Rate |
|------|-----------|-----|---------------|
| Walk | 282.9 | 54.9 | 91.1% |
| Run | 280.7 | 56.3 | 89.9% |
| Dance | 275.9 | 60.4 | 87.5% |
| Stand | 276.4 | 59.1 | 88.2% |

✅ 所有动作 ep_len ≥ 240 验收通过

### 动作序列评估 (blend=15, 5 trials)

| 序列 | 总步数 | Survival | Avg Reward |
|------|--------|----------|------------|
| walk→stand→walk | 250 | 10.7% | 286.1 |
| walk→run→walk→stand→run | 300 | 0.0% | 324.4 |
| dance→stand→dance | 200 | 17.9% | 307.1 |
| stand→walk→run→dance→stand→walk | 300 | 0.0% | 193.7 |

### 转换平滑度 (action_diff at transition points)

| 转换 | Action Diff | 评估 |
|------|------------|------|
| walk→stand | 0.0215 | ✅ 非常平滑 |
| stand→walk | 0.0612 | ✅ 平滑 |
| walk→run | 0.0213 | ✅ 非常平滑 |
| run→walk | 0.0238 | ✅ 非常平滑 |
| dance→stand | 0.0238 | ✅ 非常平滑 |
| stand→dance | 0.0463 | ✅ 平滑 |

**分析**: 假肢动作转换非常平滑 (action_diff < 0.07)，survival 低的根本原因是 body policy 突变。
body policy 从一种全身动作突然切换到另一种，造成物理状态不连续。

### Switch 训练后序列评估 (blend=15, 256 envs, switch model)

| 序列 | 总步数 | Survival (switch) | Survival (旧) | 变化 |
|------|--------|-------------------|--------------|------|
| walk→stand→walk | 250 | **22.8%** | 10.7% | ↑ 12.1pp |
| walk→run→walk→stand→run | 300 | 0.0% | 0.0% | — |
| dance→stand→dance | 200 | 13.0% | 17.9% | ↓ 4.9pp |
| full-cycle (6段) | 300 | 0.0% | 0.0% | — |

Per-Motion 独立表现 (switch model):

| 动作 | Avg Ep Len | Survival | vs 旧模型 |
|------|-----------|----------|----------|
| Walk | 282.0 | 89.9% | ≈ 持平 |
| Run | 283.1 | 90.8% | ≈ 持平 |
| Dance | 276.5 | 88.0% | ≈ 持平 |
| Stand | 268.2 | 84.0% | ↓ 4.2pp |

**Switch 训练分析**:
- ✅ walk→stand→walk survival 提升 2x (10.7% → 22.8%)
- ⚠️ 长序列 (4+段) 仍然失败，根因不变: body policy 突变是硬性不连续
- ⚠️ Stand 单动作略有下降 (88.2% → 84.0%)，可能因为新站立参考动作的双脚对称姿态与走路差异更大
- 假肢动作平滑度依旧优秀 (action_diff < 0.025)

**结论**: mid-episode switching 训练有一定效果，但根本瓶颈是 **body policy 的物理不连续性**。

**后续改进路径**:
1. ~~短期: transition blending (已实现 ✅)~~
2. ~~中期: switch-motion 重训 (已完成 ✅, 效果有限)~~
3. **长期方案 A**: Multi-motion conditioned body policy — 单个网络输入 motion_id 输出全身动作
4. **长期方案 B**: Motion interpolation body policy — 在 body action 层面做过渡 (非线性插值)
5. **长期方案 C**: Curriculum learning — 先训短序列 (2段)，逐步增加段数

### Phase 6: 站立动作修复 & 过渡训练

| 任务 | 状态 | 说明 |
|------|------|------|
| 修复站立参考动作 (单脚→双脚对称) | ✅ 完成 | walk frame116 slerp 对称化, root_z=0.842 |
| 重训 Stand Stage 0 | ✅ 完成 | ep_len=299.0@iter900 (stage0_amp_stand_900.pth) |
| 添加 --switch-motion 到 train_stage1_multi.py | ✅ 完成 | + --blend-steps 参数 |
| 添加 --switch-motion 到 train_stage2_multi.py | ✅ 完成 | + --blend-steps 参数 |
| 添加 --visualize 到 evaluate_motion_sequences.py | ✅ 完成 | headless 由参数控制 |
| 训练 Stage 1 Multi Switch | ✅ 完成 | 8000 epochs, 3.71h, best ep_len=242.6, loss=0.003 |
| 训练 Stage 2 Multi Switch | ✅ 完成 | 300M steps, best_reward=482.2, ep_len=227 |
| 评估动作序列 (switch 模型) | ✅ 完成 | walk-stop-walk 22.8% (↑2x), 长序列仍 0% |
| 更新文档 (MEMORY/METHOD/PROGRESS) | ✅ 完成 | 全量更新 |

### Phase 7: 可视化修复 & 步态分析

| 任务 | 状态 | 说明 |
|------|------|------|
| 修复序列评估 Bug (多env各做各的) | ✅ 完成 | 添加 `manual_motion_control` 标志, 防止 reset 覆盖动作 |
| 修复 evaluate_motion_sequences.py | ✅ 完成 | 启用 manual_motion_control, `--visualize` 默认 1 env, 添加 `--sequence` |
| 测试单 env 连续序列切换 | ✅ 完成 | 256 envs headless 验证通过 |
| 创建步态分析脚本 analyze_gait.py | ✅ 完成 | 膝/踝/髋 角度+力矩 vs 健侧对比 |
| 运行步态分析 | ✅ 完成 | 4动作 × 300步, 5张图 + 统计表 |
| 更新文档 (MEMORY/PROGRESS) | ✅ 完成 | 步态分析命令、结果全部记录 |

### Phase 8: Soft-Reset 动作过渡 ★

| 任务 | 状态 | 说明 |
|------|------|------|
| 诊断 Body Policy OOD 问题 | ✅ 完成 | 独立 body policy 无法处理跨动作状态观测 |
| 尝试 Deceleration-First Blending | ✅ 已放弃 | Phase1减速导致失衡摔倒, Phase2 OOD 输出不可靠 |
| 尝试 Velocity-Aware Blending | ✅ 已放弃 | 加速方向稍有改善, 减速方向仍不稳定 |
| 实现 Soft-Reset 过渡 | ✅ 完成 | `soft_reset_to_motion()` 保留XY位置+RSI关节状态 |
| Full-Cycle 单 env 测试 | ✅ 完成 | 6段全部通过, 0次摔倒 |
| 64 envs 全序列评估 | ✅ 完成 | 所有序列 survival 85-96% |
| 更新 evaluate_motion_sequences.py | ✅ 完成 | `--soft-reset`默认启用, 默认checkpoint改为switch模型 |
| 更新全部文档 | ✅ 完成 | METHOD.md/MEMORY.md/PROGRESS.md |

### Soft-Reset 序列评估结果

| 序列 | Survival (Soft-Reset) | Survival (旧Blending) | 提升 |
|------|-----------------------|----------------------|------|
| walk→stand→walk | **85.6%** | 22.8% | +62.8pp |
| walk→run→walk→stand→run | **84.4%** | 0.0% | +84.4pp |
| dance→stand→dance | **96.2%** | 13.0% | +83.2pp |
| full-cycle (6段) | **94.8%** | 0.0% | +94.8pp |

### 步态分析结果

| 动作 | 关节 | 假肢均值 | 健侧均值 | RMSE | 相关性 |
|------|------|---------|---------|------|--------|
| Walk | Knee | 25.7° | 37.2° | 28.5° | 0.277 |
| Walk | Ankle | -0.1° | -0.8° | 1.7° | 0.565 |
| Walk | Hip | 1.7° | 0.8° | 4.7° | 0.478 |
| Run | Knee | 52.3° | 65.0° | 43.9° | -0.274 |
| Run | Ankle | 0.4° | 3.3° | 3.7° | 0.212 |
| Dance | Knee | 37.8° | 35.7° | 7.8° | 0.465 |
| Stand | Knee | 41.1° | 43.0° | 1.9° | 0.674 |
| Stand | Hip | 3.6° | 3.7° | 0.2° | 0.907 |

**分析**:
- ✅ **站立**: 假肢与健侧高度一致 (knee RMSE=1.9°, hip corr=0.907)，控制精确
- ⚠️ **行走**: 假肢膝关节幅度偏小 (25.7° vs 37.2°)，周期模式清晰但 ROM 不足
- ⚠️ **奔跑**: 假肢幅度偏小 (52.3° vs 65.0°)，相位差异明显 (corr=-0.274)
- ✅ **舞蹈**: 假肢与健侧幅度相近 (knee RMSE=7.8°)

图表输出: `outputs/gait_analysis/`

---

## 模型存储结构

```
outputs/checkpoints/
├── stage0/
│   ├── stage0_amp_walk_5050.pth       # Walk ✅ ep_len=287.2
│   ├── stage0_amp_run_1700.pth        # Run ✅ ep_len=287.5
│   ├── stage0_amp_dance_8000.pth      # Dance ✅ ep_len=295.3
│   └── stage0_amp_stand_900.pth       # Stand ✅ ep_len=299.0 (对称双脚)
├── stage1/
│   └── best.pth                        # Walk-only teacher ✅
├── stage1_multi/
│   └── best.pth                        # Multi-motion teacher ✅ ep_len=261.0
├── stage1_multi_switch/
│   └── best.pth                        # Multi-motion teacher w/ switching ✅ ep_len=242.6
├── stage1_multi_realistic/
│   └── best.pth                        # Realistic teacher (obs=16D) ✅ ep_len=243.9
├── stage2/
│   └── best.pth                        # Walk-only student ✅
├── stage2_multi/
│   └── best.pth                        # Multi-motion student ✅ reward=726.08
├── stage2_multi_switch/
│   └── best.pth                        # Legacy multi-motion student (obs=20D) ✅ reward=482.2
└── stage2_multi_realistic/
    └── best.pth                        # Realistic multi-motion student (obs=16D) ✅ reward=466.53
```

---

## 训练日志

_(按时间倒序记录关键训练事件)_

### 2026-03-09
- 创建多动作扩展项目计划
- 创建基础代码: constants_multi.py, proknee_multi_motion.py
- 创建训练脚本: train_stage1_multi.py, train_stage2_multi.py
- 创建评估脚本: evaluate_stage2_multi.py
- ✅ 生成站立参考动作 amp_humanoid_stand.npy (30帧, SkeletonMotion格式)
- 更新 MEMORY.md §14, METHOD.md 多动作架构章节
- ✅ 验证 Run Stage 0: ep_len=287.5
- ✅ 验证 Dance Stage 0: ep_len=295.3
- ✅ 训练 Stand Stage 0: ep_len=299.0 (3900 iters, stage0_amp_stand_3900.pth)
- ✅ 训练 Stage 1 Multi-Motion: best ep_len=261.0, loss=0.002, 8000 epochs, 2.21h
  - 4 motions (walk/run/dance/stand), 4096 envs, obs_dim=20
- ✅ 训练 Stage 2 Multi-Motion: best_reward=726.08, 300M agent steps
  - adapt_tconv 蒸馏完成, checkpoint saved to stage2_multi/best.pth
- ✅ 评估 Stage 2 Multi-Motion per-motion:
  - Walk: ep_len=282.9, survival=91.1%
  - Run: ep_len=280.7, survival=89.9%
  - Dance: ep_len=275.9, survival=87.5%
  - Stand: ep_len=276.4, survival=88.2%
- ✅ 实现 transition blending (blend_steps=15) in proknee_multi_motion.py
- ✅ 创建 evaluate_motion_sequences.py — 序列评估脚本
- ✅ 评估动作序列 (walk-stop-walk, walk-run-walk-stop-run, etc.)
  - Per-motion 优秀，序列 survival 受限于训练时未启用 intra-episode switching
  - Action diff < 0.07: 假肢动作转换非常平滑
- ✅ 更新 PROGRESS.md: 完整评估结果记录

### 2026-03-10
- ✅ 修复站立参考动作: 全 identity 四元数(ep_len=2) → walk frame116 slerp 对称化(ep_len=299)
  - 问题: 原 frame142 是单脚站立 (右膝~71° 左膝~31°)
  - 方案: 找最对称的双支撑帧(frame116), slerp 平均左右腿四元数
  - root_z=0.842, 双脚对称膝屈~46°
- ✅ 重训 Stand Stage 0: ep_len=299.0@iter900, 保存 stage0_amp_stand_900.pth
  - 注意: 训练曲线先升后降 (iter900 peak=299, iter4000 drop=162), 必须用 best checkpoint
- ✅ 添加 --switch-motion / --blend-steps 到 train_stage1_multi.py 和 train_stage2_multi.py
- ✅ 添加 --visualize 到 evaluate_motion_sequences.py
- ✅ 更新 MEMORY.md: §10 训练监控规范 (TensorBoard + 命令行检查 + 收敛标准)
- 🔄 Stage 1 Multi Switch 训练中:
  - epoch 3800/8000, ep_len=229, loss=0.005
  - 相比无 switch 版 (ep_len=261): ep_len 较低是预期的 (mid-episode switching 增加难度)
- ✅ Stage 1 Multi Switch 完成: 8000 epochs, 3.71h, best ep_len=242.6@epoch3383, loss=0.003
- ✅ Stage 2 Multi Switch 完成: 300M steps, best_reward=482.2@263M, ep_len=227, latent_mse=0.032
- ✅ 序列评估 (switch model):
  - walk→stand→walk: **22.8%** survival (旧 10.7%, ↑2x)
  - walk→run→walk→stand→run: 0.0% (长序列仍难)
  - dance→stand→dance: 13.0% (与旧 17.9% 略降)
  - full-cycle: 0.0%
  - Per-motion 表现持平: Walk 89.9%, Run 90.8%, Dance 88.0%, Stand 84.0%
  - 结论: switch 训练对短序列有改善，但 body policy 不连续是根本瓶颈
- ✅ 更新 MEMORY.md, METHOD.md, PROGRESS.md 全量文档

### 2026-03-10 (续)
- ✅ 修复可视化 Bug: `manual_motion_control` 标志防止 reset/step 覆盖动作设定
  - `proknee_multi_motion.py`: `_reset_envs()` 和 `step()` 在 manual_motion_control=True 时跳过随机分配
  - `evaluate_motion_sequences.py`: 启用 manual_motion_control, `--visualize` 自动 num_envs=1, 添加 `--sequence` 参数
- ✅ 修复后序列评估 (256 envs, switch model):
  - walk→stand→walk: 7.5%, dance→stand→dance: 22.6%, 长序列仍 0%
  - Per-motion: Walk 84.5%, Run 84%, Dance 85.7%, Stand 100%
- ✅ 创建步态分析脚本 `scripts/analyze_gait.py`
  - 录制 300 步各动作膝关节/踝关节/髋关节角度+力矩
  - 假肢 (LEFT, DOF 24/25) vs 健侧 (RIGHT, DOF 17/18) 对比
  - 生成 5 张分析图 + 统计摘要 + 原始数据 (.npz)
- ✅ 步态分析结果:
  - 站立: 假肢与健侧高度一致 (knee RMSE=1.9°, hip corr=0.907)
  - 行走: 假肢幅度偏小 (mean 25.7° vs 37.2°), 周期模式清晰
  - 奔跑: 假肢幅度偏小 (mean 52.3° vs 65.0°), 相位差异明显
  - 舞蹈: 假肢与健侧幅度相近 (knee RMSE=7.8°)
- ✅ 更新 MEMORY.md: 步态分析命令, 项目结构, 评估结果
- ✅ 更新 PROGRESS.md: Phase 7 全部记录

### 2026-03-11
- ✅ 诊断 body policy OOD 问题: 独立训练的 body policy 无法处理跨动作状态观测
  - 例: Stand policy 接收 walking 状态 (vel=1.5 m/s) → 输出不可预测 → agent 摔倒
  - 尝试方案: deceleration-first blending, velocity-aware blending, dual-policy live blending
  - 所有 blending 方案均因 OOD 问题无法可靠工作
- ✅ 实现 Soft-Reset 过渡策略 (`soft_reset_to_motion()`)
  - 核心: 切换动作时保留 XY 位置, 但将关节状态重置为新动作的 RSI
  - 彻底避免 OOD 问题: 每个 body policy 始终在自己的训练分布内工作
- ✅ Full-Cycle 测试: stand→walk→run→dance→stand→walk, 全部 6 段 0 次摔倒
- ✅ 序列评估 (64 envs, 10 trials, soft-reset):
  - walk→stand→walk: **85.6%** survival (↑ 从 22.8% blending)
  - walk→run→walk→stand→run: **84.4%** (↑ 从 0%)
  - dance→stand→dance: **96.2%** (↑ 从 13%)
  - full-cycle (6段): **94.8%** (↑ 从 0%)
- ✅ 更新 evaluate_motion_sequences.py:
  - 默认 checkpoint 改为 stage2_multi_switch/best.pth
  - 添加 `--soft-reset` / `--no-soft-reset` 参数
  - soft_reset_to_motion() 返回 fresh observations
- ✅ 更新 METHOD.md: 动作切换策略完整技术文档
- ✅ 更新 MEMORY.md: 命令、评估结果、可视化说明全量更新
- ✅ 更新 PROGRESS.md: Phase 8 完整记录

### 2026-03-11 (续) — Phase 9: Realistic 观测架构
- ✅ 实现 Realistic 假肢观测架构 (`motion_in_obs=False`)
  - obs(16D): 纯传感器; priv_info(117D): 含 motion onehot
  - 假肢不再知道动作类型，必须从运动模式推断
- ✅ 创建交互式可视化脚本 (`scripts/interactive_visualize.py`)
  - 键盘 W/R/D/S/Q 实时切换动作, episode 100000 步
- ✅ Stage 1 Multi Realistic 训练: 8000 epochs, 3.74h, best ep_len=243.9
- ✅ Stage 2 Multi Realistic 训练: 500M steps, ~1.5h, best reward=466.53
- ✅ 修复 `evaluate_stage2_multi.py`: progress_buf → info['finished_ep_len']
- ✅ Per-Motion 评估 (Realistic): Walk 79.6%, Run 79.7%, Dance 79.2%, Stand 80.0%
- ✅ 序列评估 (Realistic): walk-stop-walk 86.7%, full-cycle 94.8%
- ✅ Legacy 向后兼容验证通过 (walk 85.0%)
- ✅ 全量文档更新 (MEMORY.md, METHOD.md, PROGRESS.md)

---

## Phase 9: Realistic 假肢观测架构 + 交互式可视化 ✅ 完成

> 日期: 2026-03-11 (续)

### 目标

1. **假肢观测修正**: motion one-hot 从 obs 移到 priv_info，假肢不再知道动作类型
2. **交互式可视化**: 键盘 (W/R/D/S) 实时切换动作
3. **全量文档更新**

### 完成记录

#### 2026-03-11

- ✅ 修改 `constants_multi.py`:
  - 新增 `OBS_DIM_REALISTIC=16`, `STUDENT_PROPRIO_DIM_REALISTIC=16`
  - 新增 `TEACHER_PRIV_INFO_DIM_MULTI=117` (113 base + 4 onehot)
  - 详细文档注释说明两种模式
- ✅ 修改 `proknee_multi_motion.py`:
  - 新增 `motion_in_obs` 参数 (默认 False)
  - `_compute_base_obs()`: motion_in_obs=False 时返回 16D
  - `_compute_student_proprio()`: 对应调整
  - `_compute_priv_info()` 覆写: motion_in_obs=False 时追加 onehot → 117D
  - 维度覆盖: obs_dim/proprio_dim 根据模式动态设置
- ✅ 测试两种模式维度正确:
  - Realistic: obs(4,16), priv(4,117), hist(4,30,16) ✓
  - Legacy: obs(4,20), priv(4,113), hist(4,30,20) ✓
- ✅ 修改 `train_stage1_multi.py`:
  - 新增 `--no-motion-in-obs` 标志
  - 根据模式自动设置 obs_dim/priv_dim/proprio_dim
  - Checkpoint 保存 `motion_in_obs` 元数据
  - Realistic 模型保存到 `stage1_multi_realistic/`
- ✅ 修改 `train_stage2_multi.py`:
  - 同上，支持 `--no-motion-in-obs`
  - ProprioAdaptMulti 接受 obs_dim/priv_dim/proprio_dim 参数
  - Realistic 模型保存到 `stage2_multi_realistic/`
- ✅ 修改 `evaluate_motion_sequences.py`:
  - 自动从 checkpoint 元数据检测观测模式
  - 支持 `--no-motion-in-obs` / `--motion-in-obs` 强制指定
  - 传递正确维度到 env 和 model
- ✅ 创建 `scripts/interactive_visualize.py`:
  - Isaac Gym 键盘事件: W=Walk, R=Run, D=Dance, S=Stand, Q=Quit
  - 使用 soft_reset_to_motion() 切换动作
  - Episode 100000 步 (不自动 reset)
  - 倒地自动恢复
  - 支持 Realistic 和 Legacy 模式
- ✅ 全量更新 MEMORY.md:
  - 新增第 9 节: 两种观测模式说明
  - 更新项目结构 (新增 interactive_visualize.py)
  - 更新模型存储 (新增 realistic 路径)
  - 更新训练命令 (Realistic/Legacy 两种模式)
  - 更新评估命令 (auto-detect, 强制指定)
  - 新增交互式可视化命令和键盘控制表
- ✅ 全量更新 METHOD.md:
  - 新增 "Realistic 假肢观测架构" 一节
  - 新增 "交互式可视化" 一节
- ✅ Stage 1 Multi Realistic 训练完成:
  - 8000 epochs, 3.74h, best ep_len=243.9@epoch3925, loss=0.004
  - 输出: `outputs/checkpoints/stage1_multi_realistic/best.pth`
- ✅ Stage 2 Multi Realistic 训练完成:
  - 500M steps, ~1.5h, best reward=466.53, Latent MSE=0.040
  - 输出: `outputs/checkpoints/stage2_multi_realistic/best.pth`
- ✅ 修复 `evaluate_stage2_multi.py` Bug:
  - 问题: `progress_buf` 在 auto-reset 后已清零，读取为 0 → ep_len 全部 0
  - 修复: 改用 `info['finished_ep_len']` (在 reset 前保存)
- ✅ Per-Motion 评估 (Realistic 模型, 256 envs, 50 episodes):
  - Walk: ep_len=262.8, survival=79.6%
  - Run: ep_len=262.5, survival=79.7%
  - Dance: ep_len=262.0, survival=79.2%
  - Stand: ep_len=263.2, survival=80.0%
- ✅ 序列评估 (Realistic 模型, 64 envs, 10 trials, soft-reset):
  - walk→stand→walk: **86.7%** survival
  - walk→run→walk→stand→run: **84.1%**
  - dance→stand→dance: **94.7%**
  - full-cycle (6段): **94.8%**
- ✅ Legacy 模型向后兼容验证:
  - stage2_multi_switch/best.pth 自动检测为 legacy (obs=20D)
  - Walk: ep_len=271.4, survival=85.0% — 正常工作

### Realistic vs Legacy 对比

| 指标 | Realistic (16D obs) | Legacy (20D obs) | 说明 |
|------|-------------------|-----------------|------|
| S1 best ep_len | 243.9 | 242.6 | ≈ 持平 |
| S2 best reward | 466.53 | 482.2 | 略低 |
| Walk survival | 79.6% | 84.5% | -4.9pp |
| Run survival | 79.7% | 84.0% | -4.3pp |
| Dance survival | 79.2% | 85.7% | -6.5pp |
| Stand survival | 80.0% | 100% | -20pp |
| walk-stop-walk | 86.7% | 85.6% | ≈ 持平 |
| full-cycle | 94.8% | 94.8% | 持平 |

**分析**: Realistic 模型 per-motion survival 约低 5-20pp (符合预期，假肢无法直接获取动作类型)。
但序列切换表现几乎相同，说明 soft-reset 是关键而非观测模式。

---

### Phase 10: 平滑动作过渡 (Smooth Transition) ✅ 完成

| 任务 | 状态 | 说明 |
|------|------|------|
| 分析 soft_reset 不连续原因 | ✅ | 1帧内瞬移 28 DOF + 清零 proprio_hist |
| 尝试 PD-target-only 插值 | ❌ 失败 | 仅设 PD targets, root state 不受控 → 0.3% survival |
| 实现内联状态插值 | ✅ | N帧渐进物理状态强制设置 (root + DOF) |
| 更新 interactive_visualize.py | ✅ | 默认 smooth, `--hard-reset` 可选 |
| 更新 evaluate_motion_sequences.py | ✅ | 默认 smooth, `--soft-reset --no-smooth-transition` 可选 |
| 全序列评估 (smooth, 10帧) | ✅ | 80-92% survival |
| 向后兼容验证 | ✅ | Legacy soft-reset 仍正常 |
| 文档更新 | ✅ | MEMORY/METHOD/PROGRESS 全量更新 |

#### Smooth Transition 评估结果 (Realistic, 64 envs, 10 trials)

| 序列 | Soft-Reset | Smooth (10f) | Δ |
|------|-----------|-------------|-----|
| walk→stand→walk | 86.7% | 81.2% | -5.5pp |
| walk→run→walk→stand→run | 84.1% | 80.0% | -4.1pp |
| dance→stand→dance | 94.7% | 92.2% | -2.5pp |
| full-cycle | 94.8% | 90.2% | -4.6pp |
