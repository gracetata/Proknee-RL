# Claude 对 ProKnee-HoraStyle 的理解

> 写于 2026-03-18，基于对 README、METHOD、IMPLEMENTATION_SUMMARY、PROGRESS 等文档的阅读

---

## 一句话总结

用 Teacher-Student 强化学习框架，训练一个只需要 16 维传感器输入就能自适应多种运动模式的主动假肢控制器——不需要告诉假肢"你现在在走路"，它自己能感知出来。

---

## 这个项目在解决什么问题

传统方法（Lenzi 等人的启发式映射）的核心局限是：**一个运动模式，一套参数**。走路有走路的标定，跑步有跑步的标定，动作一切换就需要硬编码的模式识别机制。更根本的问题是，这类方法基于瞬时状态而非运动历史，无法感知"我正在加速"或"我即将停下"这类上下文信息。

本项目的回答是：**用隐空间替代启发式函数**。让网络自己学出一个 32 维的表示，这个表示隐式地编码了肌肉激活模式、力的方向与大小、运动相位、加速度需求——换句话说，编码了"当前运动状态的本质"，而不是"当前运动的标签"。

---

## 三阶段训练管道

### Stage 0：人体怎么动（28-DOF 全身策略）

用 Adversarial Motion Priors (AMP) + PPO，基于真实动作捕捉数据，训练一个 28 自由度的全身策略。训练完成后**冻结**，不再更新。它的角色是：**示范者**——给定任何观测，告诉你"一个正常人体应该怎么动"。

目前已有 4 个单动作 body policy（walk/run/dance/stand），Phase 11 正在训练一个**统一的速度条件化 body policy**（输入 velocity_cmd，单一网络覆盖所有速度范围）。

### Stage 1：假肢应该怎么跟（Teacher，DAgger 行为克隆）

**输入：** 16D 本体感觉观测 + 113D 特权信息（训练时才有）
**输出：** 4D 假肢关节命令（膝关节力矩 + 踝关节三轴力矩）

核心逻辑是 DAgger：让 Teacher 直接模仿 Stage 0 的输出（取 action[24:28]，即左侧假肢关节）。损失函数是 MSE，目标是让 Teacher 的动作尽量贴近"全身策略如果有一条好腿会怎么动"。

中间会生成一个 **32D 隐向量**，这是整个架构的信息瓶颈，也是知识蒸馏的桥梁。

### Stage 2：没有特权信息时怎么办（Student，ProprioAdapt 蒸馏）

**输入：** 30 帧 × 16D = 480D 本体感觉历史
**输出：** 32D 隐向量（与 Teacher 的隐向量对齐）

用一个轻量级的 Temporal Convolution 编码器，从传感器历史中提取隐空间表示，然后用同样的 Actor 输出动作。损失 = MSE(student_latent, teacher_latent) + KL 散度。

**这就是可以部署的东西**：只需要传感器，不需要运动指令，不需要全身状态。

---

## 两种观测模式

| 模式 | obs 维度 | 假肢知道动作类型？ | 部署可行性 |
|------|---------|-----------------|-----------|
| Legacy | 20D（16D传感器 + 4D one-hot） | ✅ 知道 | ❌ 需要外部指令 |
| Realistic | 16D（纯传感器） | ❌ 不知道，自己推断 | ✅ 可部署 |

priv_info 在 Realistic 模式下是 117D（113D 基础 + 4D 动作 one-hot），但这只在训练时用，部署时不存在。

---

## 动作切换的演进历史

这条线索贯穿了 Phase 4 到 Phase 10，非常能说明项目思路的演进：

1. **Blending（Phase 4-6）**：在动作切换时做线性插值过渡。失败原因：每个 body policy 只在自己的训练分布内可靠，当它收到另一种运动的状态观测时（OOD），输出不可预测，直接摔倒。
2. **Switch 训练（Phase 6）**：在训练中加入 mid-episode 切换。walk→stand→walk survival 从 10.7% → 22.8%，但长序列仍然 0%。根因没解决。
3. **Soft-Reset（Phase 8）**：切换动作时，保留 XY 位置，但把关节状态重置为新动作的 Reference State Initialization (RSI)。彻底规避 OOD 问题，因为每个 body policy 始终在自己的训练分布内。结果：全序列 survival 从 0% 跳到 85-96%。**这是关键突破**。
4. **Smooth Transition（Phase 10）**：在 soft-reset 基础上加 N 帧物理插值，让视觉更自然。代价是约 4pp survival 下降，但仍有 80-92%。

---

## Phase 11 的战略意义

Phase 0-10 的架构是 **per-motion**：4 个 body policy、4 套训练、切换时靠 soft-reset 这个"技术性补丁"。

Phase 11 想做的是根本性改变：

```
per-motion: π_walk, π_run, π_dance, π_stand（4个独立模型）
                ↓
unified:    π_unified(obs, v_cmd)（1个速度参数化模型）
```

单一 body policy 接受速度命令（0 = 站立，1.0 = 行走，2.5 = 奔跑），在连续速度空间上都能输出合理动作。这样：
- 不再需要硬切换，速度连续变化动作自然过渡
- 理论上 soft-reset 可以退休（因为只有一个 policy，不存在 OOD）
- Stage 2 Student 只需要从历史中推断"当前速度"，而不是"当前动作标签"

当前状态：unified Stage 0 在训练中，ep_len=254.93 已达目标，等待收敛稳定。

---

## 关键技术选择的理由

**为什么用 DAgger 而不是 PPO 训练 Stage 1？**
DAgger 有 ground truth supervision（直接对标 body policy 的输出），收敛更快、更稳定。PPO 需要自己探索奖励空间，对于这种"模仿已知示范者"的任务是不必要的绕路。

**为什么用 TConv 而不是 Transformer？**
计算成本。TConv 在 30 帧历史上的推理延迟约 10ms，总端到端延迟约 35-40ms，在可接受范围内。Transformer 的计算量对嵌入式部署不友好。

**为什么隐空间是 32D？**
足够表达运动的核心动态特征（相位、力、加速度需求），又不过大导致蒸馏困难。这是经验性选择，来自原始 Hora 框架。

---

## 尚未解决的问题

1. **步态质量**：行走时假肢膝关节幅度偏小（25.7° vs 健侧 37.2°），奔跑时相位差异明显（相关性 -0.274）。这说明隐空间还没有完全捕捉到周期性运动的时序结构。
2. **unified Stage 1/2 未经验证**：Phase 12-13 是否能达到 per-motion 版本的性能，还是未知数。速度连续化增加了学习难度。
3. **真机验证（Phase 15）**：所有结果都在 Isaac Gym 仿真中。Sim-to-real gap 尚未被触碰。

---

## 我认为最值得关注的地方

整个项目最优雅的设计是**特权信息的使用方式**：训练时用全身状态 + 动作标签（113D/117D priv），部署时完全丢弃。Student 被迫从传感器历史中"猜"出这些信息对应的隐向量，而不是直接得到它们。这个约束逼出了泛化能力。

这和人类本体感觉的工作方式高度吻合：你的大脑不需要有人告诉你"你在跑步"，它从肌肉、关节、前庭的信号流中自动推断出来，并据此调节肌肉力量。ProKnee 的 Student policy 在做同样的事。

---

## 代码成熟度分类

> 更新于 2026-03-20，用于指导开发时哪些代码可以修改

### 🔒 成熟代码（禁止修改）

这些模块经过验证，是项目的稳定基础：

| 文件路径 | 功能 | 说明 |
|---------|------|------|
| `proknee_hora/envs/proknee_base.py` | 单动作环境基类 | Stage 1/2 的仿真核心 |
| `proknee_hora/envs/constants.py` | 单动作常量 | OBS_DIM=16, LATENT_DIM=32 等 |
| `proknee_hora/algo/models/actor_critic.py` | ProKneePolicy | Teacher/Student 网络架构 |
| `proknee_hora/algo/models/adaptation.py` | PrivilegedMLP + TConv | 隐空间编码器 |
| `proknee_hora/algo/models/running_mean_std.py` | 在线归一化 | 观测预处理 |
| `scripts/train_stage1_dagger.py` | 单动作 Stage 1 | DAgger 训练脚本 |
| `scripts/train_stage2.py` | 单动作 Stage 2 | 蒸馏训练脚本 |
| `scripts/evaluate_stage2.py` | 单动作评估 | 步态分析 |

**Checkpoint（禁止覆盖）：**
```
outputs/checkpoints/
├── stage0/stage0_amp_walk_5050.pth      # Walk body policy
├── stage0/stage0_amp_run_1700.pth       # Run body policy
├── stage0/stage0_amp_dance_8000.pth     # Dance body policy
├── stage0/stage0_amp_stand_900.pth      # Stand body policy
├── stage1/best.pth                       # 单动作 Teacher
├── stage2/best.pth                       # 单动作 Student
├── stage1_multi_realistic/best.pth       # 多动作 per-motion Teacher
└── stage2_multi_realistic/best.pth       # 多动作 per-motion Student
```

### 🔧 维护中代码（可扩展，谨慎修改）

这些模块仍在迭代，可以添加功能但要保持向后兼容：

| 文件路径 | 功能 | 当前状态 |
|---------|------|---------|
| `proknee_hora/envs/proknee_multi_motion.py` | 多动作环境 | 支持 4 种动作切换 |
| `proknee_hora/envs/constants_multi.py` | 多动作常量 | Realistic/Legacy 模式 |
| `scripts/train_stage1_multi.py` | 多动作 Stage 1 | 支持 `--no-motion-in-obs` |
| `scripts/train_stage2_multi.py` | 多动作 Stage 2 | 支持 `--no-motion-in-obs` |
| `scripts/interactive_visualize.py` | 键盘交互可视化 | W/R/D/S 离散切换 |
| `scripts/evaluate_motion_sequences.py` | 序列评估 | 多动作性能测试 |

### 🚧 开发中代码（Phase 11-13）→ ✅ 已完成

**Unified 速度控制系统已完成训练和验证：**

| 文件路径 | 功能 | 状态 |
|---------|------|------|
| `IsaacGymEnvs/isaacgymenvs/tasks/humanoid_amp_unified.py` | Unified Stage 0 环境 | ✅ 训练完成 ep_len=254.93 |
| `proknee_hora/envs/proknee_unified.py` | Unified Stage 1/2 环境 | ✅ 完成 |
| `proknee_hora/envs/constants_unified.py` | Unified 常量定义 (114D priv) | ✅ 完成 |
| `scripts/train_stage1_unified.py` | Unified Stage 1 DAgger | ✅ 完成 ep_len=268.4 |
| `scripts/train_stage2_unified.py` | Unified Stage 2 蒸馏 | ✅ 完成 reward=689.83 |
| `scripts/interactive_unified.py` | 键盘速度控制 (↑/↓/0-9) | ✅ 完成 |

**Unified Checkpoint（2026-03-21 训练完成）：**
```
outputs/checkpoints/
├── stage0/stage0_unified_1800.pth          # Unified body policy
├── stage1_unified/best.pth                 # Unified Teacher (ep_len=268)
└── stage2_unified/best.pth                 # Unified Student (reward=689)
```

---

*Claude Sonnet 4.6 | 2026-03-18*
*更新: Claude Opus 4.5 | 2026-03-20 — 添加代码成熟度分类*
*更新: Claude Opus 4.5 | 2026-03-21 — Unified 系统训练完成*
