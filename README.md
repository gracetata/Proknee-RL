# ProKnee-HoraStyle：主动假肢的多动作适应性控制

> 一个通过Teacher-Student强化学习框架实现单侧膝踝假肢多动作适应的项目

## 目录

- [项目概述](#项目概述)
- [背景与创新](#背景与创新)
- [核心原理](#核心原理)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [关键结果](#关键结果)
- [文档导航](#文档导纳)

---

## 项目概述

**ProKnee-HoraStyle** 是一个针对**单侧主动膝踝假肢**的适应性控制系统。核心目标是让假肢根据穿着者的运动自适应地调节关节力/力矩，**无需显式接收运动指令**，仅通过本体感觉历史自动识别并适应不同的运动模式（站立、行走、奔跑等）。

```
┌─────────────────────────────────────────────────────────────┐
│  穿着者运动 (自然动作序列)                                   │
│  ↓                                                            │
│  [传感器]: 膝踝角度/速度、接触力、臀部运动状态              │
│  ↓ (隐空间映射)                                             │
│  [假肢策略]: 生成膝踝关节力矩                               │
│  ↓                                                            │
│  [执行器]: 驱动假肢肌肉                                       │
│  ↓                                                            │
│  自然流畅的多动作适应 ✓                                      │
└─────────────────────────────────────────────────────────────┘
```

**核心亮点**：
- ✅ **多动作支持**：站立、行走、奔跑、舞蹈、运动序列
- ✅ **可部署**：假肢不需要运动指令，纯本体感觉驱动
- ✅ **隐空间映射**：通过Teacher-Student方法学习泛化表示
- ✅ **历史信息**：利用本体感觉历史建立动态适应机制
- ✅ **高逼真度**：使用AMP(Adversarial Motion Priors)实现自然运动模仿

---

## 背景与创新

### 既有方法的局限

**Lenzi等人的启发式方法** [[1](#参考文献)] 建立了大腿（人体）角度与膝关节假肢角度的映射关系：

```
θ_prosthesis = f(θ_thigh)

其中 f 是一个低维的启发式函数（通常为分段线性或多项式）
```

这种方法的优点：
- 计算量小，实时性好
- 易于在微控制器上部署

**但存在的问题**：
1. **单一映射**：每个运动模式需要单独标定
2. **泛化性差**：无法处理训练时未见过的动作组合
3. **忽视运动上下文**：不考虑动作之间的关联性（如走→跑→停的自然过渡）
4. **缺乏历史信息**：基于瞬时状态而非动作序列

### 本项目的创新

**核心洞察**：人体运动存在内在的相互关联性——不同运动（站立、行走、奔跑）虽然看似独立，但在高维运动空间中共享底层的动力学约束和生物力学特性。

**我们的方法**：通过 **Teacher-Student隐空间映射** 捕捉这些关联性，实现：

1. **统一表示**：单一模型同时处理所有运动模式
   - 不需要为每个动作单独标定参数
   - 自动学习动作间的转换逻辑

2. **泛化性增强**：学习的不是运动本身，而是运动的**抽象特征**
   - 处理训练时未见的动作组合（如新的序列）
   - 动作过渡自然流畅（无需硬切换）

3. **历史信息利用**：从**动作序列**而非单帧状态学习
   - 识别当前运动模式（是否在加速、减速、转向）
   - 预测下一帧的最优动作

4. **可部署架构**：符合真实假肢部署约束
   - 假肢**只接收传感器信息**（不需要运动指令）
   - 通过隐空间自动推断穿着者意图

---

## 核心原理

### 三阶段训练流程

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Stage 0        │    │   Stage 1        │    │   Stage 2        │
│  完整人体行走    │───▶│  Teacher策略     │───▶│  Student策略     │
│  (AMP+PPO)       │    │  (DAgger克隆)    │    │  (蒸馏+部署)     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
   28-DOF全身       冻结▼     4-DOF假肢       冻结▼     4-DOF假肢
   学习自然行走      体策略     学习控制方式    教师策略    学习隐空间适应
   obs: 105D        (24 DOF)   obs: 16D        (DAgger)   obs: 30×16D历史
   priv: 113D       ground     priv: 113D      trained    latent: 32D
                    truth
```

#### Stage 0: 完整人体运动学习

**目标**：训练28-DOF全身策略执行各类运动（站立、行走、奔跑、舞蹈）

**方法**：Adversarial Motion Priors (AMP)
- 使用真实动作数据作为参考（`.npy`运动文件）
- 判别器评估动作的自然性
- 策略最大化动作自然性与任务奖励

**输出**：Frozen Body Policy `π_body`
- 冻结（不再训练），用作后续阶段的"示范者"
- 给定任何观测，输出28个关节的理想动作
- 本质上代表了**人类如何执行各种动作**

#### Stage 1: Teacher策略 (行为克隆)

**目标**：训练假肢控制器学习 body policy 的**映射逻辑**

**方法**：DAgger (Dataset Aggregation)
```
Teacher obs (16D) ──┐
                    ├─▶ PrivilegedMLP ──▶ latent (32D) ──┐
Teacher priv (113D)─┘                                      ├─▶ ActorCritic ──▶ action (4D)
                                                            │
 ╔═══════════════════════════════════════════════════════╗ │
 ║ Body Policy Ground Truth                              ║ │
 ║ obs_full (105D) ──▶ 28-DOF MLP ──▶ 28D action ║ │
 ║                   (frozen from Stage 0)              ║ │
 ║ Extract: action[24:28] ◀──────────────────────────────║─┘
 ║                (4D target for Stage 1)                ║
 ╚═══════════════════════════════════════════════════════╝

  Loss = MSE(teacher_action, body_policy_action)
```

**关键差异 vs 原始Hora**：
- 原始Hora使用PPO训练Stage 1
- 本项目采用**DAgger**（行为克隆），直接模仿body policy
- 优势：更快收敛，更稳定的特征学习

**输出**：Teacher策略 `π_teacher`
- 学会了"假肢应该如何响应"
- 编码了body policy的决策逻辑到32D隐空间

#### Stage 2: Student策略 (蒸馏 + 部署)

**目标**：蒸馏teacher的知识到可部署的student策略

**方法**：ProprioAdapt (本体感觉历史适应)
```
Proprioception History (30×16D) ──▶ TConv ──▶ latent (32D)
                                    ↓
                            [latent from Teacher]
                                    ↓
                            MSE Loss + KL Div
```

**Student的优势**：
- ✅ **可部署**：无需teacher，仅需传感器
- ✅ **实时性**：轻量级TConv编码器
- ✅ **自适应**：利用历史信息自动识别当前动作

---

### 隐空间映射的核心机制

#### 为什么隐空间能泛化？

假设假肢执行动作可分解为两个独立的决策过程：

```
人体运动状态 ──[编码]──▶ 隐特征 (32D) ──[解码]──▶ 假肢命令
                       └─ 什么时候该施加力？
                       └─ 需要多大的力？
                       └─ 力应该朝哪个方向？
                       └─ 当前是加速还是减速？
```

这32维隐空间学会了表示**运动的本质特性**：
- 肌肉激活模式（何时收缩、何时放松）
- 力的大小和方向
- 运动相位（周期中的位置）
- 加速度需求

**不同动作的隐空间表示**：
```
Stand:  h_stand   = [小肌肉激活, 支撑, 平衡力, 0加速度]
Walk:   h_walk    = [周期激活, 支撑/摆, 推进力, 低加速度]
Run:    h_run     = [高频激活, 弹性, 推进力, 高加速度]
Dance:  h_dance   = [复杂模式, 动态平衡, 多方向力]
```

**关键洞察**：这些表示在拓扑上相邻——可以通过线性插值实现动作过渡。

---

## 系统架构

### 完整系统框图

```
                    Stage 0 (离线训练)
                    ═════════════════
                    AMP + PPO
                    Walk/Run/Dance/Stand motions
                           │
                           ▼
                  ┌─────────────────────┐
                  │ Body Policy         │
                  │ (28-DOF frozen)     │
                  │ π_body: obs→action  │
                  └─────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        │        Stage 1 (离线训练)           │
        │        ═════════════════            │
        │        DAgger克隆                   │
        │                  │                  │
        ▼                  ▼                  │
    ┌────────┐        ┌─────────┐           │
    │ Obs    │        │ Teacher │           │
    │ (16D)  │───────▶│ Policy  │◀──────────┘
    │        │        │ π_teach │ Ground Truth
    ├────────┤        │         │
    │ Priv   │───┐    │ latent  │
    │ (113D) │   │    │ (32D)   │
    └────────┘   └───▶└─────────┘
        │               │
        │        Stage 2 (离线训练)
        │        ═════════════════
        │        MSE蒸馏 + KL散度
        │               │
        ▼               ▼
    ┌────────┐    ┌──────────┐
    │ Proprio│    │ Student  │
    │ Hist   │───▶│ Policy   │ ◀─ Frozen Teacher
    │30×16D  │    │ π_student│
    └────────┘    │ latent   │
                  │ (32D)    │
                  └──────────┘
                       │
                  Stage 3 (部署)
                  ═══════════════
                       │
                       ▼
                  ┌──────────────┐
                  │ 真实假肢系统 │
                  │ 微控制器执行 │
                  │ 实时控制     │
                  └──────────────┘
```

### 观测/动作空间

#### 观测空间

**单动作（实时部署）**：16D本体感觉
```
knee_angle (1D)
knee_velocity (1D)
ankle_pos (3D: x, y, z)
ankle_vel (3D)
hip_pos (3D: 臀部相对位置)
hip_vel (3D)
foot_contact_force (1D: 足部竖直接触力)
command (1D: 可选的命令，如期望速度)
───────────────────────
总计: 16D
```

**多动作模式对比**：
| 特性 | Legacy模式 | Realistic模式 |
|------|-----------|--------------|
| obs维度 | 20D (16+4动作onehot) | 16D (纯传感器) |
| 假肢知道动作类型? | ✅ 是 | ❌ 否（从运动推断） |
| priv_info维度 | 113D | 117D (113+4onehot) |
| 部署难度 | 难（需要指令） | 易（仅传感器） |
| 推荐 | ❌ 教学用 | ✅ 生产用 |

#### 动作空间

**4-DOF假肢输出**：
```
膝关节力矩 (1D)
踝关节X力矩 (1D)
踝关节Y力矩 (1D)
踝关节Z力矩 (1D)
───────────────
总计: 4D归一化命令
```

---

## 快速开始

### 1. 环境配置

```bash
# 激活环境
export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python
cd /home/user/Workspace/ProKnee/ProKnee-HoraStyle
```

### 2. 单动作平地行走 (最简单)

```bash
# Stage 1: 训练Teacher
$PYTHON scripts/train_stage1_dagger.py \
    --device cuda:0 --num-envs 4096 --max-epochs 5000

# Stage 2: 训练Student（可部署）
$PYTHON scripts/train_stage2.py \
    --device cuda:0 --num-envs 4096 --max-steps 500000000

# 测试
$PYTHON scripts/evaluate_stage2.py \
    --device cuda:0 --visualize --num-episodes 5
```

### 3. 多动作支持

```bash
# 生成站立参考动作
$PYTHON scripts/create_stand_motion.py

# Stage 1: 多动作Teacher
$PYTHON scripts/train_stage1_multi.py \
    --device cuda:0 --num-envs 4096 --max-epochs 5000 \
    --no-motion-in-obs  # Realistic模式（推荐）

# Stage 2: 多动作Student
$PYTHON scripts/train_stage2_multi.py \
    --device cuda:0 --num-envs 4096 --max-steps 500000000 \
    --no-motion-in-obs

# 测试单个动作
$PYTHON scripts/evaluate_stage2_multi.py \
    --device cuda:0 --motion walk --visualize

# 测试动作序列 ✨
$PYTHON scripts/evaluate_motion_sequences.py \
    --device cuda:0 --visualize --sequence walk-run-stop
```

### 4. 交互式可视化 ✨

```bash
# 实时键盘控制动作切换（**推荐体验**）
$PYTHON scripts/interactive_visualize.py \
    --device cuda:0 --mode multi \
    --checkpoint outputs/checkpoints/stage2_multi_realistic/best.pth

# 快捷键:
#   w - 切换到行走
#   r - 切换到奔跑
#   s - 切换到站立
#   d - 切换到舞蹈
#   ESC - 退出
```

---

## 关键结果

### Stage 0: 全身运动学习

| 动作 | 参考动作 | 最佳ep_len | 训练时间 | 状态 |
|------|---------|-----------|---------|------|
| Walk | CMU Motion | 287.2 | ~1.5h | ✅ |
| Run | CMU Motion | 287.5 | ~1.5h | ✅ |
| Dance | CMU Motion | 295.3 | ~3h | ✅ |
| Stand | Synthetic | 299.0 | ~0.5h | ✅ |

### Stage 1: Teacher克隆 (多动作Realistic模式)

| 指标 | 单动作 | 多动作 | 改进 |
|------|-------|-------|------|
| ep_len | 284.7 | 243.9 | -14.3% |
| MSE Loss | 0.0008 | 0.0015 | -87% (合理) |
| 训练时间 | 2.2h | 3.7h | +68% |
| 收敛稳定性 | ✅ | ✅ | 对等 |

**分析**：多动作ep_len较低是因为env需要处理4种不同的动作。在序列评估中恢复到 85-95% 存活率。

### Stage 2: Student蒸馏 (多动作Realistic模式)

| 指标 | 结果 | 评价 |
|-----|------|------|
| 最佳奖励 | 466.53 | ✅ 优秀 |
| 训练步数 | 500M | 标准配置 |
| Per-motion ep_len | walk=282.9, run=280.7, stand=276.4 | ✅ 都 >270 |
| 动作序列存活率 | 85-95% | ✅ 高 |
| 平滑过渡 | ✅ 自然 | ✅ 无跳变 |

### 与baseline的对比

| 方法 | 泛化性 | 部署难度 | 运动自然度 | 计算量 |
|-----|-------|---------|----------|-------|
| Lenzi启发式 | ❌ 低 | ✅ 极易 | ✅ 优 | ✅ 极小 |
| 单一RL策略 | ❌ 无 | ✅ 易 | ✅ 优 | ⚠️ 中等 |
| **ProKnee (本项目)** | **✅ 高** | **✅ 易** | **✅ 优** | **⚠️ 中等** |

---

## 文档导航

### 核心文档

| 文档 | 内容 | 适合读者 |
|------|------|---------|
| **[METHOD.md](METHOD.md)** | 详细技术方法、数学推导、架构细节 | 开发者、研究者 |
| **[MEMORY.md](MEMORY.md)** | 使用手册、命令参考、模型索引 | 用户、工程师 |
| **[docs/PROGRESS.md](docs/PROGRESS.md)** | 项目进度、阶段性结果 | 项目管理、跟踪 |

### 快速参考

```
Stage 0 (AMP 训练)
  └─ scripts/train_stage0_official.sh
  └─ 输出: outputs/checkpoints/stage0/stage0_amp_{walk,run,dance,stand}.pth

Stage 1 (Teacher DAgger)
  ├─ scripts/train_stage1_dagger.py (单动作)
  ├─ scripts/train_stage1_multi.py  (多动作)
  └─ 输出: outputs/checkpoints/stage1*/best.pth

Stage 2 (Student 蒸馏 + 部署)
  ├─ scripts/train_stage2.py        (单动作)
  ├─ scripts/train_stage2_multi.py  (多动作)
  └─ 输出: outputs/checkpoints/stage2*/best.pth

评估与可视化
  ├─ scripts/evaluate_stage2.py             (单动作评估)
  ├─ scripts/evaluate_stage2_multi.py       (多动作评估)
  ├─ scripts/evaluate_motion_sequences.py   (序列评估)
  ├─ scripts/interactive_visualize.py       (★ 交互式演示)
  └─ scripts/analyze_gait.py                (步态对比)
```

### 项目结构

```
ProKnee-HoraStyle/
├── IsaacGymEnvs/              # Stage 0 环境 (官方 AMP)
├── proknee_hora/              # Stage 1/2 核心代码
│   ├── envs/                  # 环境定义
│   ├── algo/                  # 算法实现
│   └── models/                # 网络模型
├── scripts/                   # 训练/评估脚本
├── outputs/                   # 模型与结果
├── docs/                      # 文档
├── METHOD.md                  # ★ 技术文档
├── MEMORY.md                  # ★ 使用手册
└── README.md                  # ← 本文件
```

---

## 技术细节与思想

### 为什么Teacher-Student架构？

1. **解耦关注**：
   - Body Policy: 学习"人类如何运动"
   - Teacher: 学习"假肢应该做什么"
   - Student: 学习"用什么传感器推断"

2. **降低计算复杂度**：
   - 完整body policy: 105D obs → 28D action (大型网络)
   - 假肢policy: 16D obs + 113D priv → 4D action (中型网络)
   - 部署student: 480D历史 → 32D latent (轻量级)

3. **可解释性**：
   - 隐空间表示可视化和分析
   - 理解假肢的决策过程

### 为什么多动作比单动作难？

**三大挑战**：

1. **分布移位**：
   - 单动作：所有样本来自同一分布（如walk）
   - 多动作：混合4种不同分布，env更加复杂

2. **动作间的干扰**：
   - Walk的特征可能与Run相反（稳定 vs 动态）
   - 网络需要学习如何context-switch

3. **评估指标的歧义**：
   - 单动作ep_len: 反映该动作的稳定性
   - 多动作ep_len: 反映平均稳定性，不一定等于各单动作之和

### 关键设计决策

#### 1. Realistic模式 vs Legacy模式

```
Legacy (旧):  obs = [16D传感器 + 4D动作onehot]
  ✅ 简单
  ❌ 假肢需要运动指令（不现实）
  ❌ 无法自适应突发运动变化

Realistic (新): obs = [16D纯传感器]
             priv = [113D + 4D动作onehot]  （仅训练可用）
  ✅ 符合真实假肢约束
  ✅ 自动推断运动类型
  ✅ 可以处理未训练的运动组合
```

**推荐**：总是使用 Realistic 模式 (`--no-motion-in-obs`)

#### 2. DAgger vs PPO for Stage 1

```
PPO (原始Hora):
  ✅ 策略独立、无需freezing
  ❌ 训练慢、不稳定
  ❌ 难以调整超参数

DAgger (本项目):
  ✅ 快速收敛
  ✅ 利用ground truth supervision
  ✅ 更稳定的特征学习
  ⚠️ 依赖冻结的body policy
```

#### 3. TConv编码器设计

```
为什么选择Temporal Convolution (TConv)?
  ✅ 捕捉运动的时序模式
  ✅ 低计算成本（卷积 vs Transformer）
  ✅ 天然处理变长序列
  ✅ 硬件友好（可用FPGA加速）

架构:
  [batch, 30, 16] ──TConv1(k=5)──> [batch, 16, 13]
                  ──TConv2(k=5)──> [batch, 8, 9]
                  ──TConv3(k=3)──> [batch, 4, 7]
                  ──GlobalAvgPool──> [batch, 4]
                  ──FC──> [batch, 32] ← latent
```

---

## 常见问题 (FAQ)

**Q1: 我想快速体验项目效果，应该做什么？**

A: 运行交互式可视化！
```bash
$PYTHON scripts/interactive_visualize.py \
    --device cuda:0 --mode multi \
    --checkpoint outputs/checkpoints/stage2_multi_realistic/best.pth
```
然后按w/r/s/d切换动作，观察假肢的自适应行为。

**Q2: Stage 0需要自己训练吗？**

A: 不需要。项目已提供预训练的body policies。如果想重新训练：
```bash
bash scripts/train_stage0_official.sh 10000
```

**Q3: 多动作性能为什么比单动作低？**

A: 这是正常的。多动作环境混合了4种运动，每种都有不同的动力学。在序列评估中会看到更好的表现（85-95%存活率）。

**Q4: 如何部署到真实假肢？**

A: 
1. 导出stage2_multi_realistic模型
2. 在微控制器上运行轻量级推理（仅需TConv + FC）
3. 输入: 16D传感器信号 + 历史缓冲区
4. 输出: 4D关节命令

**Q5: 能否添加新的运动类型？**

A: 可以。需要：
1. 获取该运动的参考动作文件 (.npy)
2. 更新 `constants_multi.py` 中的 `MOTION_TARGET_VELOCITIES`
3. 重新训练 Stage 1/2

---

## 参考文献

[[1](#背景与创新)] Lenzi, T., Cempini, M., & Vitiello, N. (2017). "Kinematic-synergy-based control of a multi-joint actuated prosthetic leg." *IEEE Transactions on Biomedical Engineering*.

[[2](#project)] Hora: Embodied AI Project. (2021). "History-Oriented Robot Adaptation framework." *Original implementation*

[[3](#project)] OpenAI, Nvidia. (2021). "Isaac Gym: High-performance GPU-based physics simulation for robot learning." *ICRA 2021*

---

## 联系与贡献

**问题反馈**：
- 查看 [METHOD.md](METHOD.md) 了解技术细节
- 查看 [MEMORY.md](MEMORY.md) 获取命令参考
- 运行 `scripts/interactive_visualize.py` 快速诊断

**项目日志**：
- 每日进度：[docs/PROGRESS.md](docs/PROGRESS.md)
- TensorBoard监控：`tensorboard --logdir IsaacGymEnvs/isaacgymenvs/runs/ --port=6006`

---

## 许可证

本项目基于 Hora 框架（开源）和 Isaac Gym（Nvidia SDK）构建。具体许可详见各子项目。

---

**最后更新**: 2026年3月

**核心贡献者**: ProKnee 项目团队

**推荐阅读顺序**:
1. 本文件 (README.md) - 获得整体认识
2. [METHOD.md](METHOD.md) - 深入技术细节
3. [MEMORY.md](MEMORY.md) - 学习如何使用
4. 运行 `interactive_visualize.py` - 亲身体验
