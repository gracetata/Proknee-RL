# ProKnee-HoraStyle: 三阶段训练方法

## 概述

本项目实现单侧膝关节+踝关节主动假肢的 Teacher-Student 强化学习控制策略，严格基于 Hora (History-Oriented Robot Adaptation) 架构。

```
Stage 0 ──────────→ Stage 1 ─────────→ Stage 2
完整人体AMP行走      Teacher(DAgger)     ProprioAdapt(本体感觉历史)
28-DOF全身策略       4-DOF假肢控制        4-DOF假肢控制(可部署)
PPO + AMP            DAgger行为克隆       纯MSE蒸馏 (无PPO)
```

---

## 假肢配置

基于 `amp_humanoid.xml` 的 28-DOF 人体模型，**左腿**为假肢侧：

| 关节 | DOF索引 | 角色 |
|------|---------|------|
| LEFT_KNEE | 24 | **主动假肢** (训练目标) |
| LEFT_ANKLE_X/Y/Z | 25, 26, 27 | **主动假肢** (训练目标) |
| LEFT_HIP_X/Z/Y | 21, 22, 23 | Stage 0 冻结策略控制 |
| 其余 20 DOF | 0-20 | Stage 0 冻结策略控制 |

---

## Hora 架构核心原则 ★

**关键: obs = 部署可用的本体感觉, priv_info = 仅仿真可用的特权信息**

| 组件 | 维度 | 部署可用? | 说明 |
|------|------|-----------|------|
| **obs** | 16D | ✅ | knee(2) + ankle(6) + hip(6) + foot_fz(1) + command(1) |
| **priv_info** | 113D | ❌ | full_body_obs(105D) + GRF(6D) + contacts(2D) |
| **latent** | 32D | — | 压缩表示 |
| **proprio_hist** | 30×16D | ✅ | obs 的历史帧 |
| **action** | 4D | — | 膝关节 + 踝关节×3 |

**Teacher 和 Student 共享同一个 obs_dim (16D)**, 使 backbone 可以直接共享/冻结。
这与 Hora 原文完全一致: Hora 的 obs (96D) 也是纯本体感觉。

---

## Stage 0: 完整人体 AMP 行走

### 目标

训练 28-DOF 全身自然行走策略 (frozen body policy)。

### 方法: AMP + PPO

```
总奖励 = 0.0 × r_task + 1.0 × r_style
r_style = max(0, 1 - 0.25 × (D(s, s') - 1)²)
```

### 输入/输出

- **观测** (105D): root state + DOF pos/vel + key body pos
- **动作** (28D): 全部关节归一化位置目标

### 输出: 28-DOF body policy checkpoint

---

## Stage 1: Teacher 策略 (DAgger 行为克隆)

### 目标

训练假肢控制器 (4 DOF: 膝+踝)，通过 DAgger 行为克隆 body policy 的对应关节动作。

### 方法: DAgger

不同于原始 Hora 使用 PPO 训练 Stage 1，本项目使用 DAgger (Dataset Aggregation):
- **Body Policy 提供 ground truth**: 对每个状态，body policy 给出 4 个假肢关节的理想动作
- **Teacher 自己 rollout**: 环境中执行的是 teacher 自己的动作 (不是 body policy 的)
- **监督学习**: MSE(teacher_action, body_policy_action)
- **DAgger 关键**: teacher 从自己的状态分布中学习 → 解决分布漂移

### 架构

```
obs (16D) ──────────────────┐
                            │ cat → [48D]
priv_info (113D) ──→ PrivilegedMLP ──→ latent (32D) ──┘
                     [256,128] + tanh         │
                                        ┌─────▼─────┐
                                        │ ActorCritic│
                                        │[256,128,64]│
                                        │ actor → 4D │ ← MSE(output, body_policy_target)
                                        │ critic →1D │
                                        └────────────┘

    ┌──────────────────────────────────────┐
    │ Frozen Body Policy (24 DOF)          │
    │ 输入 full_body_obs(105D)             │
    │ 输出 24 个关节的动作 + 4D target     │
    └──────────────────────────────────────┘
```

### DAgger 训练流程

```python
for epoch in range(max_epochs):
    # 1. Teacher 产生动作
    teacher_action = policy(obs, priv_info)  # 通过 PrivMLP + backbone + actor
    
    # 2. Body policy 提供 ground truth
    body_action = body_policy(full_body_obs)
    target = body_action[ACTIVE_PROSTHESIS_JOINTS]  # 4D
    
    # 3. 监督损失
    loss = MSE(teacher_action, target)
    loss.backward()
    
    # 4. 添加探索噪声并执行
    exec_action = teacher_action + noise  # noise 从 0.3 退火到 0
    env.step(exec_action)
```

### 特权信息 (113D) = full_body_obs (105D) + GRF (6D) + contacts (2D)

> 注: full_body_obs 也是 body policy 的输入。部署时不可用 (需要全身传感器)。

---

## Stage 2: ProprioAdapt (Hora 风格蒸馏) — 纯 MSE ★

### 目标

训练 adapt_tconv 使其从本体感觉历史预测 teacher 的 latent。

### 核心: 无 PPO, 无 rollout buffer, 纯监督

```
Step 1: 加载 Stage 1 模型 (包含训练好的 PrivMLP, backbone, actor, critic)
Step 2: 添加 ProprioAdaptTConv 模块 (随机初始化)
Step 3: 冻结所有参数, 仅 adapt_tconv 可训练 (~33K params)

训练循环 (每个环境步 = 一次梯度更新):
  obs_dict = env.step(action)
  
  z_student = adapt_tconv(normalize(proprio_hist))   # 可训练
  z_teacher = priv_mlp(priv_info)                     # 冻结
  
  loss = MSE(z_student, z_teacher.detach())
  loss.backward()  # 仅 adapt_tconv 获得梯度
  
  action = frozen_actor(obs + z_student)  # backbone 冻结但用 student latent
```

### 架构

```
proprio_hist (30×16D) ──→ sa_mean_std (归一化)
                              │
                    ┌─────────▼──────────┐
                    │ ProprioAdaptTConv   │ ← 唯一可训练部分
                    │ Linear(16→32)      │
                    │ Conv1d(k=9,s=2)    │
                    │ Conv1d(k=5,s=1)    │
                    │ Conv1d(k=3,s=1)    │
                    │ Linear(→32) + tanh │
                    └─────────┬──────────┘
                              │ z_student (32D)
                              │
                              │  z_teacher (32D, frozen) ← PrivMLP(priv_info)
                              │     │
                              │  MSE Loss
                              │
obs (16D) ────────────────┐   │
                          │   │
                          ├───▼───┐
                          │Backbone│ ← 冻结 (来自 Stage 1)
                          │[256,128│
                          │  ,64]  │
                          ├────────┤
                          │ Actor  │ → action (4D) ← 冻结
                          │ Critic │ → value (1D)  ← 冻结
                          └────────┘
```

### 归一化 (关键)

| 模块 | 作用 | 训练 |
|------|------|------|
| running_mean_std | 归一化 obs (16D) | **identity** (Stage 1 DAgger 未使用) |
| sa_mean_std | 归一化 proprio_hist (30, 16) | **可训练** (新初始化) |

### 实现

- **代码**: `proknee_hora/algo/proprio_adapt.py` → ProprioAdapt
- **脚本**: `scripts/train_stage2.py`
- **需要**: Stage 0 + Stage 1 checkpoints
- **参考**: `hora/hora/algo/padapt/padapt.py` (原始 Hora 实现)

---

## Reward 设计

### Stage 0 (AMP)

纯风格奖励 (判别器驱动)。

### Stage 1 (DAgger)

无 reward (纯监督学习)。环境 reward 仅用于 ep_len 统计。

### Stage 2

Stage 2 不使用 reward 训练 (纯监督)。Reward 仅用于监控和保存 best model。

---

## 步态分析

训练完成后，使用 `scripts/evaluate_stage2.py` 进行评估和步态分析:

1. **无头评估**: 统计 ep_len, 存活率
2. **步态周期检测**: 基于足底接触力 (heel-strike) 分割步态周期
3. **关节角度曲线**: 每个步态周期重采样到 100 点后平均
4. **输出**: PNG 图 + CSV 数据
5. **实时可视化**: Isaac Gym 渲染 + matplotlib 关节角度实时曲线

---

## 验收标准

| Stage | 指标 | 标准 |
|-------|------|------|
| Stage 0 | Episode length (GPU) | ≥ 280步 |
| Stage 1 | Episode length (GPU) | ≥ 280步 |
| Stage 2 | Episode length (GPU) | ≥ 270步 |
| Stage 2 | Latent MSE | ≤ 0.01 |

---

## 多动作扩展 (Multi-Motion Extension)

> 代码在 `proknee_hora/envs/proknee_multi_motion.py`，不修改原始单动作 pipeline。

### 设计目标

让同一个假肢策略根据动作指令执行不同运动：行走、奔跑、跳舞、站立，并支持动作间的平滑切换。

### 核心思路

**为什么用独立 body policy 而非统一 policy？**

单动作 Stage 0 各自训练效果好（ep_len > 280），训练成本低。
如果用一个统一 body policy 做所有动作，需要大量训练时间且质量可能下降。
因此，多动作方案保持每种动作各自的 Stage 0，在上层通过 motion_id 切换。

```
                    Stage 0 (per-motion)           Stage 1 Multi            Stage 2 Multi
                ┌─────────────────────────┐   ┌─────────────────────┐   ┌───────────────────────┐
motion_id ─────►│ walk_policy  (28-DOF)   │   │                     │   │                       │
               │ run_policy   (28-DOF)   │   │  obs(20D)           │   │  proprio_hist(30×20D) │
               │ dance_policy (28-DOF)   │──►│  + priv_info(113D)  │──►│  → adapt_tconv        │
               │ stand_policy (28-DOF)   │   │  → 4D prosthesis    │   │  → 32D latent → 4D    │
               └─────────────────────────┘   └─────────────────────┘   └───────────────────────┘
                 按 env 分配不同 body policy     Teacher 学习所有动作      Student 从历史推断动作
```

### 观测空间扩展

在原始 16D 本体感觉基础上，添加 4D one-hot 编码表示当前动作类型：

```
obs_multi (20D) = obs_base (16D) + motion_onehot (4D)
                                    ├─ [1,0,0,0] = walk
                                    ├─ [0,1,0,0] = run
                                    ├─ [0,0,1,0] = dance
                                    └─ [0,0,0,1] = stand
```

| 组件 | 维度 | 部署可用? | 说明 |
|------|------|-----------|------|
| 原始 proprio | 16D | ✅ | 关节角度/角速度 + 足底力 + 速度指令 |
| motion one-hot | 4D | ✅ | 动作类型指令（用户/上层控制器提供） |
| **Total obs** | **20D** | ✅ | 假肢传感器 + 动作指令 |
| priv_info | 113D | ❌ | 与单动作完全相同 |
| proprio_hist | 30×20D | ✅ | Student 输入（历史 30 帧） |

### Multi-Motion Stage 0: Per-Motion Body Policy

每种动作独立训练一个 28-DOF AMP 策略，方法与单动作 Stage 0 完全相同。

```
walk:  amp_humanoid_walk.npy  → stage0_amp_walk_5050.pth
run:   amp_humanoid_run.npy   → stage0_amp_run_1700.pth
dance: amp_humanoid_dance.npy → stage0_amp_dance_8000.pth
stand: amp_humanoid_stand.npy → stage0_amp_stand_3900.pth  (自动生成的静止参考动作)
```

**站立参考动作生成**：从行走参考动作中提取一帧站立姿态（frame 142，root_z=0.859），复制 30 帧生成静止 clip。使用 SkeletonMotion OrderedDict 格式，必须用 numpy 1.x 保存。

### Multi-Motion Stage 1: Teacher DAgger

与单动作 DAgger 方法相同，但有以下扩展：

1. **obs 扩展为 20D**：原 16D + 4D motion one-hot
2. **Per-motion body policy**：环境根据 `env_motion_types` 为每个 env 选择对应的 body policy
3. **Per-motion RSI**：每种动作使用自己的参考动作库做 Reference State Initialization
4. **Random motion assignment**：每次 episode reset 时随机分配动作类型

```python
# DAgger 训练循环（多动作版）
for epoch in range(8000):
    obs = env.get_obs()  # 20D: proprio(16D) + motion_onehot(4D)
    teacher_action = policy(obs, priv_info)  # 4D prosthesis action

    # 每个 env 根据自己的 motion_id 获取对应 body policy 的 ground truth
    body_action = body_policies[motion_id](full_body_obs)
    target = body_action[ACTIVE_JOINTS]  # 4D

    loss = MSE(teacher_action, target)
    loss.backward()
```

### Multi-Motion Stage 2: Hora 蒸馏

与单动作 Stage 2 完全同构，只是维度从 16D 变为 20D：

```
proprio_hist (30×20D) → sa_mean_std → adapt_tconv → z_student (32D)
priv_info (113D) → priv_mlp (frozen) → z_teacher (32D)

Loss = MSE(z_student, z_teacher.detach())
```

- **仅训练** adapt_tconv（~33K 参数），其余全部冻结
- adapt_tconv 从本体感觉历史中同时学习：(a) 当前动作类型 (b) 环境适应信息
- 因为 obs 中含 motion one-hot，proprio_hist 自然包含了动作类型的时序信息

### Per-Motion 配置

| 动作 | motion_id | 速度目标 (m/s) | 参考动作文件 |
|------|-----------|---------------|-------------|
| Walk | 0 | 1.0 | amp_humanoid_walk.npy |
| Run | 1 | 2.5 | amp_humanoid_run.npy |
| Dance | 2 | 0.0 | amp_humanoid_dance.npy |
| Stand | 3 | 0.0 | amp_humanoid_stand.npy |

Per-motion 奖励权重（例）：

| 动作 | vel_reward | up_reward | height_reward | lateral_penalty |
|------|-----------|-----------|---------------|-----------------|
| Walk | 3.0 | 1.0 | 0.5 | 0.3 |
| Run | 3.0 | 1.5 | 0.5 | 0.2 |
| Dance | 0.5 | 2.0 | 1.0 | 0.1 |
| Stand | 0.5 | 2.0 | 1.5 | 0.5 |

### 动作切换与过渡策略

**问题**：mid-episode 切换动作时，body policy 突然改变，导致全身动力学不连续，humanoid 可能摔倒。
这是因为每种 body policy 独立训练在自己动作的状态分布上，当接收到其他动作的状态观测（Out-of-Distribution）时，
输出不可预测。例如：站立策略接收行走状态的观测（速度 1.5 m/s），输出的全身动作会导致失衡。

**方案一：Transition Blending（已探索，效果有限）**

对 body policy 的输出进行线性插值，在 N 步内从旧动作平滑过渡到新动作：

```
body_action(t) = α × old_body_action + (1 - α) × new_body_action
α = remaining / total    (从 1.0 线性衰减到 0.0)
```

还探索了 deceleration-first blending（先减速旧策略再加速新策略）和 velocity-aware blending，
但核心问题不变：新策略接收 OOD 观测时无法产生有效输出。

Blending 方案的序列评估结果较差：walk→stand→walk 仅 ~23% survival。

**方案二：Soft-Reset 过渡（最终方案 ✅）**

核心思路：切换动作时，对环境进行"软重置"——保留 agent 的 XY 位置，但将关节状态、
根节点高度和速度重置为新动作的参考状态初始化（RSI）。

```python
def soft_reset_to_motion(motion_id):
    saved_xy = root_states[:, 0:2]          # 保留 XY 位置
    root_states, dof_pos, dof_vel = RSI(motion_id)   # 新动作的参考状态
    root_states[:, 0:2] = saved_xy          # 恢复 XY 位置
    apply_to_simulation()
    reset_observation_history()
```

**为什么有效**：每个 body policy 在切换后立即获得自己训练过的状态分布内的观测，
完全避免了 OOD 问题。视觉上的过渡表现为姿态的瞬间调整。

**Soft-Reset 序列评估结果**（64 envs, 10 trials, switch model）：

| 序列 | Survival | 备注 |
|------|----------|------|
| walk→stand→walk | **85.6%** | ↑ 从 23% |
| walk→run→walk→stand→run | **84.4%** | ↑ 从 0% |
| dance→stand→dance | **96.2%** | ↑ 从 13% |
| stand→walk→run→dance→stand→walk | **94.8%** | ↑ 从 0% |

Per-Motion 独立表现（soft-reset, switch model）：

| 动作 | Ep Len | Survival |
|------|--------|----------|
| Walk | 815.9 | 79.4% |
| Run | 893.2 | 87.7% |
| Dance | 783.8 | 80.0% |
| Stand | 1000.0 | 100.0% |

**实现文件**：
- `proknee_multi_motion.py` → `soft_reset_to_motion()` 方法
- `evaluate_motion_sequences.py` → `--soft-reset` 参数（默认启用）

**后续改进方向**（不再阻塞当前功能）：
1. Multi-motion conditioned body policy — 单个 Stage 0 网络支持所有动作
2. Transition controller — 专门训练动作间过渡的策略
3. Curriculum learning — 先训短序列，逐步增加段数

### 验收标准 (多动作)

| Stage | 指标 | 标准 | Legacy 实际 | Realistic 实际 |
|-------|------|------|------------|---------------|
| Stage 0 per-motion | ep_len (GPU) | ≥ 250步 | ✅ 287-299 | (共享) |
| Stage 1 Multi | ep_len (overall) | ≥ 220步 | ✅ 242.6 | ✅ 243.9 |
| Stage 2 Multi | ep_len (per-motion) | ≥ 240步 | ✅ 262-271 | ✅ 262-263 |
| Stage 2 Sequence | survival (soft-reset) | ≥ 80% | ✅ 85-96% | ✅ 84-95% |

---

## Realistic 假肢观测架构

> 代码变更: `proknee_multi_motion.py`, `constants_multi.py`, 训练脚本，评估脚本

### 问题：假肢不应知道动作类型

在 Legacy 多动作模式中，motion one-hot 直接放在 obs (20D) 中：

```
Legacy: obs = proprio(16D) + motion_onehot(4D)  ← 假肢知道 "当前是行走/跑步/..."
```

**这不符合真实假肢的部署场景**：真实假肢没有上位机告知当前执行什么动作，
只能通过传感器（关节角度、角速度、足底力）感知人体运动，被动适应。

### 解决方案：两种模式

#### Realistic 模式（推荐，`motion_in_obs=False`）

```
obs (16D) = proprio(16D)          ← 纯传感器，与单动作相同
priv_info (117D) = base(113D) + motion_onehot(4D)  ← Teacher 知道动作类型
proprio_hist (30×16D)             ← Student 从运动模式推断
```

- Teacher: 通过 priv_info → PrivilegedMLP → latent，隐式编码动作类型
- Student: 通过 proprio_hist(30×16D) → adapt_tconv → latent，从运动模式推断动作类型
- 关键: `command(1D)` 保留在 obs 中。walk=1.0, run=2.5, stand/dance=0.0
  - 真实假肢可以接收用户的步速偏好设定
  - dance 和 stand 的 command 相同(0.0)，Student 必须从运动模式区分二者

#### Legacy 模式（向后兼容，`motion_in_obs=True`）

```
obs (20D) = proprio(16D) + motion_onehot(4D)
priv_info (113D) = base(113D)
proprio_hist (30×20D)
```

保持旧模型可用。训练/评估脚本不加 `--no-motion-in-obs` 时使用此模式。

### 实现

```python
class ProKneeMultiMotionEnv:
    def __init__(self, ..., motion_in_obs=False):  # 默认 realistic
        if motion_in_obs:
            self.obs_dim = 20   # legacy
            self.priv_dim = 113
        else:
            self.obs_dim = 16   # realistic
            self.priv_dim = 117

    def _compute_base_obs(self):
        obs = super()._compute_base_obs()  # 16D
        if self.motion_in_obs:
            obs = torch.cat([obs, self.motion_onehot], dim=-1)  # → 20D
        return obs

    def _compute_priv_info(self):
        priv = super()._compute_priv_info()  # 113D
        if not self.motion_in_obs:
            priv = torch.cat([priv, self.motion_onehot], dim=-1)  # → 117D
        return priv
```

### 训练命令

```bash
# Realistic (推荐)
$PYTHON scripts/train_stage1_multi.py --no-motion-in-obs --switch-motion ...
$PYTHON scripts/train_stage2_multi.py --no-motion-in-obs --switch-motion \
    --teacher-ckpt outputs/checkpoints/stage1_multi_realistic/best.pth ...

# Legacy (旧, 不加 --no-motion-in-obs)
$PYTHON scripts/train_stage1_multi.py --switch-motion ...
```

### 模型存储

```
outputs/checkpoints/
├── stage1_multi_realistic/best.pth   # Realistic Teacher (obs=16D, priv=117D)
├── stage2_multi_realistic/best.pth   # Realistic Student (obs=16D)
├── stage1_multi_switch/best.pth      # Legacy Teacher (obs=20D, priv=113D)
└── stage2_multi_switch/best.pth      # Legacy Student (obs=20D)
```

---

## 交互式可视化

> `scripts/interactive_visualize.py`

通过 Isaac Gym 键盘事件实现实时动作切换，用于验证假肢适应性。

### 键盘控制

| 按键 | 动作 | 描述 |
|------|------|------|
| W | Walk | 1.0 m/s 行走 |
| R | Run | 2.5 m/s 奔跑 |
| D | Dance | 跳舞 |
| S | Stand | 站立 |
| Q | Quit | 退出 |

### 实现原理

1. 使用 `gym.subscribe_viewer_keyboard_event()` 注册键盘事件
2. 每帧通过 `gym.query_viewer_action_events()` 轮询
3. 按键触发 `smooth_transition_to_motion()` 平滑切换动作 (默认)
4. 可选 `--hard-reset` 使用旧版 `soft_reset_to_motion()` (瞬间切换)
5. Episode 长度设为 100000（不自动 reset）
6. 倒地后自动恢复到当前动作 (hard-reset)

### 特点

- **单 env**: 1 个 agent，清晰观察动作切换
- **假肢适应**: Realistic 模式下，假肢不知道按键输入了什么动作，只从传感器信息推断
- **无限时长**: 不会自动 reset，可以持续观察

---

## 平滑动作过渡 (Smooth Transition)

> `ProKneeMultiMotionEnv.smooth_transition_to_motion()`

### 问题背景

原始 `soft_reset_to_motion()` 在一帧内将所有 28 个 DOF 瞬间切换到新动作的参考姿态 (RSI)，导致视觉上的"闪烁"跳变。这是因为 Body Policy OOD 问题导致所有 blending 方案都失败 (survival 0-23%)，只能用硬切换。

### 解决方案: 内联状态插值

`smooth_transition_to_motion()` 在函数内部执行 N 帧渐进式物理状态插值：

```
当前状态 ──interpolate──> ... ──interpolate──> 目标 RSI
  frame 0     frame 1          frame N-1       frame N
```

每一帧的操作:
1. 计算 Hermite smooth-step alpha: `α = 3t² - 2t³` (缓进缓出)
2. 插值 root state: 高度、旋转 (lerp+normalize)、速度
3. 插值 DOF state: 关节角度和角速度
4. **强制设置物理状态** (`set_actor_root_state_tensor_indexed` + `set_dof_state_tensor_indexed`)
5. 设置 PD targets 匹配插值位置
6. 运行一步物理模拟 (更新渲染和传感器)
7. 更新 proprio_hist (Student 观察到平滑过渡)

### 关键设计

| 特性 | 说明 |
|------|------|
| 不使用 body policy | 过渡期间直接设置物理状态，避免 OOD |
| XY 位置保持不变 | 人物不会在地图上跳动 |
| proprio_hist 不清零 | Student 看到平滑的 30帧历史 |
| progress_buf 不重置 | Episode 连续不中断 |
| 支持中途切换 | 过渡中按新键 → 从当前物理状态开始新过渡 |

### 与 soft_reset_to_motion 对比

| | soft_reset (hard) | smooth_transition |
|--|-------------------|-------------------|
| 过渡帧数 | 1 帧 (瞬移) | N 帧 (默认 10 ≈ 0.33s) |
| 视觉效果 | 闪烁跳变 | 平滑过渡 |
| proprio_hist | 清零 | 自然过渡 |
| progress_buf | 重置 | 不重置 |
| 存活率 | 85-95% | 80-92% |
| 用途 | 摔倒恢复, 序列评估基准 | 交互式可视化, 序列评估默认 |

### 验收结果 (Realistic 模型, 64 envs, 10 trials)

| 序列 | Soft-Reset | Smooth (10f) | 差距 |
|------|-----------|-------------|------|
| walk→stand→walk | 86.7% | 81.2% | -5.5pp |
| walk→run→walk→stand→run | 84.1% | 80.0% | -4.1pp |
| dance→stand→dance | 94.7% | 92.2% | -2.5pp |
| full-cycle | 94.8% | 90.2% | -4.6pp |
