# ProKnee-HoraStyle 训练理论与公式

## 概述

ProKnee-HoraStyle 采用三阶段训练流程，基于 **Hora** (History-Oriented Robot Adaptation)
架构，将假肢膝关节控制从仿真迁移到受限传感输入的学生策略。

---

## 1. PPO (Proximal Policy Optimization)

### 1.1 策略梯度目标

PPO 通过 clipped surrogate objective 限制策略更新幅度：

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ 为重要性采样比率
- $\hat{A}_t$ 为广义优势估计 (GAE)
- $\epsilon = 0.2$ 为裁剪参数 (`e_clip`)

### 1.2 广义优势估计 (GAE)

$$
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}
$$

$$
\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)
$$

其中：
- $\gamma = 0.99$ 为折扣因子
- $\lambda = 0.95$ 为 GAE 参数 (`tau`)
- $d_t$ 为终止标志
- $V(s)$ 为价值函数

**代码实现** (`ppo.py:104-124`)：
```python
last_gae = 0
for t in reversed(range(horizon_length)):
    next_value = last_value if t == T-1 else values[t+1]
    delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
    last_gae = delta + gamma * tau * (1 - dones[t]) * last_gae
    advantages[t] = last_gae
returns = advantages + values
```

### 1.3 价值函数损失

$$
L^{VF}(\theta) = \frac{1}{2} \mathbb{E}_t \left[ (V_\theta(s_t) - R_t)^2 \right]
$$

Stage 2 使用 clipped value loss：

$$
L^{VF}_{clip} = \frac{1}{2} \max\left[ (V - R)^2,\; (V_{clip} - R)^2 \right]
$$

$$
V_{clip} = V_{old} + \text{clip}(V - V_{old}, -\epsilon, \epsilon)
$$

### 1.4 熵正则化

$$
L^{ENT}(\theta) = \mathbb{E}_t \left[ H[\pi_\theta(\cdot|s_t)] \right]
$$

对于高斯策略：

$$
H = \frac{1}{2} \sum_{i=1}^{d} \left( \log(2\pi e \sigma_i^2) \right)
$$

### 1.5 总损失

$$
L(\theta) = L^{CLIP}(\theta) + c_v \cdot L^{VF}(\theta) - c_e \cdot L^{ENT}(\theta)
$$

**超参数** (Stage 1 / Stage 2)：

| 参数 | Stage 1 | Stage 2 |
|------|---------|---------|
| learning_rate | 5e-4 | 3e-4 |
| e_clip ($\epsilon$) | 0.2 | 0.2 |
| gamma ($\gamma$) | 0.99 | 0.99 |
| tau ($\lambda$) | 0.95 | 0.95 |
| entropy_coef ($c_e$) | 0.01 | 0.005 |
| value_coef ($c_v$) | 1.0 | 1.0 |
| max_grad_norm | 1.0 | 1.0 |
| horizon_length | 16 | 16 |
| mini_epochs | 5 | 5 |
| minibatch_size | 16384 | 16384 |

---

## 2. Reward 设计

### 2.1 Stage 0: Full Body AMP Walking

Stage 0 使用 **AMP (Adversarial Motion Prior)** 框架训练全身 28-DOF 步行策略：

$$
r_t = w_{task} \cdot r_{task} + w_{style} \cdot r_{style}
$$

- $r_{task}$: 前进速度追踪、存活等任务奖励
- $r_{style}$: 判别器给出的运动风格奖励 (对比参考动作 `amp_humanoid_walk.npy`)

**网络**: Actor MLP [1024, 512] + μ head → 28-D 动作

### 2.2 Stage 1 & 2: Prosthetic Knee Control

$$
r_t = w_1 \cdot r_{alive} + w_2 \cdot r_{upright} + w_3 \cdot r_{height} + w_4 \cdot r_{tracking} - w_5 \cdot r_{action}
$$

各分项定义：

**存活奖励** (Alive Bonus):
$$
r_{alive} = 1.0
$$

**直立奖励** (Upright):
$$
\text{up\_proj} = 1 - 2(q_x^2 + q_y^2) \quad \text{(from quaternion } q = [q_x, q_y, q_z, q_w]\text{)}
$$
$$
r_{upright} = \text{clamp}(\text{up\_proj}, 0, 1)
$$

**高度奖励** (Height):
$$
r_{height} = \exp\left( -10 \cdot (z - 0.9)^2 \right)
$$

**假肢追踪** (Prosthesis Tracking):
$$
e_{pros} = \frac{1}{N_{dof}} \sum_{i \in \text{active}} (a^{teacher}_i - a^{body}_i)^2
$$
$$
r_{tracking} = \exp(-2 \cdot e_{pros})
$$
其中 $a^{teacher}$ 为当前策略动作, $a^{body}$ 为 frozen body policy 动作, 
active DOFs = [LEFT_KNEE, LEFT_ANKLE_X, LEFT_ANKLE_Y, LEFT_ANKLE_Z] (4-DOF)

**动作惩罚** (Action Cost):
$$
r_{action} = 0.01 \cdot \sum_{i} a_i^2
$$

**死亡惩罚**:
$$
r_t = -2.0 \quad \text{if } z < 0.3
$$

**权重**:

| 分项 | 权重 |
|------|------|
| $r_{alive}$ | 1.0 |
| $r_{upright}$ | 1.0 |
| $r_{height}$ | 0.5 |
| $r_{tracking}$ | 1.5 |
| $r_{action}$ | 0.01 |
| death penalty | -2.0 |

**理论最大每步奖励**: $\approx 1.0 + 1.0 + 0.5 + 1.5 = 4.0$

### 2.3 终止条件

$$
\text{done} = (z < 0.3) \lor (t \geq T_{max})
$$

其中 $T_{max} = 300$ (Stage 1/2) 或 500 (原始配置)。

---

## 3. Teacher-Student 蒸馏框架

### 3.1 架构概览

```
                ┌─────────────────────────────────────────────┐
                │           AdaptationModule                   │
                │                                             │
  Stage 1       │  priv_info ──→ [PrivilegedMLP] ──→ z_t     │
  (Teacher)     │  (GRF, contact,    [128,64]→8      (latent)│
                │   terrain, err)                              │
                ├─────────────────────────────────────────────┤
  Stage 2       │  proprio_hist ──→ [ProprioAdaptTConv] → z_s│
  (Student)     │  (30×16-D)        Conv1d stack → 8   (latent)│
                └──────────────────────┬──────────────────────┘
                                       │
                          z (= z_t or z_s)
                                       │
                          ┌────────────┴────────────┐
                          │     ActorCritic          │
                          │ obs(105) + z(8) → [256,  │
                          │  128,64] → action(4-D)   │
                          │              → value(1-D) │
                          └─────────────────────────┘
```

### 3.2 Teacher: PrivilegedMLP

输入 18-D 特权信息：

| 分量 | 维度 | 来源 |
|------|------|------|
| 地反力 (GRF) | 6D | 左右脚力传感器 |
| 足底接触 | 2D | GRF 阈值判断 |
| 地形高度 | 4D | 占位 (平地=0) |
| 姿态误差 | 3D | 四元数虚部 |
| 速度误差 | 3D | 当前速度 - 目标速度 |

$$
z_t = \tanh\left( \text{MLP}_{[18 \to 128 \to 64 \to 8]}(x_{priv}) \right)
$$

### 3.3 Student: ProprioAdaptTConv

输入 30×16-D 本体感觉历史：

| 分量 | 维度/步 |
|------|---------|
| 膝关节位置/速度 | 2D |
| 踝关节位置/速度 | 6D |
| 髋关节位置/速度 | 6D |
| 左足 Fz | 1D |
| 目标速度指令 | 1D |

时间卷积路径：
$$
(B, 30, 16) \xrightarrow{\text{Linear}} (B, 30, 32) \xrightarrow{\text{permute}} (B, 32, 30)
$$
$$
\xrightarrow{\text{Conv1d}(k=9,s=2)} (B, 32, 15) \xrightarrow{\text{Conv1d}(k=5,s=1)} (B, 32, 15)
$$
$$
\xrightarrow{\text{Conv1d}(k=3,s=1)} (B, 32, 15) \xrightarrow{\text{flatten}} (B, 480)
$$
$$
\xrightarrow{\text{Linear}} (B, 8) \xrightarrow{\tanh} z_s
$$

### 3.4 蒸馏损失

$$
L_{distill} = \frac{1}{d_z} \sum_{i=1}^{d_z} (z_s^{(i)} - \text{sg}(z_t^{(i)}))^2
$$

其中 $\text{sg}(\cdot)$ 为 stop-gradient (teacher 参数冻结)。

### 3.5 Stage 2 总损失

$$
L_{total} = L^{CLIP} + c_v \cdot L^{VF}_{clip} - c_e \cdot L^{ENT} + \lambda_{distill} \cdot L_{distill}
$$

**Stage 2 超参数**: $\lambda_{distill} = 1.0$, $c_e = 0.005$

### 3.6 Reward 归一化 (Stage 2)

Student 训练使用 running reward normalization：

$$
\bar{r}_t = \frac{r_t}{\max(\hat{\sigma}_r, 10^{-4})}
$$

$$
\hat{\sigma}_r = \sqrt{\frac{\hat{V}_r}{\hat{n}}}
$$

其中 $\hat{V}_r$ 和 $\hat{n}$ 通过 Welford 在线算法递增更新。

---

## 4. 三阶段训练流程

### Stage 0: Full Body AMP Walking

```
目标: 训练全身 28-DOF 自然步行策略
方法: AMP (Adversarial Motion Prior) + PPO
输入: 105-D 观测 (root state + DOF + key body positions)
输出: 28-D 动作 (所有关节)
参考动作: amp_humanoid_walk.npy
训练规模: 4096 envs × 10000 epochs
```

### Stage 1: Teacher with Privileged Info

```
目标: 训练假肢控制器 (使用特权信息)
方法: PPO + PrivilegedMLP
输入: 105-D 观测 + 18-D 特权信息
输出: 4-D 动作 (knee + 3 ankle)
冻结: Stage 0 body policy (24 DOFs 由 frozen body policy 控制)
训练规模: 4096 envs × 3000 epochs
```

### Stage 2: Student Distillation

```
目标: 将 teacher 知识蒸馏到仅使用本体感觉的 student
方法: PPO + 蒸馏损失
输入: 105-D 观测 + 30×16-D 本体感觉历史
输出: 4-D 动作
蒸馏: z_s → z_t (MSE loss, λ=1.0)
冻结: Teacher 参数 + Stage 0 body policy
训练规模: 4096 envs × 3000 epochs
```

### 流程图

```
Stage 0                   Stage 1                    Stage 2
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ AMP Walking  │    │ Teacher + Priv   │    │ Student + Distill│
│ 28-DOF       │───→│ 4-DOF prosthesis │───→│ 4-DOF prosthesis │
│ Full body    │    │ + frozen body    │    │ + proprio only   │
└──────────────┘    └──────────────────┘    └──────────────────┘
  body policy ─────────→ frozen ──────────────→ frozen
                    teacher policy ─────────→ frozen (supervision)
```

---

## 5. Stage 0: AMP (Adversarial Motion Prior) 详解

### 5.1 AMP 判别器

AMP 在标准 RL 之上引入一个 **判别器** $D_\phi(s_t, s_{t+1})$，区分
"来自策略的状态转换" 与 "来自参考动作库 (`amp_humanoid_walk.npy`) 的状态转换"。

$$
L_D(\phi) = -\mathbb{E}_{(s,s') \sim \mathcal{M}} [\log D_\phi(s, s')]
           - \mathbb{E}_{(s,s') \sim \pi_\theta} [\log(1 - D_\phi(s, s'))]
           + w_{gp} \cdot L_{GP}
$$

其中：
- $\mathcal{M}$: 参考运动数据库（MoCap 步行数据）
- $\pi_\theta$: 当前策略采集的轨迹
- $w_{gp} = 5.0$: gradient penalty 权重
- $L_{GP} = \mathbb{E}\left[\|\nabla_s D_\phi(s, s')\|^2\right]$: gradient penalty

### 5.2 AMP 风格奖励

将判别器输出转化为奖励信号：

$$
r_{style}(s_t, s_{t+1}) = \max(0,\; 1 - 0.25 \cdot (D_\phi(s_t, s_{t+1}) - 1)^2)
$$

Stage 0 的总奖励：

$$
r_t = 0.5 \cdot r_{task} + 0.5 \cdot r_{style}
$$

其中 $r_{task}$ 包含存活、高度、前进速度追踪等。

### 5.3 观测构造 (105-D)

`compute_humanoid_observations()` JIT 函数生成的 105-D 观测：

| 分量 | 维度 | 计算方式 |
|------|------|----------|
| Root height ($z$) | 1D | `root_pos[:, 2:3]` |
| Root rotation | 6D | `quat_to_tan_norm(root_rot)` — 正切-法线表示 |
| Root lin vel (local) | 3D | 旋转到局部坐标系 |
| Root ang vel (local) | 3D | 旋转到局部坐标系 |
| DOF positions | 28D | 直接拼接 `dof_pos` |
| DOF velocities | 28D | 直接拼接 `dof_vel` × 0.1 缩放 |
| Key body positions (local) | 12D | 4 个关键体点 × 3D 相对于 root |
| **Observation normalization** | — | 使用 `running_mean_std` 在线归一化 |

关键体：`right_hand`, `left_hand`, `right_foot`, `left_foot`

### 5.4 PD 位置控制

动作空间为 $[-1, 1]^{28}$ 归一化值，通过以下映射转换为关节位置目标：

$$
\theta^{target}_i = \text{offset}_i + \text{scale}_i \cdot a_i
$$

其中 offset 和 scale 按关节类型计算：

| 关节类型 | 自由度 | offset | scale |
|---------|-------|--------|-------|
| 球关节 (3-DOF) | 腹部、颈、肩、髋 | $0$ | $\pi$ |
| 铰链关节 (1-DOF) | 肘、膝、踝 | $\frac{lo+hi}{2}$ | $0.7 \cdot \frac{hi-lo}{2}$ |

实现 (`proknee_base.py:574-581`):
```python
pd_targets = self._pd_action_offset + self._pd_action_scale * action
```

### 5.5 Reference State Initialization (RSI)

每个 episode 开始时，从参考动作库 $\mathcal{M}$ 中随机采样一帧作为初始状态：

$$
s_0 \sim \text{Uniform}(\mathcal{M})
$$

这避免了从 T-pose 静止状态开始训练，大幅提升探索效率。

实现 (`proknee_base.py:_reset_envs`): 从 `MotionLib` 采样 `root_pos, root_rot, dof_pos, dof_vel, root_vel, root_ang_vel, key_body_pos`。

---

## 6. 关键设计决策

1. **Observation 一致性**: Stage 1/2 的 105-D 观测与 Stage 0 完全相同
   (使用相同的 `compute_humanoid_observations` JIT 函数)

2. **PD 控制**: 动作空间为 [-1,1] 归一化，通过 `_pd_action_offset + _pd_action_scale * action`
   转换为关节位置目标（详见 §5.4）

3. **Reference State Init**: 使用 AMP 参考动作库初始化 episode 的初始姿态，
   避免从静止 T-pose 开始（详见 §5.5）

4. **Latent 空间**: 使用 tanh 激活将 latent 约束在 [-1,1]，
   8 维足够编码步态相位和地面接触信息

5. **Frozen Body Policy**: Stage 0 训练得到的 28-DOF 策略在 Stage 1/2 中被冻结，
   其网络结构为 `[105→1024→512] + μ(28)`，使用 running mean/std 归一化观测

6. **Prosthesis DOF 分配**: 主动控制 4-DOF (knee + 3 ankle)，
   冻结 24-DOF body joints，无被动锁定关节

---

## 7. 故障排除

### 7.1 `Fatal JavaScript invalid array length` 崩溃

**错误信息**:
```
Fatal error in , line 0
Fatal JavaScript invalid array length
FailureMessage Object: 0x7ffece81f3b0
Illegal instruction (core dumped)
```

**原因**: 此错误来自 **GitHub Copilot CLI 二进制文件**（`/home/user/.local/bin/copilot`），
而非 Python 训练代码。Copilot CLI 内嵌 V8 JavaScript 引擎，当通过 Copilot
会话执行训练命令时，训练产生的大量 stdout 输出（如 Stage 0 的 16,000+ 行日志）
会被 V8 引擎尝试缓冲到一个内部字符串/数组中，超过 V8 的最大数组长度限制
（约 2^32 - 1 字节或元素），导致 `invalid array length` 致命错误。

**解决方案**:

1. **重定向输出到文件**（推荐）:
   ```bash
   # 在 Copilot 会话中运行训练时，将输出重定向
   nohup bash scripts/train_stage0.sh > outputs/stage0.log 2>&1 &
   
   # 然后用 tail 查看进度
   tail -f outputs/stage0.log
   ```

2. **限制输出频率**: 修改训练脚本中的日志间隔
   ```bash
   python train.py stage=0 train.print_interval=100  # 每100个epoch才打印
   ```

3. **在独立终端运行**: 不要在 Copilot CLI 会话中直接运行长时间训练，
   使用独立的 tmux/screen 会话：
   ```bash
   tmux new -s training
   bash scripts/train_stage0.sh
   # Ctrl-B D 分离
   ```

4. **使用 `--quiet` 模式**: 如果训练脚本支持静默模式

**根本原因**: V8 引擎对单个字符串/数组有硬限制，当 Copilot CLI
累积了训练过程中产生的数十 MB 输出文本后触发此限制。
这是 Copilot CLI 的已知限制，不影响训练代码本身。
