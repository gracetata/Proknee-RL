# ProKnee-HoraStyle 使用手册

> 技术原理详见 `METHOD.md`，多动作进度详见 `docs/PROGRESS.md`

---

## 1. 环境配置

```bash
# 每次开始工作前执行
export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python
cd /home/user/Workspace/ProKnee/ProKnee-HoraStyle
```

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX 4090 (24GB) |
| Python | 3.8 (conda: `proknee_tc`) |
| Isaac Gym | Preview 4 (gym_38.so) |
| PyTorch | 2.4.1+cu124 |
| RL框架 | rl_games (Stage 0), DAgger (Stage 1), ProprioAdapt (Stage 2) |

---

## 2. 项目结构

```
ProKnee-HoraStyle/
├── IsaacGymEnvs/isaacgymenvs/         # Stage 0 训练 (官方 AMP)
│   └── assets/amp/motions/            # 参考动作 (.npy)
├── hora/                              # 原始 Hora 参考代码
├── proknee_hora/                      # Stage 1/2 核心代码
│   ├── envs/
│   │   ├── constants.py               # 单动作常量 (OBS_DIM=16)
│   │   ├── constants_multi.py         # 多动作常量 (legacy + realistic)
│   │   ├── proknee_base.py            # 单动作环境基类
│   │   └── proknee_multi_motion.py    # 多动作环境 (继承 base)
│   └── algo/
│       ├── proprio_adapt.py           # Stage 2 ProprioAdapt (Hora风格)
│       └── models/
│           ├── actor_critic.py        # ProKneePolicy (支持 16D/20D)
│           ├── adaptation.py          # PrivilegedMLP + ProprioAdaptTConv
│           └── running_mean_std.py    # 在线归一化
├── scripts/
│   ├── train_stage0_official.sh       # Stage 0 训练 (单动作)
│   ├── train_stage0_stand.sh          # Stage 0 训练 (站立)
│   ├── train_stage1_dagger.py         # Stage 1 单动作 DAgger
│   ├── train_stage1_multi.py          # Stage 1 多动作 DAgger (支持 --no-motion-in-obs)
│   ├── train_stage2.py                # Stage 2 单动作蒸馏
│   ├── train_stage2_multi.py          # Stage 2 多动作蒸馏 (支持 --no-motion-in-obs)
│   ├── evaluate_stage2.py             # 单动作评估 + 步态分析
│   ├── evaluate_stage2_multi.py       # 多动作 per-motion 评估
│   ├── evaluate_motion_sequences.py   # 动作序列评估 (支持 --visualize --sequence)
│   ├── interactive_visualize.py       # ★ 交互式可视化 (键盘切换动作)
│   ├── analyze_gait.py                # 步态分析 (膝关节/踝关节对比)
│   └── create_stand_motion.py         # 站立参考动作生成器
├── outputs/
│   ├── checkpoints/                   # 所有最终模型 ★
│   └── gait_analysis/                 # 步态分析结果 (图表+数据)
├── docs/
│   └── PROGRESS.md                    # 多动作进度跟踪
├── METHOD.md                          # 技术方法说明 ★
└── MEMORY.md                          # 本文档 ★
```

---

## 3. 模型存储

```
outputs/checkpoints/
├── stage0/                            # 28-DOF 全身策略
│   ├── stage0_amp_walk_5050.pth       # Walk   ✅ ep_len=287.2
│   ├── stage0_amp_run_1700.pth        # Run    ✅ ep_len=287.5
│   ├── stage0_amp_dance_8000.pth      # Dance  ✅ ep_len=295.3
│   └── stage0_amp_stand_900.pth       # Stand  ✅ ep_len=299.0
├── stage1/
│   └── best.pth                       # 单动作 Teacher (walk)  ✅ ep_len=284.7
├── stage1_multi/
│   └── best.pth                       # 多动作 Teacher (legacy obs=20D)  ✅
├── stage1_multi_switch/
│   └── best.pth                       # 多动作 Teacher + switching (legacy)  ✅
├── stage1_multi_realistic/
│   └── best.pth                       # ★ 多动作 Teacher (realistic obs=16D) ✅ ep_len=243.9
├── stage2/
│   └── best.pth                       # 单动作 Student (walk)  ✅ ep_len=279.3
├── stage2_multi/
│   └── best.pth                       # 多动作 Student (legacy obs=20D)  ✅
├── stage2_multi_switch/
│   └── best.pth                       # 多动作 Student + switching (legacy)  ✅
└── stage2_multi_realistic/
    └── best.pth                       # ★ 多动作 Student (realistic obs=16D) ✅ reward=466.53
```

**依赖关系**：
```
Stage 0 → Stage 1 (frozen body policy, 24 DOF ground truth)
Stage 0 + Stage 1 → Stage 2 (frozen body + frozen teacher → 仅训练 adapt_tconv)
```

---

## 4. 核心维度常量

### 单动作 (`constants.py`)

| 常量 | 值 | 说明 |
|------|-----|------|
| OBS_DIM | 16 | 本体感觉 (部署可用) |
| TEACHER_PRIV_INFO_DIM | 113 | 特权信息 (仅仿真) |
| LATENT_DIM | 32 | latent 表示 |
| PROPRIO_HISTORY_LEN | 30 | Student 历史帧数 |
| PROSTHESIS_ACTION_DIM | 4 | 膝 + 踝×3 |

### 多动作 (`constants_multi.py`) — 两种模式

| 常量 | Legacy (旧) | Realistic (新) | 说明 |
|------|------------|----------------|------|
| obs_dim | 20 (16+4D onehot) | **16** (纯传感器) | 假肢观测维度 |
| priv_info_dim | 113 | **117** (113+4D onehot) | 特权信息维度 |
| proprio_dim | 20 | **16** | Student 本体感觉维度 |
| proprio_hist | 30×20 | **30×16** | 历史帧维度 |

**Realistic 模式 (推荐)**：motion one-hot 仅在 priv_info 中，假肢不知道当前动作类型，只能从运动模式推断。

### OBS 组成

```
单动作 (16D): knee_pos(1) + knee_vel(1) + ankle_pos(3) + ankle_vel(3) 
              + hip_pos(3) + hip_vel(3) + foot_fz(1) + command(1)

Legacy 多动作 (20D): 上述 16D + motion_onehot(4D)  ← 假肢知道动作类型
Realistic 多动作 (16D): 上述 16D                    ← 假肢只有传感器信息
```

### 假肢关节

```
ACTIVE (训练目标): LEFT_KNEE(24) + LEFT_ANKLE_X/Y/Z(25,26,27) = 4 DOF
FROZEN (体策略):   其余 24 DOF (Stage 0 body policy 控制)
```

---

## 5. 训练命令 — 单动作 (平地行走)

### Stage 0: 全身 AMP (28-DOF)

```bash
# Walk
PYTHONUNBUFFERED=1 nohup bash scripts/train_stage0_official.sh 10000 \
    > outputs/stage0_walk.log 2>&1 &

# 恢复训练 (从 checkpoint)
nohup bash scripts/train_stage0_official.sh 10000 \
    runs/HumanoidAMP_XX-XX-XX/nn/HumanoidAMP_XX_750.pth \
    > outputs/stage0_resume.log 2>&1 &
```

验收标准: GPU ep_len ≥ 280

### Stage 1: Teacher DAgger (4-DOF)

```bash
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage1_dagger.py \
    --device cuda:0 --num-envs 4096 --max-epochs 5000 \
    --steps-per-epoch 32 --lr 1e-3 --noise-std 0.3 \
    > outputs/stage1_dagger_train.log 2>&1 &

# CPU 快速测试
$PYTHON scripts/train_stage1_dagger.py --device cpu --num-envs 16 --max-epochs 10
```

验收标准: GPU ep_len ≥ 280

### Stage 2: Student 蒸馏 (Hora)

```bash
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage2.py \
    --device cuda:0 --num-envs 4096 --max-steps 500000000 \
    > outputs/stage2_train.log 2>&1 &

# CPU 快速测试
$PYTHON scripts/train_stage2.py --device cpu --num-envs 4 --max-steps 100
```

验收标准: GPU ep_len ≥ 270, latent MSE ≤ 0.01

---

## 6. 训练命令 — 多动作

### 准备工作：生成站立参考动作

```bash
$PYTHON scripts/create_stand_motion.py
# 生成 IsaacGymEnvs/assets/amp/motions/amp_humanoid_stand.npy
```

### Stage 0: 各动作全身策略

```bash
# Walk (已有，跳过)
# Run
PYTHONUNBUFFERED=1 MOTION_FILE=amp_humanoid_run.npy nohup bash scripts/train_stage0_official.sh 8000 \
    > outputs/stage0_run.log 2>&1 &

# Dance
PYTHONUNBUFFERED=1 MOTION_FILE=amp_humanoid_dance.npy nohup bash scripts/train_stage0_official.sh 8000 \
    > outputs/stage0_dance.log 2>&1 &

# Stand
PYTHONUNBUFFERED=1 nohup bash scripts/train_stage0_stand.sh 8000 \
    > outputs/stage0_stand.log 2>&1 &
```

训练完成后手动复制到 `outputs/checkpoints/stage0/`：
```bash
cp <训练输出路径>/nn/<checkpoint>.pth outputs/checkpoints/stage0/stage0_amp_<动作>_<迭代>.pth
```

### Stage 1 Multi: 多动作 Teacher

```bash
# ★ Realistic 模式 (推荐, obs=16D, priv=117D)
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage1_multi.py \
    --device cuda:0 --num-envs 4096 --max-epochs 8000 \
    --motions walk run dance stand \
    --no-motion-in-obs --switch-motion \
    > outputs/stage1_multi_realistic_train.log 2>&1 &

# Legacy 模式 (obs=20D, 向后兼容)
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage1_multi.py \
    --device cuda:0 --num-envs 4096 --max-epochs 8000 \
    --motions walk run dance stand \
    --switch-motion --blend-steps 15 \
    > outputs/stage1_multi_switch_train.log 2>&1 &

# Stage 1 Multi 可视化 (使用 evaluate_stage2_multi.py 的 teacher 模式)
$PYTHON scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 1 --visualize \
    --checkpoint outputs/checkpoints/stage1_multi/best.pth --mode teacher
```

### Stage 2 Multi: 多动作 Student

```bash
# ★ Realistic 模式 (推荐, 需要对应的 realistic Stage 1)
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage2_multi.py \
    --device cuda:0 --num-envs 4096 \
    --motions walk run dance stand \
    --no-motion-in-obs --switch-motion \
    --teacher-ckpt outputs/checkpoints/stage1_multi_realistic/best.pth \
    > outputs/stage2_multi_realistic_train.log 2>&1 &

# Legacy 模式 (向后兼容)
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage2_multi.py \
    --device cuda:0 --num-envs 4096 \
    --motions walk run dance stand \
    --switch-motion --blend-steps 15 \
    --teacher-ckpt outputs/checkpoints/stage1_multi_switch/best.pth \
    > outputs/stage2_multi_switch_train.log 2>&1 &
```

---

## 7. 评估命令

### 单动作评估 + 步态分析

```bash
# 无头评估 (统计 ep_len, 存活率)
$PYTHON scripts/evaluate_stage2.py --device cuda:0 --num-envs 256

# 实时可视化 (Isaac Gym 窗口 + matplotlib 关节角度)
$PYTHON scripts/evaluate_stage2.py --device cuda:0 --num-envs 1 --visualize
```

### 多动作 Per-Motion 评估

```bash
# 全部动作
$PYTHON scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 256

# 指定动作
$PYTHON scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 256 --motion walk

# 可视化
$PYTHON scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 1 --visualize
```

### 动作序列评估

```bash
# 默认: smooth transition (推荐)
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --num-envs 64

# 指定单个序列
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --sequence walk-stop-walk

# 使用 realistic checkpoint
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 \
    --checkpoint outputs/checkpoints/stage2_multi_realistic/best.pth

# 调整过渡帧数
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --transition-frames 15

# 使用旧版 soft-reset (瞬移)
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --soft-reset --no-smooth-transition

# 强制指定观测模式 (覆盖 checkpoint 自动检测)
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --no-motion-in-obs
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --motion-in-obs

# 输出: outputs/eval_multi_motion_results.json
```

可用序列: `walk-stop-walk`, `walk-run-walk-stop-run`, `dance-stand-dance`, `full-cycle`

### 步态分析 (膝关节/踝关节角度+力矩对比)

```bash
# 全部动作分析
$PYTHON scripts/analyze_gait.py --device cuda:0 \
    --checkpoint outputs/checkpoints/stage2_multi_switch/best.pth

# 指定动作
$PYTHON scripts/analyze_gait.py --device cuda:0 --motions walk run stand

# 输出: outputs/gait_analysis/
#   ├── gait_data.npz           # 原始数据
#   ├── knee_comparison.png     # 膝关节角度+力矩 (全动作对比)
#   ├── ankle_comparison.png    # 踝关节角度+力矩
#   ├── hip_comparison.png      # 髋关节角度+力矩
#   └── root_height.png         # 重心高度
```

---

## 8. 可视化

> 需要图形界面 (X11/显示器)，SSH 需 X11 转发或 VNC。

### ★ 交互式可视化 (键盘切换动作)

```bash
# 默认: smooth transition (推荐, 平滑过渡无闪烁)
$PYTHON scripts/interactive_visualize.py --device cuda:0

# 使用 realistic 模型
$PYTHON scripts/interactive_visualize.py --device cuda:0 \
    --checkpoint outputs/checkpoints/stage2_multi_realistic/best.pth

# 调整过渡速度 (帧数越少越快, 默认10帧 ≈ 0.33秒)
$PYTHON scripts/interactive_visualize.py --device cuda:0 --transition-frames 20

# 使用旧版 hard-reset (瞬间切换, 有闪烁)
$PYTHON scripts/interactive_visualize.py --device cuda:0 --hard-reset

# 指定初始动作
$PYTHON scripts/interactive_visualize.py --device cuda:0 --initial-motion stand
```

**键盘控制**:
| 按键 | 动作 |
|------|------|
| W | Walk (行走) |
| R | Run (奔跑) |
| D | Dance (跳舞) |
| S | Stand (站立) |
| Q | Quit (退出) |

Agent 倒地后自动恢复 (hard-reset)。Episode 极长 (100000步)，不会自动 reset。

**过渡模式**:
| 模式 | 参数 | 效果 | 存活率 |
|------|------|------|--------|
| Smooth (默认) | `--transition-frames 10` | 10帧平滑插值, 无闪烁 | ~80-92% |
| Hard reset | `--hard-reset` | 1帧瞬移, 有闪烁 | ~85-95% |

### Stage 0 (全身策略)

```bash
cd IsaacGymEnvs/isaacgymenvs

# Walk
$PYTHON train.py task=HumanoidAMP train=HumanoidAMPPPO \
    task.env.motion_file=amp_humanoid_walk.npy \
    test=True num_envs=4 headless=False \
    checkpoint=../../outputs/checkpoints/stage0/stage0_amp_walk_5050.pth

# Run / Dance / Stand (类似，替换 motion_file 和 checkpoint)
cd ../..
```

### Stage 2 单动作 (假肢策略)

```bash
$PYTHON scripts/evaluate_stage2.py --device cuda:0 --num-envs 1 --visualize
```

### Stage 2 多动作 (假肢策略)

```bash
# Per-Motion 可视化
$PYTHON scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 1 --visualize

# 指定动作可视化
$PYTHON scripts/evaluate_stage2_multi.py --device cuda:0 --num-envs 1 --visualize --motion walk
```

### 动作序列可视化

```bash
# 可视化单 agent 连续动作切换 (smooth transition, 推荐)
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --visualize

# 可视化指定序列
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --visualize --sequence walk-stop-walk
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --visualize --sequence full-cycle

# 使用旧版 soft-reset 可视化 (对比用)
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 --visualize --soft-reset --no-smooth-transition
```

> `--visualize` 模式自动设置 `num_envs=1`（单 agent 连续切换），
> 默认使用 smooth transition (10帧平滑过渡)。

---

## 9. 两种观测模式说明

### Legacy 模式 (`motion_in_obs=True`, 旧默认)

```
obs (20D) = proprio(16D) + motion_onehot(4D)  ← 假肢知道动作类型
priv_info (113D) = full_body + GRF + contacts
proprio_hist (30×20D)
```

对应模型: `stage1_multi/`, `stage1_multi_switch/`, `stage2_multi/`, `stage2_multi_switch/`

### Realistic 模式 (`motion_in_obs=False`, 新推荐)

```
obs (16D) = proprio(16D)  ← 假肢只有传感器信息
priv_info (117D) = full_body + GRF + contacts + motion_onehot(4D)
proprio_hist (30×16D)
```

对应模型: `stage1_multi_realistic/`, `stage2_multi_realistic/`

**为什么需要 Realistic 模式?**  
真实假肢没有命令通道告知当前执行什么动作。假肢只能通过传感器感知人体运动，
Student 必须从 30帧历史运动模式推断动作类型。Teacher 通过 priv_info 获知动作类型。

**切换方式**: 所有训练/评估脚本均支持 `--no-motion-in-obs` 标志。
评估脚本可从 checkpoint 自动检测模式。

---

## 10. 训练监控规范

> **每次启动训练后，必须用 TensorBoard 或脚本检查训练曲线，确认收敛后再继续下一步。**

### 快速检查训练进度（命令行）

```bash
# 通用检查脚本 — 传入 tb 事件文件所在目录
$PYTHON -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os, sys
d = sys.argv[1]
events = glob.glob(os.path.join(d, 'events.out.tfevents.*'))
ea = EventAccumulator(events[0]); ea.Reload()
for tag in ea.Tags().get('scalars', []):
    vals = ea.Scalars(tag)
    if vals:
        last = vals[-1]
        print(f'{tag}: {last.value:.4f} (step {last.step})')
" <TB_DIR>
```

### TensorBoard 可视化

```bash
# Stage 1/2 多动作 Realistic
tensorboard --logdir=outputs/stage1_multi_walk_run_dance_stand_*/tb --port=6009
tensorboard --logdir=outputs/stage2_multi_realistic_*/tb --port=6010

# Stage 1/2 多动作 Switch (legacy)
tensorboard --logdir=outputs/stage1_multi_switch/tb --port=6011
tensorboard --logdir=outputs/stage2_multi_switch/tb --port=6012
```

| 指标 | 含义 | Stage | 收敛标准 |
|------|------|-------|---------|
| rewards/iter | 每 epoch 平均 reward | 0 | >250 |
| episode_lengths/iter | 平均 episode 长度 | 0 | >250 |
| loss/mse | DAgger MSE loss | 1 | <0.01 |
| reward/avg_ep_len | 平均 episode 长度 | 1 | >220 (switch模式) |
| episode_rewards/step | 滑动平均 reward | 2 | >600 |
| episode_lengths/step | 滑动平均 episode 长度 | 2 | >250 |
| latent_mse/step | MSE(z_student, z_teacher) | 2 | <0.05 |

---

## 11. 训练控制

### 查看训练状态

```bash
ps aux | grep 'train_stage\|train.py' | grep -v grep
nvidia-smi
```

### 中断训练

```bash
# 1. 找到 PID
ps aux | grep 'train_stage' | grep -v grep

# 2. 停止训练
kill <PID>
```

> ⚠️ 中断后 checkpoint 不会丢失。Stage 1 每 500 epochs 自动保存, Stage 2 每 50M steps 保存。

---

## 12. 已知问题与注意事项

| 问题 | 解决方案 |
|------|---------|
| Isaac Gym 只支持 Python 3.8 | 使用 `proknee_tc` conda 环境 |
| Isaac Gym 必须先于 torch 导入 | `import isaacgym` 在所有脚本首行 |
| 两个 GPU 训练同时运行会崩溃 | 确保只启动一个训练进程 |
| CPU PhysX ≠ GPU PhysX | **所有评估必须在 GPU 上进行** |
| Body policy OOD (动作切换) | 使用 `smooth_transition_to_motion()` (平滑) 或 `soft_reset_to_motion()` (瞬移) |
| 动作切换不连续 (teleport) | 使用 `--transition-frames 10` 平滑过渡 (默认已开启) |
| `PYTHONUNBUFFERED=1` | nohup 训练时需要，否则 stdout 被缓冲 |

---

## 13. 代码撰写规范

| 要求 | 说明 |
|------|------|
| Isaac Gym 导入顺序 | `import isaacgym` 必须在 `import torch` 之前 |
| 维度常量 | 单动作从 `constants.py`，多动作从 `constants_multi.py` 导入 |
| 模型存储 | 训练完成后复制最佳模型到 `outputs/checkpoints/stageN/` |
| 评估要求 | GPU 上评估，CPU 结果不作数 |
| 多动作扩展 | 不修改 `proknee_base.py` 和 `constants.py`，新建文件继承 |
| 观测模式 | Realistic (obs=16D) 为推荐默认；Legacy (obs=20D) 保持向后兼容 |

---

## 14. 当前训练状态

### 单动作 (平地行走) ✅ 全部完成

| Stage | 最佳 checkpoint | 关键指标 |
|-------|----------------|---------|
| Stage 0 | `stage0/stage0_amp_walk_5050.pth` | ep_len=287.2 |
| Stage 1 | `stage1/best.pth` | ep_len=284.7 |
| Stage 2 | `stage2/best.pth` | ep_len=279.3, MSE=0.005 |

### 多动作 Legacy (obs=20D) ✅ 全部完成

| Stage | 最佳 checkpoint | 关键指标 |
|-------|----------------|---------|
| Stage 0 (all) | `stage0/stage0_amp_*` | ep_len=287~299 |
| Stage 1 Multi Switch | `stage1_multi_switch/best.pth` | ep_len=242.6 |
| Stage 2 Multi Switch | `stage2_multi_switch/best.pth` | reward=482.2 |

### 多动作 Realistic (obs=16D) ✅ 全部完成

| Stage | 最佳 checkpoint | 状态 |
|-------|----------------|------|
| Stage 0 (all) | 共享 legacy Stage 0 | ✅ 完成 |
| Stage 1 Multi Realistic | `stage1_multi_realistic/best.pth` | ✅ 完成 (ep_len=243.9, 3.74h) |
| Stage 2 Multi Realistic | `stage2_multi_realistic/best.pth` | ✅ 完成 (reward=466.53, 500M steps) |

### Per-Motion 评估 (Realistic 模型, 256 envs, 50 episodes)

| 动作 | Avg Ep Len | Survival |
|------|-----------|----------|
| Walk | 262.8 | 79.6% |
| Run | 262.5 | 79.7% |
| Dance | 262.0 | 79.2% |
| Stand | 263.2 | 80.0% |

### 动作序列评估 (Realistic 模型, soft-reset)

| 序列 | Survival |
|------|----------|
| walk→stand→walk | 86.7% |
| walk→run→walk→stand→run | 84.1% |
| dance→stand→dance | 94.7% |
| full-cycle | 94.8% |

### 动作序列评估 (Realistic 模型, smooth transition, 10帧)

| 序列 | Survival |
|------|----------|
| walk→stand→walk | 81.2% |
| walk→run→walk→stand→run | 80.0% |
| dance→stand→dance | 92.2% |
| full-cycle | 90.2% |

### 动作序列评估 (Legacy Switch 模型)

| 序列 | Soft-Reset Survival |
|------|-------------------|
| walk→stand→walk | 85.6% |
| walk→run→walk→stand→run | 84.4% |
| dance→stand→dance | 96.2% |
| full-cycle | 94.8% |

---

## 15. Unified 速度控制策略 (Phase 11+)

> 📋 **开发中**: 用单一策略替代 4 个 per-motion 策略，通过连续速度命令控制

### 15.1 概念

| 对比 | Per-Motion (Phase 0-10) | Unified (Phase 11+) |
|------|------------------------|---------------------|
| Body Policy | 4 个独立模型 | 1 个速度参数化模型 |
| 动作切换 | soft-reset / smooth-transition | 自然过渡 |
| 控制方式 | 离散动作类型 (W/R/D/S) | 连续速度命令（标量 v；数值范围与训练采样见 15.1 速度映射） |
| 键盘控制 | W=Walk, R=Run, S=Stand | ↑/↓ 调节速度 |

速度映射（与当前任务代码 / YAML 一致）：

- **标准 `HumanoidAMPUnified`**：环境采样的目标前向速度只在 **`0.0`、`1.0`、`2.5` m/s** 三档（`humanoid_amp_unified.py`），分别对应站 / 走 / 跑档位，与 `motion_file` 里 walk、run 等参考动作配套；不是连续均匀采样。
- **HumanMimic `HumanoidAMPUnifiedHumanMimic_phase1`**：目标速度为 **0.0～3.0 m/s、步长 0.1** 的网格（`HumanoidAMPUnifiedHumanMimic_phase1.yaml` 中 `velocityGridStep` / `velocityMax`）；walk / run 参考速度带为 **`walkVelMin`～`walkVelMax`**（默认约 0.5～1.5）、**`runVelMin`～`runVelMax`**（默认约 2.0～3.0）；速度绝对值小于 **`standVelocityEpsilon`**（默认 0.05）时按站立侧处理。

### 15.2 Stage 0 Unified 训练 

在仓库根目录执行；环境依赖见 `docs/ENVIRONMENT_RLLEG.md`。Stage0 Unified 分两种，**训练时二选一**：标准 Unified 用 **`scripts/train_stage0_unified.sh`**；HumanMimic 用 **`scripts/train_stage0_unified_humanmimic.sh`**。两者都会 `source scripts/activate_proknee_tc_env.sh`，`cd IsaacGymEnvs/isaacgymenvs`，并按 `scripts/stage0_batch_hydra.sh` 自动设置 `train.params.config.minibatch_size` 与 `train.params.config.amp_minibatch_size`（与 `num_envs`、horizon 对齐）。默认 `num_envs=4096`，显存不足可 `export STAGE0_NUM_ENVS=2048` 或 `1024`。

**标准 Unified（`train.py`，`task=HumanoidAMPUnified`，`train=HumanoidAMPUnifiedPPO`）**

参考动作由 `IsaacGymEnvs/isaacgymenvs/cfg/task/HumanoidAMPUnified.yaml` 的 `motion_file` 决定（当前为 `multi_walk_run.yaml`）。

```bash
# 训练（第二个参数可选：断点续训；路径相对 IsaacGymEnvs/isaacgymenvs 或写绝对路径）
PYTHONUNBUFFERED=1 nohup bash scripts/train_stage0_unified.sh 10000 \
    > outputs/stage0_unified.log 2>&1 &
# PYTHONUNBUFFERED=1 nohup bash scripts/train_stage0_unified.sh 10000 \
#     runs/HumanoidAMPUnified_<时间>/nn/<名>.pth \
#     > outputs/stage0_unified_resume.log 2>&1 &
```

**HumanMimic Stage0（`train_humanmimic_unified.py`，`task=HumanoidAMPUnifiedHumanMimic_phase1`，`train=HumanoidAMPUnifiedHumanMimicPPO`）**

与标准 Unified 的检查点、网络与入口不同，勿混用。

```bash
PYTHONUNBUFFERED=1 nohup bash scripts/train_stage0_unified_humanmimic.sh 10000 \
    > outputs/stage0_unified_humanmimic.log 2>&1 &
```

**从「不换速」检查点续训换速版（自动选 `runs/HumanoidAMPUnifiedHumanMimic_*/nn/` 下最新 `.pth`，并强制 `enableCmdSwitch` 等）**：`bash scripts/train_stage0_unified_humanmimic_finetune_switch.sh [max_iterations] [可选：显式checkpoint路径]`（默认 `max_iterations=16000`）；或 `HUMANMIMIC_BASE_CKPT=/path/to/xxx.pth bash ...`。续训时 **`max_iterations` 必须大于基座检查点里已完成的 epoch**（例如基座已训满 8000 仍设 8000 会立刻 `MAX EPOCHS NUM!` 退出），应改为 12000、16000 等。

**验收标准**: ep_len ≥ 250
**当前状态**: 在进行中

### 15.3 Stage 0 Unified 播放检查点（可视化）

在**仓库根目录**执行；需本机图形界面（脚本内为 `test=True`、`headless=False`）。环境与训练一致时依赖 `docs/ENVIRONMENT_RLLEG.md` 中的 conda；`play_stage0_unified_humanmimic.sh` 会 `source scripts/activate_proknee_tc_env.sh`。

**与检查点对应关系（勿混用）**

| 训练产物 | 播放脚本 | 入口 |
|---------|----------|------|
| `HumanoidAMPUnified_*/nn/*.pth` | `scripts/play_stage0_unified.sh` | `train.py`，`task=HumanoidAMPUnified` |
| `HumanoidAMPUnifiedHumanMimic_*/nn/*.pth` | `scripts/play_stage0_unified_humanmimic.sh` | `train_humanmimic_unified.py`，`task=HumanoidAMPUnifiedHumanMimic_phase1` |

**标准 Unified（`play_stage0_unified.sh`）**

- **用法**：`bash scripts/play_stage0_unified.sh`；可选第一个参数为 checkpoint（`runs/.../*.pth`，或仓库根下的 `outputs/checkpoints/stage0/...` 等相对路径，脚本会解析到 `IsaacGymEnvs/isaacgymenvs` 下）。
- **未传 checkpoint 时**：依次尝试 `runs/HumanoidAMPUnified_*/nn/` 中最新 `.pth`，否则 `outputs/checkpoints/stage0/stage0_unified_1800.pth` 或同目录下最新的 `stage0_unified*.pth`。
- **环境变量**：`MOTION_FILE`（默认 `multi_walk_run.yaml`，**须与训练该权重时一致**）；`NUM_ENVS`（默认 4，显存紧可用 `NUM_ENVS=1`）。
- **等价命令要点**：`train.py` + `task.env.motion_file="$MOTION_FILE"` + `checkpoint=...`。

**HumanMimic Stage0（`play_stage0_unified_humanmimic.sh`）**

- **用法**：`bash scripts/play_stage0_unified_humanmimic.sh`；可选第一个参数为 checkpoint（支持仓库根或 `isaacgymenvs` 下的相对路径，脚本会 `realpath` 成绝对路径）。
- **未传 checkpoint 时**：取 `runs/HumanoidAMPUnifiedHumanMimic_*/nn/` 中按时间最新 `.pth`。
- **环境变量**：`NUM_ENVS`（默认 4）；脚本附带 Hydra 追加项 `+train.params.config.torch_compile=False`（结构化配置里无该键时需加 `+`）。
- **等价命令要点**：`train_humanmimic_unified.py` + `train=HumanoidAMPUnifiedHumanMimicPPO` + `checkpoint=...`。

**HumanMimic Stage0 键盘控速（与 Stage2 `interactive_unified.py` 类似）**

- **用法**：`bash scripts/play_humanmimic_unified_interactive.sh`；可选第一个参数为 checkpoint。默认 `NUM_ENVS=1`。
- **回合长度（回原点间隔）**：脚本默认 `PLAY_EPISODE_LENGTH=900`（Hydra：`task.env.episodeLength`），比训练 YAML 常见的 300 更长，**超时**才回到起点；仍会因摔倒等提前终止。需要更久可 `export PLAY_EPISODE_LENGTH=1500` 等。
- **行为**：打开 `manualVelocityControl` 与 `interactiveKeyboard`，关闭回合内随机换速；初速与每次倒地重置均为训练同款 **0～3 m/s、步长 0.1** 的随机网格；↑/↓ 每次 **±0.1 m/s**；终端用 `\r` **实时刷新**当前 `v_cmd`；`W`/`F`/`M`/`S` 等快捷；`F`≈2.5 m/s（快跑），因 viewer 的 `R` 已用于录屏；`Q` 退出。
- **可选 Hydra**：`task.env.interactiveInitialVelocity`、`task.env.interactiveVelocityStep`。

```bash
# 标准 Unified：自动找 checkpoint 或显式指定
bash scripts/play_stage0_unified.sh
bash scripts/play_stage0_unified.sh runs/HumanoidAMPUnified_<时间>/nn/<名>.pth
MOTION_FILE=multi_walk_run.yaml NUM_ENVS=1 bash scripts/play_stage0_unified.sh
bash scripts/play_stage0_unified.sh outputs/checkpoints/stage0/stage0_unified_1800.pth

# HumanMimic：自动找最新或显式指定
bash scripts/play_stage0_unified_humanmimic.sh
bash scripts/play_stage0_unified_humanmimic.sh runs/HumanoidAMPUnifiedHumanMimic_<时间>/nn/<名>.pth
NUM_ENVS=1 bash scripts/play_stage0_unified_humanmimic.sh

# HumanMimic：键盘控速播放（↑/↓、W/F/M/S、Q 退出）
bash scripts/play_humanmimic_unified_interactive.sh
bash scripts/play_humanmimic_unified_interactive.sh runs/HumanoidAMPUnifiedHumanMimic_<时间>/nn/<名>.pth
```

### 15.4 保存 Unified Stage 0 Checkpoint

```bash
# 标准 Unified 输出目录
ls -la IsaacGymEnvs/isaacgymenvs/runs/HumanoidAMPUnified_*/nn/

# HumanMimic Stage0 输出目录
ls -la IsaacGymEnvs/isaacgymenvs/runs/HumanoidAMPUnifiedHumanMimic_*/nn/

# 复制到标准位置（按需命名）
cp IsaacGymEnvs/isaacgymenvs/runs/HumanoidAMPUnified_<timestamp>/nn/<best>.pth \
    outputs/checkpoints/stage0/stage0_unified.pth
```

### 15.5 Stage 1 Unified 训练 ✅ (已完成 2026-03-21)

```bash
# 训练 Unified Stage 1 Teacher (DAgger)
# 结果: ep_len=268.4 @ epoch 2267
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage1_unified.py \
    --device cuda:0 --num-envs 4096 --max-epochs 8000 \
    --body-policy outputs/checkpoints/stage0/stage0_unified_1800.pth \
    > outputs/stage1_unified_train.log 2>&1 &
```

### 15.6 Stage 2 Unified 训练 ✅ (已完成 2026-03-21)

```bash
# 训练 Unified Stage 2 Student (蒸馏)
# 结果: reward=689.83, latent_mse=0.013
PYTHONUNBUFFERED=1 nohup $PYTHON scripts/train_stage2_unified.py \
    --device cuda:0 --num-envs 4096 \
    --teacher-ckpt outputs/checkpoints/stage1_unified/best.pth \
    --body-policy outputs/checkpoints/stage0/stage0_unified_1800.pth \
    > outputs/stage2_unified_train.log 2>&1 &
```

**检查点保存**：每次训练会新建时间戳目录 `outputs/stage2_unified_<时间戳>/checkpoints/`，训练过程中按 reward 更新 **`best.pth`**，另有 `last.pth`、`step_*` 等；训练正常结束时会将本次 run 的 **`best.pth` 复制到固定位置 `outputs/checkpoints/stage2_unified/best.pth`**（便于 `interactive_unified.py` 与文档示例统一引用）。若尚未产生过 `best`（例如极早中断），可改用同目录下 `last.pth` 或对应 `step_*.pth`。

### 15.7 Unified 交互式可视化 ✅

```bash
# 使用 Unified Stage 2 模型进行交互式速度控制（Student 用上面的 best.pth）
$PYTHON scripts/interactive_unified.py --device cuda:0 \
    --checkpoint outputs/checkpoints/stage2_unified/best.pth \
    --body-policy outputs/checkpoints/stage0/stage0_unified_1800.pth

# 指定初始速度
$PYTHON scripts/interactive_unified.py --device cuda:0 --initial-velocity 1.0
```

**键盘控制**:
| 按键 | 功能 |
|------|------|
| ↑ | 增加目标速度 (+0.25) |
| ↓ | 减少目标速度 (-0.25) |
| W | 设置速度 = 1.0 (Walk) |
| R | 设置速度 = 2.5 (Run) |
| S | 设置速度 = 0.0 (Stand) |
| 0-9 | 直接设置速度 (0=0.0, 5=1.25, 9=2.25) |
| Q | 退出 |

---

## 16. 快速命令参考

### Per-Motion 系统（已完成）

```bash
# 环境配置
export LD_LIBRARY_PATH=/home/user/anaconda3/envs/proknee_tc/lib:$LD_LIBRARY_PATH
PYTHON=/home/user/anaconda3/envs/proknee_tc/bin/python
cd /home/user/Workspace/ProKnee/ProKnee-HoraStyle

# 键盘交互可视化 (推荐使用 realistic 模型)
$PYTHON scripts/interactive_visualize.py --device cuda:0 \
    --checkpoint outputs/checkpoints/stage2_multi_realistic/best.pth

# 评估序列切换
$PYTHON scripts/evaluate_motion_sequences.py --device cuda:0 \
    --checkpoint outputs/checkpoints/stage2_multi_realistic/best.pth
```

### Unified 系统 ✅

```bash
# Stage 0 Unified 播放（test 模式，见 §15.3）
bash scripts/play_stage0_unified.sh
bash scripts/play_stage0_unified_humanmimic.sh

# Unified 交互式速度控制 (↑↓调速, W/R/S预设, 0-9直接设值)
$PYTHON scripts/interactive_unified.py --device cuda:0

# 指定初始速度
$PYTHON scripts/interactive_unified.py --device cuda:0 --initial-velocity 2.0
```

---

*更新: 2026-03-20 — 添加 Unified 速度控制策略章节*
*更新: 2026-03-21 — Unified Stage 1/2 训练完成*

