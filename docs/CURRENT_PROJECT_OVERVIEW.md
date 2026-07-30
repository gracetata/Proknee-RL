# 假肢训练架构：MuJoCo Warp + HORA PPO

更新日期：2026-07-30。本文是当前训练方案、代码入口和执行命令的唯一主文档。

## 1. 项目路径

```text
本机：/home/user/Workspace/Proknee-RL-muscle
A100：/workspace/Proknee-RL-muscle
GitHub：gracetata/Proknee-RL，muscle 分支
```

新架构位于：

```text
torque_replay_training/
├── configs/
│   ├── flat_walk_warp_v2.yaml          4096-env 正式配置
│   └── flat_walk_warp_v2_smoke.yaml    smoke 配置
├── src/torque_replay_training/warp_v2/
│   ├── replay.py                       compact 数据 GPU 打包
│   ├── env.py                          MuJoCo Warp 向量环境
│   ├── observations.py                 96维可部署 Actor 观测
│   ├── rewards.py                      无模仿项任务奖励
│   ├── curriculum.py                   replay torque 退火
│   └── ppo.py                          HORA 派生非对称 PPO
└── scripts/
    ├── train_warp_v2.py                训练入口
    ├── evaluate_warp_v2_policy.py      beta=0 CPU 验证
    ├── run_warp_v2_smoke_a100.sh       4096-env smoke
    ├── start_warp_v2_training_a100.sh  正式训练
    └── warp_v2_training_status.py      只读状态
```

`data/replay_model/musclemimic_replay.xml` 是跨MuJoCo版本的中间模型；
安装脚本使用MuJoCo 3.5将其重新编译为
`musclemimic_replay_warp_3_5.mjb`。不能直接用MuJoCo 3.5加载旧的3.4 MJB。

旧的 `hora_env.py` 是16个 CPU MuJoCo环境的 residual-policy 原型，只保留用于回归，
不再作为正式训练入口。

## 2. 数据和物理环境

平地训练使用 `flat_walk_compact_v1`：

- 281条已验证轨迹；
- 229条训练，52条跨受试者验证；
- `qpos[89]`、`qvel[88]`、`qacc_warmstart[88]`；
- 每个100 Hz控制步包含5组500 Hz的全身广义力 `qfrc_actuator[88]`。

训练时：

1. 4096个world分别选择轨迹和随机起点。
2. 健康人体84个DOF回放记录的500 Hz广义力。
3. 左膝、左踝、左距下、左MTP由策略控制。
4. 重力、阻尼、碰撞、接触和摩擦由MuJoCo Warp重新计算。
5. 轨迹数据、仿真状态、观测、rollout和PPO均常驻物理GPU 5。

正式环境为：

```text
MuJoCo 3.5.0
mujoco-warp 3.5.0
warp-lang 1.12.1
PyTorch 2.7.1+cu126
4096 worlds
100 Hz policy
500 Hz physics
```

CPU MuJoCo继续用于原始数据验证、最终策略验证和本机可视化。

## 3. Replay torque 只初始化探索

没有Behavior Cloning，也没有imitation loss或imitation reward。

策略输出四维归一化动作：

```text
a_policy ∈ [-1,1]^4
torque_limit = [260, 280, 80, 15] Nm
```

训练初期：

```text
a_replay  = replay_prosthesis_torque / torque_limit
a_execute = clip(beta * a_replay + a_policy, -1, 1)
torque    = a_execute * torque_limit
```

`a_policy`均值层零初始化，所以初始探索以记录力矩为中心。退火为：

```text
0–2M steps:     beta = 1
2M–20M steps:   beta 线性下降到0
20M–100M steps: beta = 0
```

回放假肢力矩不会进入：

- Actor观测；
- Critic观测；
- reward；
- PPO loss；
- checkpoint中的最终策略输入。

正式CPU验证强制 `beta=0`，并拒绝仍带非零beta的checkpoint。

## 4. 策略观测和动作

Actor每帧24维，堆叠最近4帧，共96维：

| 观测 | 维数 |
|---|---:|
| 假肢关节位置、速度 | 4 + 4 |
| 骨盆坐标系投影重力 | 3 |
| 骨盆坐标系线速度、角速度 | 3 + 3 |
| 目标 `vx, vy, yaw_rate` | 3 |
| 上一次策略动作 | 4 |

Actor不读取未来帧、motion ID、相位、参考姿态或回放假肢力矩。

Critic使用非对称privileged observation：

```text
Actor 96维历史
+ 当前全身 qpos[89]
+ 当前全身 qvel[88]
+ root up-z
+ 双脚接触
+ episode progress
= 277维
```

Actor网络：

```text
96 → 256 → 256 → 128 → 4
```

Critic网络：

```text
277 → 512 → 256 → 128 → 1
```

策略是tanh-Gaussian。PPO保存未变换的Gaussian样本并使用tanh Jacobian修正后的
log-probability，环境执行有界动作，避免旧实现中“保存未裁剪动作、执行裁剪动作”的偏差。

## 5. Reward

Reward只评价稳定行走，不模仿回放假肢动作：

```text
R =
  + 1.00 alive
  + 1.00 upright
  + 0.50 root_height
  + 1.00 root_velocity_tracking
  + 0.50 yaw_rate_tracking
  - 0.10 foot_slip
  - 0.25 both_feet_airborne
  - 2e-6 prosthesis_torque_l2
  - 1e-7 prosthesis_qacc_l2
  - 0.006 action_rate_l2
  - 1.00 joint_limit_violation
  - 200.00 fall
```

指数项：

```text
upright      = exp(-gravity_error² / 0.10²)
root_height  = exp(-height_error² / 0.05²)
root_velocity= exp(-velocity_error² / 0.40²)
yaw_rate     = exp(-yaw_rate_error² / 0.40²)
```

跌倒判据：

```text
root height < 0.55 m
root up-z   < 0.35
状态出现 NaN/Inf
MuJoCo contact/constraint buffer overflow
```

非超时跌倒额外扣200。送入PPO前乘 `reward_scale=0.01`；TensorBoard同时记录每个
原始加权reward、总return、fall rate和动作饱和率。

## 6. PPO

PPO核心复用HORA的MLP、RunningMeanStd、GAE、clipped PPO、KL学习率调节、
checkpoint和TensorBoard思路，并改成非对称Actor/Critic与tanh-Gaussian。

正式参数：

```text
num_envs             4096
horizon                32
batch_size         131072
minibatch_size      32768
mini_epochs             4
gamma                 0.99
GAE lambda            0.95
clip epsilon          0.20
learning_rate         3e-4
target KL             0.02
entropy coefficient   5e-4
critic coefficient     2.0
gradient norm           1.0
total transitions      100M
```

Loss：

```text
L = clipped_policy_loss
  + 0.5 * 2.0 * clipped_value_loss
  - 5e-4 * Gaussian_entropy
```

不存在BC loss、torque imitation loss或AMP discriminator loss。

## 7. 环境和命令

### 7.1 本机测试

本机可运行全部CPU单元测试：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python -m pytest torque_replay_training/tests
```

本机可视化完整回放，以下命令只能在有图形界面的本机运行：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset 'torque_replay_training/data/flat_walk_compact_v1/*.npz'
```

### 7.2 A100安装

```bash
ssh -p 6029 root@39.105.12.60
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/setup_warp_v2_env_a100.sh
```

### 7.3 GPU占用检查

任何训练或GPU smoke前必须运行：

```bash
cd /workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5
```

返回非零表示物理GPU 5有人使用，必须等待。禁止使用GPU 0–4、6、7。

### 7.4 4096-env smoke

```bash
cd /workspace/Proknee-RL-muscle
NUM_ENVS=4096 MAX_AGENT_STEPS=65536 \
  bash torque_replay_training/scripts/run_warp_v2_smoke_a100.sh
```

smoke必须完成两次PPO update、保存checkpoint且没有OOM、NaN或buffer overflow。

### 7.5 正式训练

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/start_warp_v2_training_a100.sh
```

状态：

```bash
cd /workspace/Proknee-RL-muscle
.venv-warp/bin/python \
  torque_replay_training/scripts/warp_v2_training_status.py
```

TensorBoard：

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/start_tensorboard_a100.sh
ssh -N -L 6011:127.0.0.1:6011 -p 6029 root@39.105.12.60
```

浏览器打开 `http://127.0.0.1:6011`。

## 8. 验收

正式训练结束后自动使用CPU MuJoCo和 `beta=0` 执行：

1. 52条validation轨迹逐条完整运行；
2. 1000个validation随机轨迹/起点episode；
3. 两组验证均要求0次跌倒；
4. 力矩始终在硬限制内；
5. 记录root高度、up-z、动作饱和率和足部滑移p95。

只有CPU验证通过才创建complete marker。seed 0通过后，再在同一张物理GPU 5上按顺序
训练seed 1和seed 2。

## 9. 当前实现状态

- warp-v2独立代码、配置、训练/验证/状态入口已建立；
- 新增核心测试7项，项目测试共24项全部通过；
- compact replay CPU打包和索引已通过真实数据检查；
- 本机RTX 4090已完成4096-world、131,072 transitions的一次完整PPO更新：
  约11,122 env-steps/s、0跌倒、0 buffer overflow；
- A100的4096-world GPU smoke尚未执行：物理GPU 5当前存在其他计算进程，不能抢卡；
- GPU 5空闲后，先运行4096-env smoke，再启动100M正式训练。
