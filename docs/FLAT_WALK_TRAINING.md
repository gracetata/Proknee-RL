# 平地假肢训练：官方 MJLAB + RSL-RL PPO

> 当前正式方案，更新于 2026-07-31。旧 HORA/自写 Warp PPO 已停止。

## 1. 结论

- 环境：官方 `MJLAB 1.5.3` 的 `ManagerBasedRlEnv`。
- 物理：`MuJoCo 3.10.0 + mujoco-warp 3.10.0.3`，4096 个 GPU 并行环境。
- 算法：官方 MJLAB 封装的 `RSL-RL 5.4.0 PPO`；没有复制或自写 PPO。
- 硬件：A100 物理 GPU 5，进程内为 `cuda:0`；禁止 GPU 0–4、6、7。
- 动作：左膝 1 DOF、左踝/距下/MTP 3 DOF，共 4 维关节力矩。
- 目标：非假肢身体继续回放记录的广义力；策略最终独立控制四个假肢 DOF，并保持人体不摔倒。
- 不使用 AMP、behavior cloning、轨迹模仿 reward 或 replay torque loss。

代码入口：

- 环境配置：`torque_replay_training/src/torque_replay_training/mjlab_task/config.py`
- 回放/假肢动作：`torque_replay_training/src/torque_replay_training/mjlab_task/action.py`
- 观测/reward/done：`torque_replay_training/src/torque_replay_training/mjlab_task/mdp.py`
- PPO 启动：`torque_replay_training/scripts/train_mjlab_prosthesis.py`

## 2. 纯回放与 RL 回放是否一致

两者相同的核心：

1. 每个 100 Hz 控制步包含 5 个 500 Hz 物理子步。
2. 每个子步把记录的完整 88 维 `qfrc_actuator[t,s,:]` 写入
   `qfrc_applied`。
3. 原模型 354 个 actuator 的 gain/bias 全部置零，不会再次产生主动肌肉力。
4. `qfrc_passive`、重力、约束力、碰撞和地面反力均不回放，由当前 MuJoCo 状态重新计算。
5. 模型的 362 个 tendon stiffness/damping 均为 0，关节 stiffness 为 0；
   仍存在模型本身的 DOF damping（82 个非零，最大 1.05），由 MuJoCo 计算。

因此：没有主动肌肉或被动肌腱弹簧重复施力；存在 MuJoCo 模型的关节阻尼；地面反力是
仿真生成的，不在 NPZ 中。

四个假肢 DOF 的差异：

\[
a_{\mathrm{exec}}=\beta a_{\mathrm{replay}}+(1-\beta)a_{\pi},\qquad
\tau_{\mathrm{prosthesis}}=a_{\mathrm{exec}}\odot[260,280,80,15].
\]

`beta=1` 时随机 policy 对回放没有影响；`beta=0` 时只执行 policy。回放动作只作为探索
初值，不进入 reward 或 loss。

## 3. 已完成的物理审计

| 测试 | 结果 | 解释 |
|---|---:|---|
| 原 MuJoCo 3.4 MJB，纯 `float64` 精确回放 | 0/20 摔倒 | 数据本身可用 |
| 旧训练模型经 MuJoCo 3.5 重编译，完整回放 | 20/20 摔倒 | 旧训练基线已失真 |
| MuJoCo 3.10 CPU，完整回放 | 20/20 摔倒 | 跨版本不再逐步等价 |
| 官方 MJLAB/MJWarp，无辅助长回放 | 21/21 摔倒 | GPU/float32/版本差异会累积 |
| 旧 Warp checkpoint，关闭 replay | 20/20 摔倒 | 旧 policy 未学会独立站立 |
| 新 MJLAB 1-iteration smoke checkpoint，关闭 replay/辅助 | 16/16 摔倒 | 正常的训练初态，并验证评估链路 |

不能把旧失败简单归因于 PPO 未收敛：旧后端在 policy 不参与时就无法维持回放。

MJLAB 正式训练的初期使用随 `beta` 同步衰减的健康关节/root 状态稳定辅助。默认辅助在
512 步测试中 0/8 摔倒，但最大 `qpos` 跟踪误差为 3.33，所以它叫
**assisted replay**，不是精确回放。所有辅助项都乘以 `beta`，在最终 `beta=0`
评估时严格为零。

## 4. 训练阶段

- 0–2M agent transitions：`beta=1`，四维假肢使用健康回放力矩，稳定收集初始数据。
- 2M–20M：`beta` 线性降到 0，policy 逐步接管四个假肢 DOF。
- 20M 以后：`beta=0`；回放只保留非假肢 84 个 DOF 的记录广义力。
- checkpoint 验收：固定 `beta=0`，因此假肢 replay 和全部稳定辅助均关闭。

4096 个环境、每环境 rollout 24 步，每次 PPO iteration 为 98,304 transitions：

- `beta` 保持阶段约 21 iterations；
- 完全接管约在 204 iterations；
- 默认 3000 iterations，共 294,912,000 transitions。

这里的 transition 不是单个环境的 episode step；所以不能与“3000 个 PPO
iterations”直接比较。

## 5. 观测与动作

Actor 每帧 24 维，堆叠 4 帧为 96 维：root 姿态/速度、四个假肢关节位置与速度、速度
命令、上一动作等。Critic 使用 203 维 privileged state。动作是四维归一化力矩，
clamp 到 `[-1,1]` 后按上述上限缩放。

Actor MLP 为 `256-256-128 ELU`；Critic 为 `512-256-128 ELU`，二者使用官方
observation normalization。高斯策略初始标准差为 0.10。

## 6. Reward

每个 policy step：

| 项 | 权重 | 定义 |
|---|---:|---|
| alive | +1.0 | 未结束时常数 |
| upright | +1.0 | root 朝上误差的指数核 |
| minimum height | +0.5 | pelvis 高度 0.55–0.85 m 线性饱和 |
| forward velocity | +1.0 | body-x 速度接近 1 m/s 的指数核 |
| lateral velocity² | -0.10 | 抑制侧滑 |
| yaw rate² | -0.05 | 平地直行抑制偏航 |
| prosthesis torque² | -2e-6 | 四个假肢力矩能耗 |
| prosthesis acceleration² | -1e-7 | 四个假肢 DOF 加速度 |
| action rate² | -0.006 | 动作平滑 |
| joint-limit violation | -1.0 | 四个假肢关节越界 |

跌倒条件：pelvis 高度低于 0.55 m、root up-z 低于 0.35，或状态非有限数。轨迹数据不
出现在 reward 中。

## 7. PPO loss

完全使用 RSL-RL：

\[
L=L_{\mathrm{clip}}+1.0L_{\mathrm{value}}-5\cdot10^{-4}H.
\]

- PPO ratio clip 0.2；
- clipped value loss；
- GAE：`gamma=0.99`、`lambda=0.95`；
- 每批 5 epochs、4 minibatches；
- 初始 learning rate `3e-4`，desired KL `0.01` 自适应；
- gradient norm clip 1.0。

不存在 BC loss、AMP loss、reference pose loss 或 replay torque imitation loss。

## 8. 环境版本

- Python 3.11
- PyTorch `2.7.1`（本机 `+cu128`；A100 复用已验证的 `+cu126` wheel）
- MJLAB `1.5.3`
- RSL-RL `5.4.0`
- MuJoCo `3.10.0`
- MuJoCo Warp `3.10.0.3`
- Warp `1.15.0`

本机官方 MJLAB 源码参考固定在 commit
`15ebce8840ee2205f4c62d0c20df65dd1794cab4`；项目运行依赖使用同版本发布包。

全部可执行命令见 [COMMANDS.md](COMMANDS.md)。
