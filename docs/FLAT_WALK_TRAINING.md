# 平地行走假肢策略训练（HORA PPO）

## 1. 当前正式架构

先给结论：

- 物理环境是 **MuJoCo 3.4 原生 CPU `mj_step`**，不是 MuJoCo Warp；
- 强化学习算法是 **PPO**；
- PPO 网络、经验缓存、GAE、clipped actor/value loss、KL 自适应学习率、
  RunningMeanStd、checkpoint 和 TensorBoard 布局复用
  [HaozhiQi/hora](https://github.com/HaozhiQi/hora) 的 PyTorch 实现；
- HORA 原项目使用 Isaac Gym Preview 4 / PhysX GPU pipeline。本项目不迁移到
  Isaac Gym，因为现有 787 条可验证回放数据和人体模型都建立在 MuJoCo 上；
- MuJoCo 的 16 个并行 actor 在 CPU 线程中执行，观测和动作转为 Torch tensor，
  HORA Actor-Critic 与 PPO 更新在 A100 物理 GPU 5 上执行；
- 不使用原项目旧的 JAX/Flax PPO 作为本阶段正式训练器。

HORA 参考仓库固定为 commit
`410d95824dd28b198f6d6910cb4c09330a6303be`。本机参考副本在
`/home/user/Workspace/hora`，A100 参考副本在 `/workspace/hora`。本项目中经过适配的
代码位于 `torque_replay_training/src/torque_replay_training/hora/`，保留 HORA 的
MIT License 和来源说明。

没有复用的部分包括 AllegroHand 任务、Isaac Gym 环境、PhysX、手部奖励和 HORA
第二阶段 proprio adaptation。当前先训练有 reference 信息的 privileged policy；
传感器可部署策略是后续阶段。

## 2. 本阶段目标

只训练平地向前行走。非假肢人体关节回放录制的健康广义力，左膝 1 DOF 与左踝—足
3 DOF 使用：

\[
\tau_P(t,s)=\tau_P^{healthy}(t,s)
 a_t\odot[20,20,8,5]\ {\rm Nm}.
\]

四维动作分别对应左膝、左踝、左距下和左 MTP。策略学习的是健康 baseline 上的
residual，而不是从零预测整条假肢力矩。

Actor mean 输出层为零初始化，所以训练前确定性动作严格为零，等价于 mask 前的健康
完整人体。探索标准差初值为 `exp(-3)≈0.05`。

## 3. 为什么健康人体不加 PD

每个 100 Hz policy 步包含 5 个 500 Hz MuJoCo 子步。每个子步把数据中的
`qfrc_actuator` 作为 `qfrc_applied` 注入；原肌肉/电机 actuator 的 gain 和 bias 已全部
置零，避免重复施力。

健康人体 DOF 使用：

\[
\tau_H(t,s)=\tau_H^{recorded}(t,s).
\]

本阶段 `healthy_kp=healthy_kd=0`。原因是 schema-v3 数据已经保存了能够逐子步精确
复现状态的广义力、`qpos/qvel` 和 `qacc_warmstart`。额外 PD 会把极小的浮点差异反馈
回系统，长时回放反而偏离录制轨迹。

同理，`exact_baseline=true`：健康假肢 baseline 不做二次裁剪，只有 residual 动作本身
受 `[20,20,8,5] Nm` 限制。这是保证第 0 个策略与健康完整人完全一致的必要条件。

本机在全部 52 条未见受试者验证轨迹上随机执行 100 个 zero-residual episode 的结果：

- 跌倒 `0/100`；
- `healthy_pos_rms/vel_rms = 0`；
- `prosthesis_pos_rms/vel_rms = 0`。

该基线检查是 A100 正式训练的硬门槛。

## 4. 数据与受试者隔离

数据来自 787 条通过 schema-v3 精确验证的全身回放。平地 v1 选择：

| 类别 | 全部 | 训练 | 验证 |
|---|---:|---:|---:|
| `generic_walking` | 6 | 2 | 4 |
| `straight_forward` | 47 | 40 | 7 |
| `walking_slow` | 66 | 50 | 16 |
| `walking_medium` | 95 | 77 | 18 |
| `walking_fast` | 67 | 60 | 7 |
| 合计 | 281 | 229 | 52 |

共 150,076 个 100 Hz 控制步，约 25 分钟。原始 schema-v3 为 3,088,889,783 bytes
（2.88 GiB）；训练使用的无损 compact-v1 为 796,666,901 bytes（约 760 MiB，
原体积的 25.79%）。训练为 229 条、121,805 步；验证为 52 条、28,271 步。

划分按受试者隔离：`subject 9` 和 `subject 425` 只用于验证，其余 15 个受试者用于
训练。固定清单位于
`torque_replay_training/configs/flat_walk_split.json`。

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/build_flat_walk_split.py
.venv/bin/python torque_replay_training/scripts/compact_flat_walk_data.py --workers 8
.venv/bin/python torque_replay_training/scripts/verify_compact_flat_walk_data.py
```

## 5. Compact 回放格式

正式训练不读取原始 tracker 的全部导出字段。每条 compact NPZ 只有 8 个 key：

| key | 形状 | 用途 |
|---|---|---|
| `motion_path` | scalar | 轨迹身份 |
| `dt_control` | scalar | 100 Hz 控制周期 |
| `dt_physics` | scalar | 500 Hz 物理周期 |
| `rollout_qpos` | `[T+1,89]` | 全身位置与随机重置 |
| `rollout_qvel` | `[T+1,88]` | 全身速度与随机重置 |
| `qacc_warmstart` | `[T+1,88]` | MuJoCo 接触求解器精确 warm-start |
| `qfrc_actuator` | `[T,5,88]` | 每个物理子步的全身广义力 |
| `metadata` | JSON scalar | 关节顺序、DOF/qpos 映射和四个假肢索引 |

所有决定物理演化的数组保持 `float64`，没有量化。以下内容不进入训练数据：

- 354 维肌肉 `actuator_ctrl` 和 `actuator_force`；
- tracker 的 `policy_action`；
- `qfrc_passive`、`qfrc_constraint`；
- contact 记录；
- reference、reward、done、absorbing；
- 肌肉名称、tracker checkpoint 和采集诊断 metadata。

被动力、约束力和接触在 MuJoCo 每个 `mj_step` 中由当前状态、模型和环境重新计算。
`qfrc_actuator_mean` 不保存，在加载后由 `[T,5,88]` 的广义力在线求均值。

每个 compact 文件都记录大小和 SHA-256：
`torque_replay_training/configs/flat_walk_compact_manifest.json`。批处理会在写入后重新加载
并校验维度；最终验证还检查 281 个文件的 SHA-256 和 schema。compact 与原始数据的
`qpos/qvel/qfrc/warm-start` 均逐元素相等。

## 6. MuJoCo—HORA 适配

`HoraTorqueReplayVecEnv` 只替换 HORA 的任务接口，不改变物理：

1. 共享一个只读 `MjModel` 和已加载的 NPZ 数据；
2. 每个 actor 有独立 `MjData`、随机数状态、轨迹和起始帧；
3. `ThreadPoolExecutor` 并行执行 CPU `mj_step`；
4. `reset/step` 返回 HORA PPO 所需的 Torch tensor；
5. done 后 actor 自动换轨迹并随机重置，不影响同一批其他 actor。

单 actor 连续 8 步的适配环境与原 `TorqueReplayEnv` 已做逐步对照，相同动作下
`qpos` 和 `qvel` 最大误差均为 0。

观测为 39 维，包括 motion phase、四个假肢关节状态及 reference 误差、健康 baseline
力矩、上一步 residual、pelvis 高度/朝向、root 速度、健康关节误差统计和接触数量。
这是 privileged policy，不能直接等同于实物可部署观测。

## 7. HORA PPO 配置

正式配置：`torque_replay_training/configs/flat_walk_hora.yaml`。

| 参数 | 数值 |
|---|---:|
| 物理 / policy 频率 | 500 Hz / 100 Hz |
| 并行 MuJoCo actor | 16 |
| CPU worker | 16 |
| 总 agent steps | 2,000,000 |
| horizon | 256 |
| 每批样本 | 4,096 |
| minibatch | 1,024 |
| mini epochs | 5 |
| learning rate | 3e-4，按 KL 自适应 |
| gamma / GAE lambda | 0.99 / 0.95 |
| PPO clip | 0.2 |
| Actor-Critic MLP | 256, 256，ELU |
| device | `cuda:0`（映射到物理 GPU 5） |
| seed | 0 |

动作从对角高斯分布采样并在送入环境前 clamp 到 `[-1,1]`；确定性评估使用 mean。
checkpoint 是 HORA 风格的 `.pth`，同时保存策略、观测/价值归一化器、optimizer、配置、
步数和 Git commit。

## 8. 环境

项目原 `.venv` 用于 MuscleMimic/JAX、数据导出和 GPU 占用检查。正式 PPO 使用独立
`.venv-hora`：

- Python 3.11；
- PyTorch `2.7.1+cu126`；
- MuJoCo `3.4.0`；
- NumPy `2.2.6`、PyYAML、tensorboardX。

A100 无法直接访问 PyTorch wheel 源，因此从本机生成离线包：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/prepare_hora_env_bundle_local.sh
scp -P 6029 \
  torque_replay_training/runtime/hora_site_packages_py311_cu126.tar.zst \
  root@39.105.12.60:/workspace/Proknee-RL-muscle/torque_replay_training/runtime/
```

A100 解包：

```bash
cd /workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/setup_hora_env_a100.sh
CUDA_VISIBLE_DEVICES=5 .venv-hora/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 9. 本机测试

```bash
cd /home/user/Workspace/Proknee-RL-muscle
PYTHONPATH=torque_replay_training/src:. \
  .venv/bin/python -m pytest torque_replay_training/tests -q
```

本机 CPU HORA smoke：

```bash
OUT=/tmp/proknee_hora_local_smoke
rm -rf "$OUT"
.venv/bin/python torque_replay_training/scripts/train_hora_policy.py \
  --config torque_replay_training/configs/flat_walk_hora_smoke.yaml \
  --split torque_replay_training/configs/flat_walk_split.json \
  --data-dir torque_replay_training/data/flat_walk_compact_v1 \
  --model torque_replay_training/data/replay_model/musclemimic_replay.mjb \
  --output "$OUT/seed_0" --limit-datasets 2
```

## 10. Git、数据与 A100 同步

本机、GitHub 和 A100 必须位于 `muscle` 分支的同一 commit。约 760 MiB compact NPZ
和 119 MiB MJB 是运行数据，不提交 Git：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
git push origin muscle

ssh -p 6029 root@39.105.12.60 \
  'cd /workspace/Proknee-RL-muscle && git pull --ff-only origin muscle'

bash torque_replay_training/scripts/sync_flat_walk_data_to_a100.sh
```

## 11. A100 卡 5 smoke 与正式训练

只能使用物理 GPU 5。每次启动前都必须检查是否已有他人计算进程；脚本检测到占用会
退出，不杀进程、不抢卡。

```bash
ssh -p 6029 root@39.105.12.60
cd /workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5

bash torque_replay_training/scripts/run_flat_walk_smoke_a100.sh
bash torque_replay_training/scripts/start_tensorboard_a100.sh
bash torque_replay_training/scripts/start_flat_walk_training_a100.sh
```

状态：

```bash
.venv/bin/python torque_replay_training/scripts/flat_walk_training_status.py
tmux capture-pane -pt proknee-a100-flat-walk -S -100
tail -f torque_replay_training/outputs/a100_flat_walk_v1/train.log
```

本机 TensorBoard：

```bash
ssh -N -L 6011:127.0.0.1:6011 -p 6029 root@39.105.12.60
```

浏览器访问 `http://127.0.0.1:6011`。可视化回放仍只在本机使用，A100 只运行
headless MuJoCo。

## 12. 验收与制品

最低条件：

1. 正式训练前 baseline 验证 100 episodes：`fall_rate=0`、四项 RMS 为 0；
2. 训练过程无 NaN/Inf；
3. `.pth` checkpoint 可重新加载；
4. 最终 200 个未见受试者 episode 不增加跌倒率；
5. 假肢误差不高于 baseline 的容差，residual 不长期饱和。

```text
torque_replay_training/outputs/a100_flat_walk_v1/
├── baseline_validation.json
├── baseline_validation.log
├── train.log
├── policy_validation.json
├── policy_validation.log
└── seed_0/
    ├── metrics.jsonl
    ├── stage1_nn/
    │   ├── last.pth
    │   └── ep_*_step_*.pth
    └── stage1_tb/
```

运行标记为
`torque_replay_training/runtime/a100_flat_walk_v1.{running,complete,failed}`。

## 13. 后续阶段

v1 先证明 HORA PPO 与精确 MuJoCo 回放链路。通过后再逐项加入 baseline 降额、状态和
动力学扰动、延迟/低通/slew-rate、真实假肢惯量和硬件限幅、足底 GRF/滑移奖励、
多 seed，以及转弯、跑步和后退。最后再参考 HORA stage 2，把 privileged policy
蒸馏/适配为只使用实物传感器历史的策略。
