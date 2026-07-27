# 全部轨迹清点、批量导出与回放

## 1. 轨迹全集

本机 GMR cache 当前共有 1,089 条轨迹：

- 1,080 条官方 KIT 动作；
- 9 条项目内生成的 custom 长序列；
- 共 662,991 个 100 Hz 控制转移，参考时长约 6,629.91 秒，即 110.50 分钟。

完整逐条目录位于：

- `torque_replay_training/configs/motion_catalog.csv`：动作、受试者、类别、帧数、时长和历史评测；
- `torque_replay_training/configs/all_available_motions.txt`：批量执行顺序；
- `torque_replay_training/configs/motion_catalog_summary.json`：汇总统计。

历史 `mm-10m-2` 无终止评测覆盖 1,080 条官方动作，其中 1,079 条走到末尾；
`KIT/3/walk_6m_straight_line06_poses` 历史覆盖率约 67%。这些历史结果只用于分析，
不能替代当前 schema-v3 的直立检查和精确回放验证。

## 2. 具体动作

| 类别 | 数量 | 动作含义 |
|---|---:|---|
| `walking_slow` | 108 | 慢速直行 |
| `walking_medium` | 109 | 中速直行 |
| `walking_fast` | 71 | 快速直行 |
| `walking_run` | 90 | 跑步 |
| `straight_forward` | 64 | 指定步数或指定距离的直线前进 |
| `straight_backward` | 78 | 直线倒走 |
| `turn_left` | 165 | 离散左转动作 |
| `turn_right` | 176 | 离散右转动作 |
| `clockwise_circle` | 77 | 顺时针连续绕圈 |
| `counterclockwise_circle` | 77 | 逆时针连续绕圈 |
| `supported_walking` | 9 | 左、右或双侧支撑条件下行走 |
| `handrail_walking` | 40 | 左/右扶栏及横梁扶栏行走 |
| `special_walking` | 3 | Nordic walking、Egyptian 风格行走 |
| `generic_walking` | 13 | 未标速度的普通行走 |
| `custom` | 9 | 拼接或延长的直行长序列 |

文件名中的第一层数字是 KIT 受试者或记录组，例如
`KIT/314/walking_medium09_poses` 表示 314 组的第 9 条中速行走。

## 3. “可回放”的判定

批处理会尝试全部 1,089 条，但不会把所有缓存直接称为正式训练数据。每条轨迹依次经过：

1. tracker 走完整条 reference；
2. 记录 500 Hz 子步实际使用的 `qfrc_actuator` 和 `qacc`；
3. 检查有限值、`root_height >= 0.55`、`root_up_z >= 0.35`；
4. 通过者保存为普通 `.npz`；
5. 摔倒或异常者保存为 `.rejected.npz`；
6. 对通过者执行全长 `all`、零 residual `split` 和 25%/50%/75% 随机 reset 验证。

因此：

- `manifest.json` 中 `passed=true`：tracker 全程直立；
- `validation_manifest.json` 中 `passed=true`：关节广义力能够精确重放；
- 只有两者都通过的文件才进入假肢训练集；
- `.rejected.npz` 只用于分析失败原因。

schema v3 不在子步前插入额外 `mj_forward`，严格保持官方 `env.step` 的 MuJoCo 物理顺序。
v1、v2 数据不能与 v3 混用。

## 4. 本机重新生成轨迹目录

只有 GMR cache 或历史评测结果发生变化时才需要执行：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
PYTHONPATH=torque_replay_training/src \
  .venv/bin/python torque_replay_training/scripts/build_motion_catalog.py \
  --cache-root /home/user/.musclemimic/caches/AMASS/MyoFullBody/gmr \
  --evaluation-root \
    /home/user/Workspace/musclemimic/musclemimic/outputs/eval/runs/official_mm10m2 \
  --catalog-output torque_replay_training/configs/motion_catalog.csv \
  --motion-list-output torque_replay_training/configs/all_available_motions.txt \
  --summary-output torque_replay_training/configs/motion_catalog_summary.json
```

CSV、TXT 和 summary JSON 都提交 Git，保证本机、GitHub 和 A100 使用相同动作集合及顺序。

## 5. 本机批量导出与验证

本机 RTX 4090 使用单个常驻 exporter、每批 32 条轨迹复用模型和 checkpoint。轨迹在批内
逐条处理，避免多进程重复占用显存、互相覆盖 manifest 或引入不可复现的执行顺序。整个流程
无窗口、可断点续跑，预计生成约十余 GB 数据。

一键在后台启动全部 1,089 条导出和验证：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/start_all_motion_collection_local.sh
```

启动前脚本要求物理 GPU 0 是 RTX 4090 且没有计算进程；桌面图形进程不影响检查。任务运行
在 `proknee-local-all-replays` tmux 会话中，固定设置 `CUDA_VISIBLE_DEVICES=0` 和
`XLA_PYTHON_CLIENT_PREALLOCATE=false`。

即使部分动作被拒绝，manifest 仍会逐条原子保存结果；再次执行会同时跳过已有合格文件和
已有 `.rejected.npz` 审计文件。需要在修改算法或判据后重新计算失败项时，显式给
`collect_rollouts.py` 传入 `--retry-rejected`。

查看进度和日志：

```bash
tmux capture-pane -pt proknee-local-all-replays -S -80
.venv/bin/python torque_replay_training/scripts/all_motion_replay_status.py
tail -f torque_replay_training/runtime/collect_all_available_local.log
```

若任务因关机或异常中断，重新运行同一启动命令即可从已完成轨迹之后继续。前台调试时可直接
运行底层脚本：

```bash
bash torque_replay_training/scripts/collect_all_available_local.sh
```

## 6. A100 批量执行

A100 无法访问 Hugging Face，因此先从本机传输清单中的 GMR cache。该步骤不使用 GPU：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
bash torque_replay_training/scripts/sync_motion_caches_to_a100.sh
```

网络中断时重新执行即可；远端 tar 会覆盖同名 cache，但不会修改 Git 文件。

代码必须先通过 Git 同步到相同 `muscle` 提交。缓存完整后，在 A100 检查并只使用物理
GPU 5：

```bash
ssh -p 6029 root@39.105.12.60
cd /workspace/Proknee-RL-muscle
git branch --show-current
git status --short
.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5
bash torque_replay_training/scripts/start_all_motion_collection_a100.sh
```

启动脚本会再次检查 GPU 5。只要 GPU 5 有其他计算进程、显存或利用率超限，就拒绝启动；
不会杀进程、抢卡或改用 GPU 0–4、6、7。

查看远端进度：

```bash
tmux capture-pane -pt proknee-a100-all-replays -S -80
.venv/bin/python torque_replay_training/scripts/all_motion_replay_status.py
tail -f torque_replay_training/runtime/collect_all_available.log
tail -f torque_replay_training/runtime/validate_all_available.log
```

正式输出：

```text
torque_replay_training/data/fullbody_all_v3/manifest.json
torque_replay_training/data/fullbody_all_v3/validation_manifest.json
torque_replay_training/data/fullbody_all_v3/*.npz
torque_replay_training/data/fullbody_all_v3/*.rejected.npz
```

## 7. 本机可视化

以下 GUI 命令只能在本机执行，A100 不运行 MuJoCo viewer。先从
`validation_manifest.json` 选择 `passed=true` 的文件，然后执行：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset \
    torque_replay_training/data/fullbody_all_v3/KIT_7_WalkInClockwiseCircle01_poses.npz
```

批量无窗口检查：

```bash
mapfile -t datasets < <(
  find torque_replay_training/data/fullbody_all_v3 -maxdepth 1 \
    -type f -name '*.npz' ! -name '*.rejected.npz' | sort
)
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset "${datasets[@]}" --check-only
```

桌面环境不要强制 `MUJOCO_GL=egl`。

## 8. Git 与大文件同步

Git 管理：

- 导出、验证、状态和同步脚本；
- 1,089 条动作的 TXT/CSV/JSON 目录；
- 文档和测试。

Git 不管理：

- GMR cache；
- schema-v3 回放 `.npz`；
- rejected 审计数据；
- 日志、checkpoint 和 TensorBoard 文件。

代码一致性检查：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
git rev-parse HEAD
git ls-remote origin refs/heads/muscle
ssh -p 6029 root@39.105.12.60 \
  'git -C /workspace/Proknee-RL-muscle rev-parse HEAD'
```

三处 Git SHA 必须一致。大文件通过独立传输，使用 manifest、schema version 和 SHA-256
进行制品管理。
