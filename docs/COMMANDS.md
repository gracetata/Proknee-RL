# 命令合集

## 1. 本机：纯 MuJoCo 回放可视化

以下 GUI 命令只能在本机执行。空格切换到下一条轨迹：

```bash
cd /home/user/Workspace/Proknee-RL-muscle
.venv/bin/python \
  torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --manifest torque_replay_training/data/fullbody_all_v3/validation_manifest.json
```

指定轨迹：

```bash
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --dataset torque_replay_training/data/fullbody_all_v3/KIT_314_walking_medium09_poses.npz
```

同一脚本 headless 检查：

```bash
.venv/bin/python torque_replay_training/scripts/visualize_fullbody_replay_local.py \
  --manifest torque_replay_training/data/fullbody_all_v3/validation_manifest.json \
  --check-only
```

## 2. 本机：MJLAB 环境

```bash
cd /home/user/Workspace/Proknee-RL-muscle
python3.11 -m venv .venv-mjlab
.venv-mjlab/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
.venv-mjlab/bin/pip install \
  -r torque_replay_training/configs/mjlab_requirements.txt
.venv-mjlab/bin/pip install -e torque_replay_training
```

验证版本：

```bash
.venv-mjlab/bin/python -c \
  'import torch,mjlab,mujoco,mujoco_warp,warp; print(torch.__version__,mujoco.__version__)'
```

## 3. 本机：回放审计与 PPO smoke

先设置模型路径：

```bash
export MODEL_ROOT=/home/user/Workspace/musclemimic/musclemimic/.venv/lib/python3.11/site-packages/musclemimic_models/model
```

官方 MJLAB 无辅助回放审计；当前预期失败，用于监测版本漂移：

```bash
.venv-mjlab/bin/python torque_replay_training/scripts/audit_mjlab_replay.py \
  --split torque_replay_training/configs/flat_walk_split.json \
  --data-dir torque_replay_training/data/flat_walk_compact_v1 \
  --model-xml torque_replay_training/data/replay_model/musclemimic_replay.xml \
  --model-root "$MODEL_ROOT" --group validation --num-envs 8 \
  --episode-steps 512 --steps 512 --device cuda:0
```

正式初始 assisted-replay gate：

```bash
.venv-mjlab/bin/python torque_replay_training/scripts/audit_mjlab_replay.py \
  --split torque_replay_training/configs/flat_walk_split.json \
  --data-dir torque_replay_training/data/flat_walk_compact_v1 \
  --model-xml torque_replay_training/data/replay_model/musclemimic_replay.xml \
  --model-root "$MODEL_ROOT" --group validation --num-envs 8 \
  --episode-steps 512 --steps 512 --device cuda:0 \
  --healthy-kp 50 --healthy-kd 5 --healthy-correction-limit 300 \
  --root-position-kp 5000 --root-velocity-kd 1000 --root-force-limit 10000 \
  --root-orientation-kp 1000 --root-angular-velocity-kd 100 \
  --root-torque-limit 1000 --max-fall-rate 0 --max-tracking-error 5
```

官方 RSL-RL PPO smoke：

```bash
.venv-mjlab/bin/python torque_replay_training/scripts/train_mjlab_prosthesis.py \
  --split torque_replay_training/configs/flat_walk_split.json \
  --data-dir torque_replay_training/data/flat_walk_compact_v1 \
  --model-xml torque_replay_training/data/replay_model/musclemimic_replay.xml \
  --model-root "$MODEL_ROOT" --group validation --num-envs 16 \
  --episode-steps 64 --max-iterations 1 --device cuda:0 \
  --log-root torque_replay_training/outputs/mjlab_smoke --run-name smoke
```

## 4. checkpoint 独立能力测试

此命令强制 `replay_beta=0`，稳定辅助也为 0：

```bash
.venv-mjlab/bin/python torque_replay_training/scripts/evaluate_mjlab_checkpoint.py \
  --checkpoint /absolute/path/to/model_50.pt \
  --split torque_replay_training/configs/flat_walk_split.json \
  --data-dir torque_replay_training/data/flat_walk_compact_v1 \
  --model-xml torque_replay_training/data/replay_model/musclemimic_replay.xml \
  --model-root "$MODEL_ROOT" --group validation --num-envs 128 \
  --episodes 1024 --episode-steps 512 --device cuda:0 \
  --output /tmp/proknee_model_50_eval.json
```

判定：

- assisted replay 成功、`beta=0` checkpoint 失败：训练尚未学会独立控制；
- assisted replay 失败：先修环境/replay，不继续归因于 PPO；
- `beta=0` fall rate 随 checkpoint 下降：policy 正在收敛。

## 5. A100 GPU 5

```bash
ssh -p 6029 root@39.105.12.60
cd /workspace/Proknee-RL-muscle
git pull --ff-only origin muscle
bash torque_replay_training/scripts/setup_mjlab_env_a100.sh

.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5
bash torque_replay_training/scripts/run_mjlab_smoke_a100.sh

.venv/bin/python torque_replay_training/scripts/a100_gpu_guard.py --gpus 5
bash torque_replay_training/scripts/start_tensorboard_a100.sh
bash torque_replay_training/scripts/start_mjlab_training_a100.sh
```

状态：

```bash
.venv-mjlab/bin/python torque_replay_training/scripts/mjlab_training_status.py
tmux capture-pane -pt proknee-a100-mjlab -S -100
tail -f torque_replay_training/runtime/a100_mjlab.launch.log
```

每次运行 smoke、正式训练或 checkpoint GPU 评估之前，都要重新执行 GPU 5 guard。

## 6. 三处代码一致性

```bash
git -C /home/user/Workspace/Proknee-RL-muscle rev-parse HEAD
git -C /home/user/Workspace/Proknee-RL-muscle ls-remote origin refs/heads/muscle
ssh -p 6029 root@39.105.12.60 \
  'git -C /workspace/Proknee-RL-muscle rev-parse HEAD'
```

三者必须相同。NPZ、checkpoint、TensorBoard 和审计 JSON 是制品，不提交 Git。
