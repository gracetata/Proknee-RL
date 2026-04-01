# Stage0 Unified 训练（另一台机器拉取后）

## 拉取本分支

```bash
git clone -b zjt/test https://github.com/gracetata/Proknee-RL.git
# 或已有仓库：
# git fetch origin zjt/test && git checkout zjt/test
```

## 依赖（需自行准备）

1. **Isaac Gym**：从 NVIDIA 官方获取并解压，与本仓库 **同级或自定义路径** 均可。
2. **Conda 环境**（示例名 `rlleg`）：Python 3.8；安装与 Isaac Gym 匹配的 **PyTorch + CUDA** wheel。
3. **安装**（在对应目录执行）：
   - `cd <isaacgym>/python && pip install -e .`
   - `cd <本仓库>/IsaacGymEnvs && pip install -e .`
4. **NumPy**：建议 `numpy<1.24`（与旧版 Isaac Gym 代码兼容）。

## 激活环境

在仓库根目录：

```bash
source scripts/activate_rlleg_env.sh
```

（脚本会 `conda activate rlleg` 并设置 `LD_LIBRARY_PATH`。）

## HumanMimic Stage0（默认）

```bash
bash scripts/train_stage0_unified_humanmimic_phase1.sh
```

可选：`bash scripts/train_stage0_unified_humanmimic_phase1.sh <max_iterations> [checkpoint.pth]`

显存不足：`export STAGE0_NUM_ENVS=2048` 或 `1024` 后再运行。

## 基础 Unified（HumanoidAMPUnified）

```bash
bash scripts/train_stage0_unified.sh
```

## Curriculum Unified

```bash
bash scripts/train_stage0_unified_curriculum_phase1.sh
# phase2 / phase3 同理
```

## 播放检查点（HumanMimic）

```bash
cd IsaacGymEnvs/isaacgymenvs
python train_humanmimic_unified.py \
  task=HumanoidAMPUnifiedHumanMimic_phase1 \
  train=HumanoidAMPUnifiedHumanMimicPPO \
  test=True headless=False num_envs=4 \
  checkpoint=/绝对路径/xxx.pth
```

Checkpoint 默认写在 `IsaacGymEnvs/isaacgymenvs/runs/` 下（不随 Git 提交）。
