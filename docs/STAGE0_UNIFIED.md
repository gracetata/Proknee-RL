# Stage0 Unified 训练（另一台机器拉取后）

本分支在 `zjt/test` 上做了 **Stage0 Unified 专用精简**（删去无关任务与大型资源），详见 [STAGE0_UNIFIED_SCOPE.md](STAGE0_UNIFIED_SCOPE.md)。

## 拉取本分支

```bash
git clone -b zjt/test https://github.com/gracetata/Proknee-RL.git
# 或已有仓库：
# git fetch origin zjt/test && git checkout zjt/test
```

## 依赖（需自行准备）

1. **Isaac Gym**：从 NVIDIA 官方获取并解压，与本仓库 **同级或自定义路径** 均可。
2. **Conda 环境**（默认名 **`proknee_tc`**）：Python 3.8；安装与 Isaac Gym 匹配的 **PyTorch + CUDA** wheel。
3. **安装**（在对应目录执行）：
   - `cd <isaacgym>/python && pip install -e .`
   - `cd <本仓库>/IsaacGymEnvs && pip install -e .`
4. **NumPy**：建议 `numpy<1.24`（与旧版 Isaac Gym 代码兼容）。

## 激活环境

在仓库根目录：

```bash
source scripts/activate_proknee_tc_env.sh
```

（脚本会 `conda activate proknee_tc` 并设置 `LD_LIBRARY_PATH` 等；详见 [ENVIRONMENT_RLLEG.md](ENVIRONMENT_RLLEG.md)。）

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

## 播放检查点（HumanMimic）

推荐脚本（仓库根目录，自动 `source` 环境；不传参则选 `runs/` 下最新 HumanMimic 权重）：

```bash
bash scripts/play_stage0_unified_humanmimic.sh
# 或指定 checkpoint：
bash scripts/play_stage0_unified_humanmimic.sh /绝对路径/.../nn/xxx.pth
```

等价手动命令：

```bash
cd IsaacGymEnvs/isaacgymenvs
python train_humanmimic_unified.py \
  task=HumanoidAMPUnifiedHumanMimic_phase1 \
  train=HumanoidAMPUnifiedHumanMimicPPO \
  test=True headless=False num_envs=4 \
  checkpoint=/绝对路径/xxx.pth
```

Checkpoint 默认写在 `IsaacGymEnvs/isaacgymenvs/runs/` 下（不随 Git 提交）。
