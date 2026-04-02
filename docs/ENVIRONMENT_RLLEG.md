# `proknee_tc` 虚拟环境（远程 / 另一台电脑复现）

Stage0 Unified、HumanMimic 等脚本通过 **`source scripts/activate_proknee_tc_env.sh`** 激活 conda 环境，**默认环境名为 `proknee_tc`**。

若你仍使用旧环境名 **`rlleg`**，可在 `source` 前执行：`export CONDA_ENV_NAME=rlleg`  
（兼容：也可 `source scripts/activate_rlleg_env.sh`，其行为与下面一致。）

按本文在 **`proknee_tc`** 中装好依赖后，在仓库根目录：

```bash
source scripts/activate_proknee_tc_env.sh
bash scripts/train_stage0_unified_humanmimic_phase1.sh
```

---

## 1. 硬件与系统前提

| 项 | 说明 |
|----|------|
| **GPU** | NVIDIA GPU，驱动需支持将要安装的 **CUDA**（见 [PyTorch 起步](https://pytorch.org/get-started/locally/)）。 |
| **OS** | Linux x86_64（与 Isaac Gym 预览版一致）。 |
| **磁盘** | 建议 **≥30GB** 宽裕。 |

---

## 2. 安装 Miniconda / Conda

从 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 安装，确保可使用 `conda activate`。

---

## 3. 创建 conda 环境 `proknee_tc`（Python 3.8）

Isaac Gym 预览版 Python API 通常要求 **Python &lt; 3.9**：

```bash
conda create -n proknee_tc python=3.8 -y
conda activate proknee_tc
```

---

## 4. 安装 PyTorch（CUDA，与驱动匹配）

```bash
conda activate proknee_tc
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

（若 CUDA 11.8，请改用 [pytorch.org](https://pytorch.org) 上对应命令。）

自检：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## 5. NumPy

```bash
pip install "numpy<1.24"
```

---

## 6. Isaac Gym（NVIDIA，非 pip）

1. 从 NVIDIA 获取 Isaac Gym 预览包并解压。  
2. 在 **`proknee_tc`** 中：

```bash
conda activate proknee_tc
cd /你的路径/isaacgym/python
pip install -e .
```

---

## 7. 安装本仓库 `IsaacGymEnvs`

```bash
conda activate proknee_tc
cd /你的路径/RLleg/IsaacGymEnvs
pip install -e .
```

---

## 8. 训练前务必 source

`activate_proknee_tc_env.sh` 会设置 **`LD_LIBRARY_PATH`**（`$CONDA_PREFIX/lib`）与 **`TORCHDYNAMO_DISABLE`**。不要只 `conda activate` 而跳过该脚本。

---

## 9. TensorBoard（查看训练曲线，可选）

训练时事件会写入 **`IsaacGymEnvs/isaacgymenvs/runs/<实验名>/summaries/`**。查看需安装：

```bash
pip install tensorboard
```

另开终端：

```bash
cd IsaacGymEnvs/isaacgymenvs
tensorboard --logdir=./runs --port=6006
```

（`train_stage0_unified_humanmimic_phase1.sh` 启动时会打印相同命令。）

---

## 10. 快速验证

```bash
source scripts/activate_proknee_tc_env.sh
cd IsaacGymEnvs/isaacgymenvs
python -c "import isaacgym; import isaacgymenvs; print('ok')"
```

---

## 11. 环境一览（摘要）

| 项目 | 建议值 |
|------|--------|
| Conda 环境名 | **`proknee_tc`**（`activate_proknee_tc_env.sh` 默认） |
| Python | **3.8** |
| PyTorch | **CUDA 版**（与驱动匹配） |
| NumPy | **&lt; 1.24** |
| Isaac Gym | 官方预览 + `pip install -e isaacgym/python` |
| IsaacGymEnvs | `pip install -e IsaacGymEnvs` |
| 训练前 | `source scripts/activate_proknee_tc_env.sh` |
