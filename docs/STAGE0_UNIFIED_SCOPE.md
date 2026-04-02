# Stage0 Unified 仓库范围说明

本分支**仅保证**以下 **Unified Stage0** 训练路径可用：

| 用途 | 脚本 |
|------|------|
| HumanMimic | `scripts/train_stage0_unified_humanmimic_phase1.sh` → `train_humanmimic_unified.py` |
| 基础 Unified | `scripts/train_stage0_unified.sh` → `train.py` |
| Curriculum | `scripts/train_stage0_unified_curriculum_phase1.sh` 等 → `train_curriculum_unified.py` |

## 已包含（相对完整 IsaacGymEnvs 的删减）

- **脚本**：`scripts/activate_proknee_tc_env.sh`、`stage0_batch_hydra.sh`、上述 `train_*` / `play_stage0_unified*.sh`
- **配置**：`isaacgymenvs/cfg/` 下与 `HumanoidAMPUnified*`、`HumanoidAMPUnifiedHumanMimic*`、`HumanoidAMPUnifiedCurriculum*` 相关的 task/train YAML
- **代码**：`tasks/humanoid_amp*.py`、`tasks/amp/`、`tasks/base/`；`tasks/__init__.py` **仅注册** `HumanoidAMPUnified`
- **学习**：`isaacgymenvs/learning/` 中 AMP / HumanMimic 相关模块
- **入口**：`train.py`、`train_humanmimic_unified.py`、`train_curriculum_unified.py`
- **资源**：`assets/amp/motions/` 中 `multi_walk_run.yaml` 与 `amp_humanoid_walk.npy`、`amp_humanoid_run.npy`；`assets/mjcf/amp_humanoid.xml`

## 已删除（减小体积）

- 非 AMP 人形所需的大型 `assets/` 子目录（如 factory、urdf 等）
- 非 Humanoid AMP Unified 的 `tasks/` 下任务（Ant、Factory、Franka 等）

## 不包含

- **Isaac Gym**：仍需在本地按 NVIDIA 要求单独安装，并 `pip install -e isaacgym/python`

## 依赖安装

```bash
cd IsaacGymEnvs && pip install -e .
```
