# Copyright (c) 2018-2023, NVIDIA Corporation
# Stage0 Unified 精简分支：仅注册 Humanoid AMP Unified 栈，避免导入整套任务环境（Ant/Factory/…）。
# train_humanmimic_unified.py 会在运行时向 isaacgym_task_map 注入 HumanoidAMPUnifiedHumanMimic。

from .humanoid_amp_unified import HumanoidAMPUnified

isaacgym_task_map = {
    "HumanoidAMPUnified": HumanoidAMPUnified,
}
