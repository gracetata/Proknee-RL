# MuscleMimic 与左侧假肢训练文档

> 更新时间：2026-07-15  
> 当前路线：完整人体轨迹跟踪 → 导出关节广义力矩 → 人体力矩回放 → 左膝/左踝假肢策略训练。

核心方法文档：

1. [MuscleMimic 原理](MUSCLEMIMIC_PRINCIPLE.md)
   - MuscleMimic 为什么本质上是 motion tracker；
   - 策略怎样控制肌肉；
   - MuJoCo 怎样把肌肉力转换成关节广义力矩；
   - 为什么应导出 `qfrc_actuator` 作为力矩回放数据。

2. [左膝—左踝假肢训练计划](PROSTHESIS_TRAINING_PLAN.md)
   - 选择完整人体能稳定跟踪的直行、左转、右转和曲线行走轨迹；
   - 导出完整人体状态和逐物理步关节广义力矩；
   - 关闭肌肉执行器，在非假肢 DOF 回放参考力矩；
   - 将左膝 1 DOF 和左踝—足复合体 3 DOF 替换为假肢 policy；
   - 使用健康人体在这 4 个 DOF 上的原始力矩作为 baseline，policy 从零残差开始训练。

3. [平地行走假肢策略训练 v1](FLAT_WALK_TRAINING.md)
   - 281 条平地轨迹的受试者隔离训练/验证划分；
   - baseline 力矩统计、PPO 配置与验收门槛；
   - 本机检查、A100 GPU 5 训练、状态和 TensorBoard 命令。

可执行代码、环境和命令入口位于 [torque_replay_training/README.md](../torque_replay_training/README.md)。新代码放在独立目录 `torque_replay_training/`，不会调用旧假肢训练、蒸馏或 DAgger 实现。

A100 服务器部署、当前固定 GPU 5 约束及远端训练命令见 [A100 部署文档](A100_DEPLOYMENT.md)。

本机独立 checkout、测试、结果同步和 MuJoCo 可视化命令见
[本机开发与可视化文档](LOCAL_DEVELOPMENT.md)。其中 MuJoCo GUI 指令明确标注为本机专用。

## 当前方法边界

- 假肢侧：左侧。
- 主动控制维度：4。
  - 左膝：`knee_angle_l`。
  - 左踝—足复合体：`ankle_angle_l`、`subtalar_angle_l`、`mtp_angle_l`。
- 当前只研究仿真中的 privileged policy，不包含之前的假肢训练、蒸馏或真实传感 student。
- 接触力不回放，由 MuJoCo 根据当前人—假肢状态重新求解。
- 人体回放的是完整人体成功 rollout 中的主动广义力 `qfrc_actuator`，不是逆动力学总力，也不是原始肌肉控制向量。
