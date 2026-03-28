# Checkpoint Storage

统一存放各阶段最终训练好的模型。所有脚本默认从这里加载模型。

```
checkpoints/
├── stage0/
│   └── stage0_amp_walk_5050.pth    # 28-DOF AMP全身行走策略
├── stage1/
│   └── best.pth                     # DAgger教师策略 (4-DOF假肢)
└── stage2/
    └── best.pth                     # ProprioAdapt学生策略 (可部署)
```

## 使用关系

```
stage0 → Stage 1 (frozen body policy, 24 DOF)
stage0 + stage1 → Stage 2 (frozen body + teacher → train adapt_tconv)
stage0 + stage2 → 评估/可视化 (frozen body + student)
```

## 默认路径 (脚本 argparse 中设定)

| 参数 | 默认值 |
|------|--------|
| `--body-policy` | `outputs/checkpoints/stage0/stage0_amp_walk_5050.pth` |
| `--teacher-ckpt` | `outputs/checkpoints/stage1/best.pth` |
| `--student-ckpt` | `outputs/checkpoints/stage2/best.pth` |
