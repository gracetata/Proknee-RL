HumanMimic Unified Stage0 — 可选目录（自行训 Stage0 时 rl_games 写入此处）

Stage1/Stage2 默认加载的 Stage0 路径（仓库根相对）:
  outputs/HumanoidAMPUnifiedHumanMimic_02-18-11-01.pth
远程：把该文件放到仓库根下 outputs/ 即可，无需改脚本。

备选：本目录 body_policy.pth（仅当默认文件不存在时由封装脚本尝试）

训练新 Stage0 时（scripts/train_stage0_unified_humanmimic.sh）:
  rl_games 将写入本目录下:
    HumanoidAMPUnifiedHumanMimic_<日-时-分-秒>/nn/*.pth
  TensorBoard: tensorboard --logdir=<本目录>
