# Vendored HORA PPO core

This package adapts the PPO training core from
[`HaozhiQi/hora`](https://github.com/HaozhiQi/hora), commit
`410d95824dd28b198f6d6910cb4c09330a6303be`.

Reused design and code:

- PyTorch Actor-Critic with a state-independent Gaussian standard deviation;
- running observation/value normalization;
- GPU experience buffer and GAE;
- clipped PPO actor/value losses, action-bounds loss and entropy term;
- adaptive KL learning-rate scheduler;
- TensorBoard and `.pth` checkpoint structure.

Project-specific changes:

- removed Isaac Gym, AllegroHand, Hydra and stage-2 hand adaptation dependencies;
- added a MuJoCo schema-v3 torque-replay vector-environment interface;
- zero-initialized the policy mean so deterministic action zero exactly preserves
  the recorded prosthesis baseline;
- added fall/tracking metrics, JSONL logs and robust checkpoint metadata.

The original HORA MIT license is included in this directory. HORA itself notes
that its PPO implementation is based on `rl_games`; the original source headers
are retained in adapted files.
