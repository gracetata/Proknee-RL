# Hora Project Analysis Document

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture - Teacher-Student Two-Stage Training](#2-architecture---teacher-student-two-stage-training)
3. [File Structure](#3-file-structure)
4. [Input/Output Definitions](#4-inputoutput-definitions)
5. [Training Pipeline](#5-training-pipeline)
6. [Key Algorithms](#6-key-algorithms)
7. [Configuration Files](#7-configuration-files)
8. [Code Snippets for Key Modules](#8-code-snippets-for-key-modules)

---

## 1. Project Overview

### What is Hora?

**Hora** (In-Hand Object Rotation via Rapid Motor Adaptation) is a reinforcement learning framework for dexterous in-hand object manipulation. The project was developed by Haozhi Qi, Ashish Kumar, Roberto Calandra, Yi Ma, and Jitendra Malik, and presented at the Conference on Robot Learning (CoRL) 2022.

### Purpose

The primary goal of Hora is to enable a robotic hand (Allegro Hand) to perform **continuous in-hand object rotation** starting from a stable initial grasp. The system achieves this through:

1. **Rapid Motor Adaptation (RMA)**: A two-stage teacher-student learning paradigm
2. **Privileged Information**: Using object physical properties during training that aren't available at deployment
3. **Proprioceptive Adaptation**: Learning to infer hidden object properties from proprioceptive history alone

### Key Features

- **IsaacGym Integration**: Uses NVIDIA IsaacGym for GPU-accelerated parallel simulation
- **Domain Randomization**: Extensive randomization of object mass, friction, scale, center of mass, and PD gains
- **Sim-to-Real Transfer**: Designed for real-world deployment on Allegro Hand hardware
- **Torque Control**: Direct torque control for more realistic motor dynamics

### Citation
```bibtex
@InProceedings{qi2022hand,
  author={Qi, Haozhi and Kumar, Ashish and Calandra, Roberto and Ma, Yi and Malik, Jitendra},
  title={{In-Hand Object Rotation via Rapid Motor Adaptation}},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```

---

## 2. Architecture - Teacher-Student Two-Stage Training

### Overview

Hora employs a **Teacher-Student** training paradigm with two distinct stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HORA TRAINING ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        STAGE 1: TEACHER POLICY                      │   │
│  │                                                                     │   │
│  │   Observations (96D) ──┬──► Actor MLP ──► Actions (16D)            │   │
│  │                        │                                            │   │
│  │   Privileged Info (9D) ─► Priv MLP ──► Latent (8D) ─┘              │   │
│  │                                                                     │   │
│  │   Training: PPO with privileged object information                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        STAGE 2: STUDENT POLICY                      │   │
│  │                                                                     │   │
│  │   Observations (96D) ──┬──► Actor MLP ──► Actions (16D)            │   │
│  │                        │                                            │   │
│  │   Proprio History ─────► ProprioAdaptTConv ──► Latent (8D) ─┘      │   │
│  │   (30 × 32D)                                                        │   │
│  │                                                                     │   │
│  │   Training: Supervised learning (distillation from teacher)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Teacher Policy (PPO with Privileged Information)

The teacher policy has access to **privileged information** about the object's physical properties that would not be available on a real robot:

- Object 3D position (3D)
- Object scale (1D)
- Object mass (1D)
- Object friction coefficient (1D)
- Object center of mass offset (3D)

This privileged information is encoded through an MLP into an 8-dimensional latent embedding, which is concatenated with the proprioceptive observations to inform the policy.

### Stage 2: Student Policy (Proprioceptive Adaptation)

The student policy **does not have access to privileged information**. Instead, it learns to infer the latent embedding from:

- **Proprioceptive history**: A sliding window of past joint positions and target positions

The student uses a **temporal convolutional network (ProprioAdaptTConv)** to compress the proprioceptive history into the same 8-dimensional latent space as the teacher.

### Distillation Process

During Stage 2 training:
1. The teacher's encoder (Priv MLP) is frozen
2. The student's temporal convolution network is trained
3. **Loss**: MSE between student latent and teacher latent
4. The action policy weights are shared and frozen

---

## 3. File Structure

### Directory Layout

```
hora/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── task/                  # Task-specific configs
│   │   ├── AllegroHandHora.yaml
│   │   ├── AllegroHandGrasp.yaml
│   │   ├── PublicAllegroHandHora.yaml
│   │   └── PublicAllegroHandGrasp.yaml
│   └── train/                 # Training configs
│       ├── AllegroHandHora.yaml
│       ├── AllegroHandGrasp.yaml
│       ├── PublicAllegroHandHora.yaml
│       └── PublicAllegroHandGrasp.yaml
│
├── hora/                      # Main source code
│   ├── __init__.py
│   ├── algo/                  # RL algorithms
│   │   ├── __init__.py
│   │   ├── deploy/           # Deployment utilities
│   │   ├── models/           # Neural network models
│   │   │   ├── models.py     # ActorCritic, MLP, ProprioAdaptTConv
│   │   │   └── running_mean_std.py
│   │   ├── padapt/           # Proprioceptive adaptation
│   │   │   └── padapt.py     # Stage 2 training
│   │   └── ppo/              # PPO algorithm
│   │       ├── experience.py # Experience buffer
│   │       └── ppo.py        # Stage 1 training
│   ├── tasks/                # Environment definitions
│   │   ├── __init__.py
│   │   ├── allegro_hand_hora.py    # Main task
│   │   ├── allegro_hand_grasp.py   # Grasp generation
│   │   └── base/                    # Base classes
│   └── utils/                # Utilities
│       ├── misc.py
│       └── reformat.py
│
├── scripts/                   # Training/evaluation scripts
│   ├── train_s1.sh           # Stage 1 training
│   ├── train_s2.sh           # Stage 2 training
│   ├── vis_s1.sh             # Stage 1 visualization
│   ├── vis_s2.sh             # Stage 2 visualization
│   ├── eval_s1.sh            # Stage 1 evaluation
│   └── eval_s2.sh            # Stage 2 evaluation
│
├── train.py                   # Main training entry point
├── deploy.py                  # Hardware deployment
├── gen_grasp.py              # Grasp pose generation
└── requirements.txt
```

### Key Files and Their Roles

| File | Role |
|------|------|
| `hora/algo/models/models.py` | Neural network architectures (ActorCritic, ProprioAdaptTConv) |
| `hora/algo/ppo/ppo.py` | Stage 1 PPO training loop |
| `hora/algo/padapt/padapt.py` | Stage 2 proprioceptive adaptation training |
| `hora/algo/ppo/experience.py` | Experience buffer for PPO rollouts |
| `hora/tasks/allegro_hand_hora.py` | IsaacGym environment for in-hand rotation |
| `train.py` | Hydra-based training entry point |
| `configs/task/AllegroHandHora.yaml` | Environment configuration |
| `configs/train/AllegroHandHora.yaml` | Training hyperparameters |

---

## 4. Input/Output Definitions

### 4.1 Observations (96D)

The observation space consists of **3 timesteps × 32D proprioceptive data**:

```
┌───────────────────────────────────────────────────────────────────┐
│                    OBSERVATION BUFFER (96D)                       │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Timestep t-2: [joint_pos (16D) | joint_target (16D)] = 32D      │
│  Timestep t-1: [joint_pos (16D) | joint_target (16D)] = 32D      │
│  Timestep t:   [joint_pos (16D) | joint_target (16D)] = 32D      │
│                                                                   │
│  Total: 3 × 32D = 96D                                            │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Per-timestep breakdown (32D):**
- **Joint Positions (16D)**: Normalized Allegro Hand DOF positions
- **Joint Targets (16D)**: Current target positions for PD control

**Source code reference:**
```python
# From allegro_hand_hora.py:compute_observations()
cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # [N, 1, 32]
self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
t_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()  # [N, 96]
```

### 4.2 Privileged Information (9D)

Available only during Stage 1 training:

```
┌───────────────────────────────────────────────────────────────────┐
│                 PRIVILEGED INFO BUFFER (9D)                       │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [0:3]  Object Position (3D)     - x, y, z position              │
│  [3:4]  Object Scale (1D)        - relative scale factor         │
│  [4:5]  Object Mass (1D)         - mass in kg                    │
│  [5:6]  Object Friction (1D)     - friction coefficient          │
│  [6:9]  Object COM Offset (3D)   - center of mass x, y, z        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Configuration:**
```yaml
# From configs/task/AllegroHandHora.yaml
hora:
  propHistoryLen: 30
  privInfoDim: 9
```

### 4.3 Actions (16D)

The action space consists of 16 joint torque commands for the Allegro Hand:

```
┌───────────────────────────────────────────────────────────────────┐
│                      ACTIONS (16D)                                │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Actions represent delta target positions (clipped to [-1, 1])   │
│                                                                   │
│  Index 0-3:   Index finger (4 DOF)                               │
│  Index 4-7:   Middle finger (4 DOF)                              │
│  Index 8-11:  Ring finger (4 DOF)                                │
│  Index 12-15: Thumb (4 DOF)                                      │
│                                                                   │
│  Target update: targets = prev_targets + (1/24) * actions        │
│  Torque: τ = Kp * (target - pos) - Kd * vel                      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Source code reference:**
```python
# From allegro_hand_hora.py:pre_physics_step()
targets = self.prev_targets + 1 / 24 * self.actions
self.cur_targets[:] = tensor_clamp(targets, lower_limits, upper_limits)

# Torque computation in update_low_level_control()
torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
self.torques = torch.clip(torques, -0.5, 0.5)
```

### 4.4 Proprioceptive History (30 × 32D) for Stage 2

The proprioceptive history buffer stores 30 timesteps of observations:

```
┌───────────────────────────────────────────────────────────────────┐
│              PROPRIOCEPTIVE HISTORY (30 × 32D)                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Shape: [batch_size, 30, 32]                                     │
│                                                                   │
│  Time t-29: [joint_pos (16D) | joint_target (16D)]               │
│  Time t-28: [joint_pos (16D) | joint_target (16D)]               │
│  ...                                                              │
│  Time t-1:  [joint_pos (16D) | joint_target (16D)]               │
│  Time t:    [joint_pos (16D) | joint_target (16D)]               │
│                                                                   │
│  Total: 30 × 32 = 960 elements per environment                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Source code reference:**
```python
# From allegro_hand_hora.py:_allocate_task_buffer()
self.prop_hist_len = self.config['env']['hora']['propHistoryLen']  # 30
self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device)

# Updated in compute_observations()
self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
```

---

## 5. Training Pipeline

### 5.1 Stage 1: PPO with Privileged Information

**Command:**
```bash
scripts/train_s1.sh ${GPU_ID} ${SEED} ${OUTPUT_NAME}
```

**Full command expansion:**
```bash
CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
    task=AllegroHandHora \
    headless=True \
    seed=${SEED} \
    task.env.forceScale=2 \
    task.env.randomForceProbScalar=0.25 \
    train.algo=PPO \
    task.env.object.type=cylinder_default \
    train.ppo.priv_info=True \
    train.ppo.proprio_adapt=False \
    train.ppo.output_name=AllegroHandHora/${CACHE}
```

**Training Loop:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1 TRAINING LOOP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for epoch in range(max_agent_steps / batch_size):             │
│      │                                                          │
│      ├─► 1. COLLECT ROLLOUTS (play_steps)                      │
│      │       - Run policy for horizon_length steps             │
│      │       - Store obs, actions, rewards, values             │
│      │       - Compute GAE advantages                          │
│      │                                                          │
│      ├─► 2. PPO UPDATE (train_epoch)                           │
│      │       for mini_epoch in range(mini_epochs_num):         │
│      │           for minibatch in storage:                     │
│      │               - Compute actor loss (clipped surrogate)  │
│      │               - Compute critic loss (clipped value)     │
│      │               - Compute entropy bonus                    │
│      │               - Compute bounds loss                      │
│      │               - Backprop and update                      │
│      │               - Adaptive LR based on KL                  │
│      │                                                          │
│      └─► 3. SAVE CHECKPOINTS                                   │
│              - Save every save_freq epochs                      │
│              - Save best based on mean reward                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Hyperparameters (from configs/train/AllegroHandHora.yaml):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 5e-3 | Initial learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.95 | GAE lambda |
| `horizon_length` | 8 | Rollout length |
| `minibatch_size` | 32768 | PPO minibatch size |
| `mini_epochs` | 5 | PPO epochs per update |
| `e_clip` | 0.2 | PPO clipping parameter |
| `entropy_coef` | 0.0 | Entropy bonus coefficient |
| `critic_coef` | 4 | Value loss coefficient |
| `max_agent_steps` | 1.5B | Maximum training steps |

### 5.2 Stage 2: Proprioceptive Adaptation with Supervised Learning

**Command:**
```bash
scripts/train_s2.sh ${GPU_ID} ${SEED} ${OUTPUT_NAME}
```

**Full command expansion:**
```bash
CUDA_VISIBLE_DEVICES=${GPUS} python train.py \
    task=AllegroHandHora \
    headless=True \
    seed=${SEED} \
    task.env.numEnvs=20000 \
    task.env.forceScale=2 \
    task.env.randomForceProbScalar=0.25 \
    train.algo=ProprioAdapt \
    train.ppo.priv_info=True \
    train.ppo.proprio_adapt=True \
    train.ppo.output_name=AllegroHandHora/${CACHE} \
    checkpoint=outputs/AllegroHandHora/${CACHE}/stage1_nn/best.pth
```

**Training Loop:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2 TRAINING LOOP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD STAGE 1 CHECKPOINT                                    │
│      - Load teacher policy weights                              │
│      - Freeze all weights except adapt_tconv                    │
│                                                                 │
│  2. ONLINE DISTILLATION                                        │
│      while agent_steps <= 1e9:                                 │
│          │                                                      │
│          ├─► Get student latent: e = adapt_tconv(proprio_hist) │
│          │                                                      │
│          ├─► Get teacher latent: e_gt = env_mlp(priv_info)     │
│          │                                                      │
│          ├─► Compute MSE loss: L = ||e - e_gt||²               │
│          │                                                      │
│          ├─► Backprop through adapt_tconv only                 │
│          │                                                      │
│          ├─► Step environment with current policy              │
│          │                                                      │
│          └─► Log statistics and save checkpoints               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key differences from Stage 1:**
- Only `adapt_tconv` parameters are trainable
- No PPO updates - only supervised learning
- Uses online data collection (no experience buffer)
- Simple MSE loss for distillation

---

## 6. Key Algorithms

### 6.1 ProprioAdaptTConv: Temporal Convolutional Encoder

The ProprioAdaptTConv network compresses the proprioceptive history (960D) into an 8D latent:

```
┌───────────────────────────────────────────────────────────────────┐
│                    PROPRIOADAPTTCONV ARCHITECTURE                 │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: proprio_hist [N, 30, 32]                                 │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ CHANNEL TRANSFORM (per-timestep MLP)                        │ │
│  │   Linear(32 → 32) + ReLU                                    │ │
│  │   Linear(32 → 32) + ReLU                                    │ │
│  │   Output: [N, 30, 32]                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                       │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ TEMPORAL AGGREGATION (1D Conv along time)                   │ │
│  │   Permute: [N, 30, 32] → [N, 32, 30]                       │ │
│  │   Conv1d(32, 32, kernel=9, stride=2) + ReLU → [N, 32, 11]  │ │
│  │   Conv1d(32, 32, kernel=5, stride=1) + ReLU → [N, 32, 7]   │ │
│  │   Conv1d(32, 32, kernel=5, stride=1) + ReLU → [N, 32, 3]   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                       │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ LOW-DIM PROJECTION                                          │ │
│  │   Flatten: [N, 32, 3] → [N, 96]                            │ │
│  │   Linear(96 → 8)                                            │ │
│  │   Output: [N, 8]                                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Compression: 960D → 96D → 8D                                    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Source code:**
```python
class ProprioAdaptTConv(nn.Module):
    def __init__(self):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, 8)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 30, 32)
        x = x.permute((0, 2, 1))        # (N, 32, 30)
        x = self.temporal_aggregation(x) # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))  # (N, 8)
        return x
```

### 6.2 Distillation Loss: MSE Between Student and Teacher Latents

The distillation process minimizes the mean squared error between:
- **Student latent**: `e = adapt_tconv(proprio_hist)`
- **Teacher latent**: `e_gt = env_mlp(priv_info)`

```python
# From padapt.py:train()
def train(self):
    obs_dict = self.env.reset()
    while self.agent_steps <= 1e9:
        input_dict = {
            'obs': self.running_mean_std(obs_dict['obs']).detach(),
            'priv_info': obs_dict['priv_info'],
            'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
        }
        mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
        
        # MSE distillation loss
        loss = ((e - e_gt.detach()) ** 2).mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        # Step environment
        mu = torch.clamp(mu.detach(), -1.0, 1.0)
        obs_dict, r, done, info = self.env.step(mu)
```

### 6.3 Actor-Critic Forward Pass

```python
def _actor_critic(self, obs_dict):
    obs = obs_dict['obs']
    extrin, extrin_gt = None, None
    
    if self.priv_info:
        if self.priv_info_stage2:
            # Stage 2: Use temporal convolution on proprio history
            extrin = self.adapt_tconv(obs_dict['proprio_hist'])
            # Also compute teacher latent for distillation
            extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
            extrin_gt = torch.tanh(extrin_gt)
            extrin = torch.tanh(extrin)
            obs = torch.cat([obs, extrin], dim=-1)
        else:
            # Stage 1: Use privileged info MLP
            extrin = self.env_mlp(obs_dict['priv_info'])
            extrin = torch.tanh(extrin)
            obs = torch.cat([obs, extrin], dim=-1)

    x = self.actor_mlp(obs)
    value = self.value(x)
    mu = self.mu(x)
    sigma = self.sigma
    return mu, mu * 0 + sigma, value, extrin, extrin_gt
```

---

## 7. Configuration Files

### 7.1 Main Configuration (`configs/config.yaml`)

```yaml
# Task name - used to pick the class to load
task_name: ${task.name}

# Number of environments (overridable)
num_envs: ''

# Random seed (-1 for random)
seed: 42

# Physics and device config
physics_engine: 'physx'
pipeline: 'gpu'
sim_device: 'cuda:0'
rl_device: 'cuda:0'
graphics_device_id: 0

# PhysX configuration
num_threads: 4
solver_type: 1  # TGS solver
num_subscenes: 4

# Testing/checkpointing
test: False
checkpoint: ''
headless: False

# Default configurations
defaults:
  - _self_
  - task: AllegroHandHora
  - train: ${task}
```

### 7.2 Task Configuration (`configs/task/AllegroHandHora.yaml`)

```yaml
name: AllegroHandHora
physics_engine: ${..physics_engine}

env:
  # Environment settings
  numEnvs: ${resolve_default:16384,${...num_envs}}
  numObservations: 96   # 3 timesteps × 32D
  numActions: 16        # 16 DOF Allegro Hand
  envSpacing: 0.25
  episodeLength: 400

  # Controller settings
  controller:
    torque_control: True
    controlFrequencyInv: 6  # 20Hz control
    pgain: 3
    dgain: 0.1

  # Hora-specific settings
  hora:
    propHistoryLen: 30   # History length for Stage 2
    privInfoDim: 9       # Privileged info dimension

  # Reward function
  reward:
    angvelClipMin: -0.5
    angvelClipMax: 0.5
    rotateRewardScale: 1.0
    objLinvelPenaltyScale: -0.3
    poseDiffPenaltyScale: -0.3
    torquePenaltyScale: -0.1
    workPenaltyScale: -2.0

  # Domain randomization
  randomization:
    randomizeMass: True
    randomizeMassLower: 0.01
    randomizeMassUpper: 0.25
    randomizeCOM: True
    randomizeCOMLower: -0.01
    randomizeCOMUpper: 0.01
    randomizeFriction: True
    randomizeFrictionLower: 0.3
    randomizeFrictionUpper: 3.0
    randomizeScale: True
    scaleListInit: True
    randomizeScaleList: [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86]
    randomizePDGains: True
    jointNoiseScale: 0.02

  # Privileged info flags
  privInfo:
    enableObjPos: True
    enableObjScale: True
    enableObjMass: True
    enableObjCOM: True
    enableObjFriction: True

sim:
  dt: 0.0083333  # 120 Hz simulation
  substeps: 1
  up_axis: 'z'
  gravity: [0.0, 0.0, -9.81]
```

### 7.3 Training Configuration (`configs/train/AllegroHandHora.yaml`)

```yaml
seed: ${..seed}
algo: PPO

network:
  mlp:
    units: [512, 256, 128]    # Actor/Critic MLP
  priv_mlp:
    units: [256, 128, 8]      # Privileged info encoder → 8D latent

load_path: ${..checkpoint}

ppo:
  output_name: 'debug'
  
  # Normalization
  normalize_input: True
  normalize_value: True
  value_bootstrap: True
  normalize_advantage: True
  
  # PPO hyperparameters
  num_actors: ${...task.env.numEnvs}
  gamma: 0.99
  tau: 0.95
  learning_rate: 5e-3
  kl_threshold: 0.02
  
  # Batch settings
  horizon_length: 8
  minibatch_size: 32768
  mini_epochs: 5
  
  # Loss coefficients
  clip_value: True
  critic_coef: 4
  entropy_coef: 0.0
  e_clip: 0.2
  bounds_loss_coef: 0.0001
  
  # Gradient clipping
  truncate_grads: True
  grad_norm: 1.0
  
  # Checkpointing
  save_best_after: 0
  save_frequency: 500
  max_agent_steps: 1500000000
  
  # Hora settings
  priv_info: False         # Set True for Stage 1
  priv_info_dim: 9
  priv_info_embed_dim: 8
  proprio_adapt: False     # Set True for Stage 2
```

---

## 8. Code Snippets for Key Modules

### 8.1 Environment Reset and Observation Computation

```python
# From allegro_hand_hora.py

def compute_observations(self):
    self._refresh_gym()
    
    # Sliding window for observation history
    prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
    
    # Add noise to joint positions
    joint_noise_matrix = (torch.rand(self.allegro_hand_dof_pos.shape) * 2.0 - 1.0) * self.joint_noise_scale
    cur_obs_buf = unscale(
        joint_noise_matrix.to(self.device) + self.allegro_hand_dof_pos, 
        self.allegro_hand_dof_lower_limits, 
        self.allegro_hand_dof_upper_limits
    ).clone().unsqueeze(1)
    
    # Concatenate current position and target
    cur_tar_buf = self.cur_targets[:, None]
    cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # [N, 1, 32]
    
    # Update history buffer
    self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
    
    # Extract last 3 timesteps for observation (96D)
    t_buf = self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)
    self.obs_buf[:, :t_buf.shape[1]] = t_buf
    
    # Update proprioceptive history buffer (30 timesteps)
    self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
    
    # Update privileged info (object position)
    self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())
```

### 8.2 Reward Computation

```python
# From allegro_hand_hora.py

def compute_reward(self, actions):
    self.rot_axis_buf[:, -1] = -1  # Rotate around z-axis
    
    # Pose difference penalty (deviation from initial grasp)
    pose_diff_penalty = ((self.allegro_hand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
    
    # Torque and work penalties
    torque_penalty = (self.torques ** 2).sum(-1)
    work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2
    
    # Angular velocity reward (rotation about target axis)
    angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
    object_angvel = angdiff / (self.control_freq_inv * self.dt)
    vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
    rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
    
    # Linear velocity penalty (keep object stable)
    object_linvel = (self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)
    object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

    # Combine rewards
    self.rew_buf[:] = compute_hand_reward(
        object_linvel_penalty, self.object_linvel_penalty_scale,
        rotate_reward, self.rotate_reward_scale,
        pose_diff_penalty, self.pose_diff_penalty_scale,
        torque_penalty, self.torque_penalty_scale,
        work_penalty, self.work_penalty_scale,
    )
```

### 8.3 PPO Training Step

```python
# From ppo.py

def train_epoch(self):
    # Collect rollouts
    self.set_eval()
    self.play_steps()
    
    # PPO update
    self.set_train()
    a_losses, b_losses, c_losses = [], [], []
    
    for _ in range(self.mini_epochs_num):
        for i in range(len(self.storage)):
            value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                returns, actions, obs, priv_info = self.storage[i]
            
            obs = self.running_mean_std(obs)
            batch_dict = {'prev_actions': actions, 'obs': obs, 'priv_info': priv_info}
            res_dict = self.model(batch_dict)
            
            # Actor loss (clipped surrogate objective)
            ratio = torch.exp(old_action_log_probs - res_dict['prev_neglogp'])
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
            a_loss = torch.max(-surr1, -surr2)
            
            # Critic loss (clipped value loss)
            value_pred_clipped = value_preds + (res_dict['values'] - value_preds).clamp(-self.e_clip, self.e_clip)
            value_losses = (res_dict['values'] - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            c_loss = torch.max(value_losses, value_losses_clipped)
            
            # Bounds loss (action regularization)
            soft_bound = 1.1
            mu_loss_high = torch.clamp_max(res_dict['mus'] - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(-res_dict['mus'] + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
            
            # Combined loss
            loss = a_loss.mean() + 0.5 * c_loss.mean() * self.critic_coef \
                   - res_dict['entropy'].mean() * self.entropy_coef \
                   + b_loss.mean() * self.bounds_loss_coef
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.optimizer.step()
```

### 8.4 Stage 2 Adaptation Training

```python
# From padapt.py

def train(self):
    obs_dict = self.env.reset()
    self.agent_steps += self.batch_size
    
    while self.agent_steps <= 1e9:
        input_dict = {
            'obs': self.running_mean_std(obs_dict['obs']).detach(),
            'priv_info': obs_dict['priv_info'],
            'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
        }
        
        # Forward pass: get both student and teacher latents
        mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
        
        # Distillation loss: MSE between student and teacher latents
        loss = ((e - e_gt.detach()) ** 2).mean()
        
        # Only update adaptation module
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        # Step environment with policy
        mu = torch.clamp(mu.detach(), -1.0, 1.0)
        obs_dict, r, done, info = self.env.step(mu)
        self.agent_steps += self.batch_size
```

### 8.5 Model Initialization and Selective Freezing

```python
# From padapt.py

def __init__(self, env, output_dir, full_config):
    # ... initialization code ...
    
    # Freeze all parameters except adaptation module
    adapt_params = []
    for name, p in self.model.named_parameters():
        if 'adapt_tconv' in name:
            adapt_params.append(p)  # Keep trainable
        else:
            p.requires_grad = False  # Freeze
    
    self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
```

---

## Summary

The Hora project implements a sophisticated two-stage training pipeline for in-hand object rotation:

1. **Stage 1 (Teacher)**: Trains a policy with PPO using privileged object information
2. **Stage 2 (Student)**: Distills the teacher's knowledge into a temporal convolutional network that operates only on proprioceptive history

Key innovations include:
- **Rapid Motor Adaptation**: Quick inference of object properties from proprioceptive feedback
- **Temporal Convolution**: Efficient compression of 960D history into 8D latent
- **Domain Randomization**: Extensive randomization for sim-to-real transfer
- **Online Distillation**: Continuous learning during environment interaction

This architecture enables robust sim-to-real transfer for dexterous manipulation tasks.
