# ProKnee-Simulator Project Analysis

## 1. Project Overview

ProKnee-Simulator is a prosthetic knee simulation system built on **NVIDIA Isaac Gym** for physics-based humanoid simulation. The project implements **RMA (Rapid Motor Adaptation)** two-stage training for training a humanoid with a prosthetic leg to walk and perform locomotion tasks.

### Key Features
- **Isaac Gym Integration**: GPU-accelerated physics simulation for parallel RL training
- **AMP (Adversarial Motion Priors)**: Human motion imitation using discriminator-based rewards
- **RMA Teacher-Student Architecture**: Two-stage training pipeline for robust prosthetic control
- **Prosthetic Leg Simulation**: Left leg modeled as prosthesis (knee + ankle)

### Technical Stack
- **Simulation**: Isaac Gym Preview 4
- **RL Framework**: rl_games (PPO continuous)
- **Configuration**: Hydra + YAML
- **Logging**: WandB integration

---

## 2. Architecture: RMA Two-Stage Training

The project follows the HORA-style RMA paradigm for humanoid locomotion:

### Stage 1: Teacher Policy (Privileged Information)
```
priv_obs (64D) → Teacher MLP → z_t (8D latent)
policy_input = concat(obs, z_t) → Actor/Critic → actions
```
- Policy learns using **privileged information** (full GRF, body state)
- No distillation loss; pure PPO training
- Teacher network parameters are trainable

### Stage 2: Student Distillation
```
proprio_hist (50×16D) → TConv Adaptor → z_s (8D latent)
policy_input = concat(obs, z_s) → Actor/Critic → actions
Loss = L_PPO + λ × MSE(z_s, stopgrad(z_t))
```
- Student learns to predict latent from **proprioceptive history only**
- Teacher weights **frozen**; provides supervision target
- Distillation coefficient (λ) typically 1.0

### Data Flow Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                     HumanoidAMPRMA Task                         │
├─────────────────────────────────────────────────────────────────┤
│  step() / reset() returns dict:                                 │
│    - obs: Tensor[B, 105]         (base AMP observation)         │
│    - priv_obs: Tensor[B, 64]     (teacher privileged info)      │
│    - proprio_hist: Tensor[B,50,16] (student history window)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ComplexObsRLGPUEnv                            │
│  Flattens proprio_hist to (B, 800) for rl_games buffer          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ModelRMAContinuous                            │
│  Reshapes proprio_hist back to (B, 50, 16) before network       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RMABuilder.Network                           │
│  Stage1: z_t = tanh(MLP(priv_obs)); policy_in = [obs, z_t]     │
│  Stage2: z_s = tanh(TConv(history)); policy_in = [obs, z_s]    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

### Root Directory
```
ProKnee-Simulator/
├── train.py                 # Main training entry point
├── requirements.txt         # Python dependencies
├── RMA_TRAINING.md          # RMA training documentation
├── README.md                # Project README
├── cfg/                     # Hydra configuration files
├── isaacgymenvs/            # Core simulation & learning code
├── assets/                  # Robot models and motion data
├── scripts/                 # Utility scripts
└── runs/                    # Training checkpoints & logs
```

### Configuration (`cfg/`)
```
cfg/
├── config.yaml              # Global config (device, seed, wandb)
├── task/
│   ├── HumanoidAMP.yaml           # Base AMP task
│   ├── HumanoidAMPRMAFlat.yaml    # RMA task (flat terrain)
│   ├── HumanoidAMPTerrain.yaml    # Terrain task
│   └── Humanoid.yaml              # Basic humanoid
├── train/
│   ├── HumanoidAMPRMA_stage1.yaml # Stage1 teacher training
│   ├── HumanoidAMPRMA_stage2.yaml # Stage2 distillation
│   ├── HumanoidAMPPPO.yaml        # Standard AMP PPO
│   └── HumanoidAMPTerrainPPO.yaml # Terrain PPO
└── pbt/                     # Population-based training configs
```

### Isaac Gym Environments (`isaacgymenvs/`)
```
isaacgymenvs/
├── __init__.py
├── tasks/
│   ├── __init__.py               # Task registry
│   ├── humanoid.py               # Basic humanoid task
│   ├── humanoid_amp.py           # AMP humanoid wrapper
│   ├── humanoid_amp_terrain.py   # Terrain variant
│   ├── base/vec_task.py          # VecTask base class
│   └── amp/
│       ├── humanoid_amp_base.py      # Core AMP implementation
│       ├── humanoid_amp_rma.py       # RMA dict-obs wrapper
│       ├── humanoid_amp_terrain_base.py
│       ├── poselib/                  # Motion retargeting tools
│       └── utils_amp/                # AMP utilities
├── learning/
│   ├── amp_continuous.py         # AMP PPO agent
│   ├── amp_models.py             # AMP model wrapper
│   ├── amp_network_builder.py    # AMP network builder
│   ├── rma_continuous.py         # RMA PPO agent (with distill loss)
│   ├── common_agent.py           # Base agent class
│   └── amp_datasets.py           # Experience buffer
├── rma/
│   ├── rma_network_builder.py    # RMA network (teacher/student)
│   ├── rma_models.py             # rl_games model wrapper
│   ├── rma_modules.py            # MLP, TConv modules
│   └── rma_agent_mixin.py        # Latent extraction helpers
├── humanipro/                    # HumaniPro (multi-agent) extensions
└── utils/
    ├── rlgames_utils.py          # ComplexObsRLGPUEnv wrapper
    ├── torch_jit_utils.py        # Quaternion utilities
    └── wandb_utils.py            # Logging
```

### Assets (`assets/`)
```
assets/
├── mjcf/
│   └── amp_humanoid.xml     # Main humanoid MJCF model
└── amp/
    └── motions/             # Motion capture data (.npy)
```

---

## 4. Humanoid Model

### 4.1 `amp_humanoid.xml` Joint Definitions

The humanoid model is defined in MuJoCo XML format with **28 actuated DOFs**:

| Body Part | Joint Names | DOF Count | Range (degrees) |
|-----------|-------------|-----------|-----------------|
| **Abdomen** | abdomen_x, abdomen_y, abdomen_z | 3 | ±60, -60/+90, ±50 |
| **Neck** | neck_x, neck_y, neck_z | 3 | ±50, -40/+60, ±45 |
| **Right Arm** | right_shoulder_x/y/z, right_elbow | 4 | Various |
| **Left Arm** | left_shoulder_x/y/z, left_elbow | 4 | Various |
| **Right Leg** | right_hip_x/y/z, **right_knee**, right_ankle_x/y/z | 7 | Various |
| **Left Leg** | left_hip_x/y/z, **left_knee**, left_ankle_x/y/z | 7 | Various |

### 4.2 Prosthetic Joints (Left Leg)

In the ProKnee configuration, the **left leg** is designated as the prosthetic limb:

| Joint Name | DOF Index | Type | Range | Motor Gear |
|------------|-----------|------|-------|------------|
| `left_knee` | 24 | Hinge | 0-160° | 100 |
| `left_ankle_x` | 25 | Hinge | ±30° | 50 |
| `left_ankle_y` | 26 | Hinge | ±55° | 50 |
| `left_ankle_z` | 27 | Hinge | ±40° | 50 |

The prosthetic joints are configured in `cfg/task/HumanoidAMPRMAFlat.yaml`:
```yaml
env:
  rma:
    proprio_dof_names: ["left_knee", "left_ankle_x", "left_ankle_y", "left_ankle_z"]
```

### 4.3 DOF Indices Mapping

The actuator order in `amp_humanoid.xml` defines the action space:

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | abdomen_x | 14 | right_hip_x |
| 1 | abdomen_y | 15 | right_hip_z |
| 2 | abdomen_z | 16 | right_hip_y |
| 3 | neck_x | 17 | **right_knee** |
| 4 | neck_y | 18 | right_ankle_x |
| 5 | neck_z | 19 | right_ankle_y |
| 6 | right_shoulder_x | 20 | right_ankle_z |
| 7 | right_shoulder_y | 21 | left_hip_x |
| 8 | right_shoulder_z | 22 | left_hip_z |
| 9 | right_elbow | 23 | left_hip_y |
| 10 | left_shoulder_x | 24 | **left_knee** ★ |
| 11 | left_shoulder_y | 25 | **left_ankle_x** ★ |
| 12 | left_shoulder_z | 26 | **left_ankle_y** ★ |
| 13 | left_elbow | 27 | **left_ankle_z** ★ |

★ = Prosthetic DOFs (student proprio features focus on these)

### 4.4 Rigid Body Structure

```
pelvis (root)
├── torso
│   ├── head
│   ├── right_upper_arm → right_lower_arm → right_hand
│   └── left_upper_arm → left_lower_arm → left_hand
├── right_thigh → right_shin → right_foot (force sensor)
└── left_thigh → left_shin → left_foot (force sensor) ★ Prosthesis
```

---

## 5. Input/Output Definitions

### 5.1 Base Observations (105D)

Defined in `humanoid_amp_base.py`:

```python
NUM_OBS = 13 + 52 + 28 + 12  # = 105

OBS_SLICE = {
    'root_h':              (0, 1),     # 1D - root height
    'root_rot_tan_norm':   (1, 7),     # 6D - root rotation (tan-norm encoding)
    'root_vel_local':      (7, 10),    # 3D - root linear velocity (local frame)
    'root_ang_vel_local':  (10, 13),   # 3D - root angular velocity (local frame)
    'dof_obs':             (13, 65),   # 52D - DOF observations (28 joints × ~2)
    'dof_vel':             (65, 93),   # 28D - DOF velocities
    'key_body_pos_local':  (93, 105),  # 12D - key body positions (4 bodies × 3)
}
```

Key bodies: `right_hand`, `left_hand`, `right_foot`, `left_foot`

### 5.2 Privileged Information for Teacher (64D)

Built in `HumanoidAMPRMA._build_priv_obs()`:

```python
priv_obs = concat([
    base_obs,           # 105D base observations
    grf_3d_right_foot,  # 3D ground reaction force (Fx, Fy, Fz)
    grf_3d_left_foot,   # 3D ground reaction force
])  # Total > 64D, truncated/padded to 64D
```

**Purpose**: Teacher has access to full GRF vectors that real sensors cannot measure.

### 5.3 Proprioceptive History for Student (50×16D → 800D flattened)

Built in `HumanoidAMPRMA._build_proprio_step()`:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `dof_pos[prosthesis]` | 4D | Prosthesis joint positions |
| `dof_torque[prosthesis]` | 4D | Prosthesis joint torques |
| `foot_fz` | 2D | Vertical force (Fz only) per foot |
| Padding | 6D | Zero-padded to `proprio_dim=16` |

**History window**: 50 timesteps × 16 features = **800D** (flattened for buffer storage)

The student can only observe:
- Prosthesis angles and torques
- Vertical foot pressure (not full GRF)

### 5.4 Actions (28D)

Actions map to PD position targets for all 28 actuated DOFs:

```python
NUM_ACTIONS = 28
# Processed as: target_pos = action × scale → PD control
```

Motor efforts (gear ratios from XML):
- Core (abdomen): 125
- Neck: 20
- Shoulders: 70
- Elbows: 60
- Hips: 125
- Knees: 100
- Ankles: 50

---

## 6. Training Pipeline

### 6.1 Stage 1: Teacher with Privileged Info

**Configuration**: `cfg/train/HumanoidAMPRMA_stage1.yaml`

```yaml
params:
  algo:
    name: rma_continuous
  network:
    rma:
      stage: stage1
      freeze_teacher: False
  config:
    rma_distill_coef: 0.0  # No distillation
    max_epochs: 100000
```

**Training Command**:
```bash
python train.py task=HumanoidAMPRMAFlat train=HumanoidAMPRMA_stage1 \
    wandb_activate=True motion=walk
```

**What happens**:
1. Environment returns `{obs, priv_obs, proprio_hist}`
2. Teacher MLP: `priv_obs → z_t`
3. Policy: `concat(obs, z_t) → actions`
4. Standard PPO loss (no distillation)

### 6.2 Stage 2: Student Distillation

**Configuration**: `cfg/train/HumanoidAMPRMA_stage2.yaml`

```yaml
params:
  network:
    rma:
      stage: stage2
      freeze_teacher: True
  config:
    rma_distill_coef: 1.0  # Enable distillation
    load_path: ${...checkpoint}  # Load stage1 weights
```

**Training Command**:
```bash
python train.py task=HumanoidAMPRMAFlat train=HumanoidAMPRMA_stage2 \
    checkpoint=runs/stage1/nn/best.pth wandb_activate=True motion=walk
```

**What happens**:
1. Load stage1 checkpoint (teacher + policy weights)
2. Freeze teacher MLP parameters
3. Student TConv: `proprio_hist → z_s`
4. Policy: `concat(obs, z_s) → actions`
5. Loss = PPO Loss + λ × MSE(z_s, stopgrad(z_t))

### 6.3 Loss Function (Stage 2)

```python
# In rma_continuous.py RMAAgent.calc_gradients()
loss = (
    actor_loss
    + critic_coef * critic_loss
    - entropy_coef * entropy
    + bounds_loss_coef * bounds_loss
    + rma_distill_coef * MSE(student_latent, teacher_latent.detach())
)
```

---

## 7. Key Configuration Files

### 7.1 Task Config: `HumanoidAMPRMAFlat.yaml`

```yaml
name: HumanoidAMPRMA

env:
  numEnvs: 4096
  episodeLength: ${...episode_length}
  
  # Contact/termination
  contactBodies: ["right_foot", "left_foot"]
  terminationHeight: 0.5
  groundContactHeight: 0.12
  
  # RMA-specific
  rma:
    enabled: True
    proprio_hist_len: 50        # History window length
    proprio_dim: 10             # Per-step feature dimension
    priv_obs_dim: 64            # Teacher privileged dimension
    proprio_dof_names:          # Prosthesis DOFs to track
      - "left_knee"
      - "left_ankle_x"
      - "left_ankle_y"
      - "left_ankle_z"

  asset:
    assetFileName: "mjcf/amp_humanoid.xml"
```

### 7.2 Train Config: Stage 1

```yaml
params:
  algo:
    name: rma_continuous       # RMAAgent
  model:
    name: continuous_rma       # ModelRMAContinuous
  network:
    name: rma                  # RMABuilder
    
    mlp:
      units: [512, 256]
      activation: elu
    
    rma:
      enabled: True
      stage: stage1
      freeze_teacher: False
      latent_dim: 8
      teacher_mlp_units: [128, 128]
      student_tconv_channels: 32
  
  config:
    rma_distill_coef: 0.0      # No distillation in stage1
    learning_rate: 3e-4
    horizon_length: 16
    minibatch_size: 32
```

### 7.3 Train Config: Stage 2

Key differences from stage1:
```yaml
rma:
  stage: stage2
  freeze_teacher: True

config:
  rma_distill_coef: 1.0        # Enable distillation
  load_path: ${...checkpoint}  # Load stage1 checkpoint
```

### 7.4 Global Config: `config.yaml`

```yaml
task_name: ${task.name}
seed: 42
physics_engine: 'physx'
pipeline: 'gpu'
sim_device: 'cuda:0'
rl_device: 'cuda:0'

defaults:
  - task: Ant                  # Overridden by CLI: task=HumanoidAMPRMAFlat
  - train: ${task}PPO          # Overridden by CLI: train=HumanoidAMPRMA_stage1
```

---

## 8. RMA Modules

### 8.1 `rma_network_builder.py` - RMABuilder

The core network builder extending rl_games' A2CBuilder:

```python
class RMABuilder(network_builder.A2CBuilder):
    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # Build teacher MLP: priv_obs → latent
            self._teacher_mlp = MLP(priv_obs_dim, [128, 128], 'elu')
            self._teacher_out = nn.Linear(mlp_out, latent_dim)  # → z_t
            
            # Build student TConv: proprio_hist → latent
            self._student_tconv = ProprioAdaptTConv(proprio_dim, latent_dim)
            
            # Expand actor/critic first layer to accept (obs + latent)
            _expand_first_linear(self.actor_mlp)
            _expand_first_linear(self.critic_mlp)
        
        def _compute_rma_latents(self, input_dict):
            # Teacher: priv_obs → z_t
            teacher_latent = tanh(self._teacher_out(self._teacher_mlp(priv_obs)))
            
            # Stage1: only teacher
            if self._stage == 'stage1':
                return None, teacher_latent
            
            # Stage2: student + teacher
            student_latent = tanh(self._student_tconv(proprio_hist))
            return student_latent, teacher_latent
        
        def forward(self, input_dict):
            student_z, teacher_z = self._compute_rma_latents(input_dict)
            
            # Choose latent based on stage
            if self._stage == 'stage1':
                obs = concat([obs, teacher_z], dim=-1)
            else:
                obs = concat([obs, student_z], dim=-1)
            
            mu, sigma, value, states = super().forward({**input_dict, 'obs': obs})
            
            # Pack latents into states for distillation loss
            states = {
                'rma_student_latent': student_z,
                'rma_teacher_latent': teacher_z,
                ...
            }
            return mu, sigma, value, states
```

### 8.2 `rma_modules.py` - Neural Network Modules

**MLP (Teacher Encoder)**:
```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, activation='elu'):
        layers = []
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
        self.net = nn.Sequential(*layers)
```

**ProprioAdaptTConv (Student Temporal Conv)**:
```python
class ProprioAdaptTConv(nn.Module):
    """
    Input: (B, T=50, proprio_dim=16)
    Output: (B, latent_dim=8)
    """
    def __init__(self, proprio_dim, latent_dim, channels=32):
        # Per-timestep feature transform
        self.channel_transform = nn.Sequential(
            nn.Linear(proprio_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        
        # Temporal aggregation (1D conv over time)
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=9, stride=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, stride=1),
            nn.ReLU(),
        )
        
        # Pool to fixed length, then project to latent
        self.temporal_pool = nn.AdaptiveAvgPool1d(3)
        self.low_dim_proj = nn.Linear(channels * 3, latent_dim)
    
    def forward(self, x):
        x = self.channel_transform(x)      # (B, T, C)
        x = x.permute(0, 2, 1)             # (B, C, T)
        x = self.temporal_aggregation(x)   # (B, C, T')
        x = self.temporal_pool(x)          # (B, C, 3)
        x = self.low_dim_proj(x.flatten(1))# (B, latent_dim)
        return x
```

### 8.3 `rma_models.py` - ModelRMAContinuous

Wraps the network for rl_games compatibility:

```python
class ModelRMAContinuous(ModelA2CContinuousLogStd):
    class Network(ModelA2CContinuousLogStd.Network):
        def norm_obs(self, observation):
            # Handle dict observations: only normalize 'obs' key
            if isinstance(observation, dict):
                out = dict(observation)
                out['obs'] = self.running_mean_std(observation['obs'])
                return out
            return self.running_mean_std(observation)
        
        def forward(self, input_dict):
            # Unpack nested dict obs if needed
            # Reshape flattened proprio_hist back to (B, T, D)
            # Call underlying network
            # Return result dict with latents in 'rnn_states'
```

### 8.4 `rma_agent_mixin.py` - Latent Utilities

```python
def extract_rma_latents(rnn_states):
    """Extract student/teacher latents from network output."""
    if isinstance(rnn_states, dict):
        return rnn_states.get('rma_student_latent'), rnn_states.get('rma_teacher_latent')
    return None, None

def rma_distill_loss(student_latent, teacher_latent):
    """MSE distillation loss with stop-gradient on teacher."""
    return torch.mean((student_latent - teacher_latent.detach()) ** 2)
```

### 8.5 `rma_continuous.py` - RMAAgent

PPO agent with distillation loss integration:

```python
class RMAAgent(CommonAgent):
    def _load_config_params(self, config):
        self.rma_distill_coef = float(config.get('rma_distill_coef', 0.0))
    
    def calc_gradients(self, input_dict):
        # Standard PPO forward pass
        res_dict = self.model(batch_dict)
        
        # Standard PPO losses
        actor_loss = ...
        critic_loss = ...
        entropy = ...
        bounds_loss = ...
        
        # RMA distillation loss (only if coef > 0)
        distill_loss = torch.zeros(())
        if self.rma_distill_coef > 0.0:
            student_z, teacher_z = extract_rma_latents(res_dict['rnn_states'])
            if student_z is not None and teacher_z is not None:
                distill_loss = rma_distill_loss(student_z, teacher_z)
        
        # Total loss
        loss = (
            actor_loss
            + critic_coef * critic_loss
            - entropy_coef * entropy
            + bounds_loss_coef * bounds_loss
            + rma_distill_coef * distill_loss  # ← RMA term
        )
        
        # Backprop and optimize
        self.scaler.scale(loss).backward()
        ...
```

---

## 9. Quick Reference

### Training Commands

```bash
# Stage 1: Teacher
python train.py task=HumanoidAMPRMAFlat train=HumanoidAMPRMA_stage1 \
    wandb_activate=True motion=walk max_iterations=100000

# Stage 2: Student (load stage1 checkpoint)
python train.py task=HumanoidAMPRMAFlat train=HumanoidAMPRMA_stage2 \
    checkpoint=runs/stage1/nn/best.pth wandb_activate=True motion=walk

# Test/Visualize
python train.py task=HumanoidAMPRMAFlat train=HumanoidAMPRMA_stage2 \
    checkpoint=runs/stage2/nn/best.pth test=True num_envs=1
```

### Dry-Run Validation (No Isaac Gym)

```bash
python scripts/rma_dry_run.py
```

### Debug DOF Mapping

```bash
PRINT_DOF_MAP=1 python train.py task=HumanoidAMPRMAFlat num_envs=1 max_iterations=1
```

---

## 10. Summary Tables

### Observation Dimensions

| Component | Dimension | Purpose |
|-----------|-----------|---------|
| Base obs | 105D | Policy input (both stages) |
| Privileged info | 64D | Teacher encoder input |
| Proprio history | 50×16D | Student encoder input |
| Latent | 8D | Compressed environment estimate |
| Actions | 28D | Joint position targets |

### Network Architecture

| Module | Input | Output | Parameters |
|--------|-------|--------|------------|
| Teacher MLP | 64D | 8D | [128, 128] → Linear |
| Student TConv | (50, 16) | 8D | Linear + Conv1d ×3 + Pool |
| Actor MLP | 113D | 28D | [512, 256] |
| Critic MLP | 113D | 1D | [512, 256] |

### Training Hyperparameters (Default)

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Learning rate | 3e-4 | 3e-4 |
| Horizon length | 16 | 16 |
| Minibatch size | 32 | 32 |
| Mini epochs | 4 | 4 |
| Distill coef | 0.0 | 1.0 |
| Freeze teacher | False | True |
