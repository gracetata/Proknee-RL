"""Configuration factories for the official MJLAB and RSL-RL stack."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .action import ReplayProsthesisActionCfg
from .model import load_replay_spec


def make_env_cfg(
    *,
    replay_paths: tuple[str, ...],
    model_xml: str,
    model_root: str | None = None,
    num_envs: int = 4096,
    episode_steps: int = 512,
    seed: int = 0,
    fixed_replay_beta: float | None = None,
    healthy_kp: float = 50.0,
    healthy_kd: float = 5.0,
    healthy_correction_limit: float = 300.0,
    root_position_kp: float = 5000.0,
    root_velocity_kd: float = 1000.0,
    root_force_limit: float = 10000.0,
    root_orientation_kp: float = 1000.0,
    root_angular_velocity_kd: float = 100.0,
    root_torque_limit: float = 1000.0,
) -> ManagerBasedRlEnvCfg:
    """Create the task without imitation/BC rewards."""

    replay_paths = tuple(str(Path(path).resolve()) for path in replay_paths)
    model_xml = str(Path(model_xml).resolve())
    action = ReplayProsthesisActionCfg(
        entity_name="human",
        replay_paths=replay_paths,
        episode_steps=episode_steps,
        fixed_replay_beta=fixed_replay_beta,
        healthy_kp=healthy_kp,
        healthy_kd=healthy_kd,
        healthy_correction_limit=healthy_correction_limit,
        root_position_kp=root_position_kp,
        root_velocity_kd=root_velocity_kd,
        root_force_limit=root_force_limit,
        root_orientation_kp=root_orientation_kp,
        root_angular_velocity_kd=root_angular_velocity_kd,
        root_torque_limit=root_torque_limit,
    )
    observations = {
        "actor": ObservationGroupCfg(
            terms={
                "frame": ObservationTermCfg(
                    func=mdp.actor_observation,
                    history_length=4,
                    flatten_history_dim=True,
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={"state": ObservationTermCfg(func=mdp.critic_observation)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    rewards = {
        "alive": RewardTermCfg(func=mdp.alive, weight=1.0),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
            params={"std": 0.20},
        ),
        "minimum_height": RewardTermCfg(
            func=mdp.minimum_height,
            weight=0.5,
            params={"minimum": 0.55, "margin": 0.30},
        ),
        "forward_velocity": RewardTermCfg(
            func=mdp.forward_velocity,
            weight=1.0,
            params={"std": 0.50},
        ),
        "lateral_velocity": RewardTermCfg(func=mdp.lateral_velocity_l2, weight=-0.10),
        "yaw_rate": RewardTermCfg(func=mdp.yaw_rate_l2, weight=-0.05),
        "prosthesis_torque": RewardTermCfg(
            func=mdp.prosthesis_torque_l2,
            weight=-2.0e-6,
        ),
        "prosthesis_acceleration": RewardTermCfg(
            func=mdp.prosthesis_acceleration_l2,
            weight=-1.0e-7,
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.006),
        "joint_limit": RewardTermCfg(func=mdp.prosthesis_joint_limit, weight=-1.0),
    }
    terminations = {
        "fallen": TerminationTermCfg(
            func=mdp.fallen,
            params={"minimum_height": 0.55, "minimum_up_z": 0.35},
        ),
        "replay_exhausted": TerminationTermCfg(
            func=mdp.replay_exhausted,
            time_out=True,
        ),
    }
    metrics = {
        "replay_beta": MetricsTermCfg(func=mdp.replay_beta_metric),
        "max_tracking_error": MetricsTermCfg(func=mdp.tracking_error_metric),
        "root_height": MetricsTermCfg(func=mdp.root_height_metric),
        "root_up_z": MetricsTermCfg(func=mdp.root_up_z_metric),
    }
    return ManagerBasedRlEnvCfg(
        decimation=5,
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=2.0,
            entities={
                "human": EntityCfg(
                    spec_fn=partial(load_replay_spec, model_xml, model_root),
                    init_state=EntityCfg.InitialStateCfg(
                        joint_pos={".*": 0.0},
                        joint_vel={".*": 0.0},
                    ),
                )
            },
        ),
        observations=observations,
        actions={"prosthesis": action},
        events={},
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        seed=seed,
        episode_length_s=episode_steps * 0.01,
        is_finite_horizon=False,
        scale_rewards_by_dt=False,
        sim=SimulationCfg(
            nconmax=128,
            njmax=1024,
            mujoco=MujocoCfg(
                timestep=0.002,
                integrator="euler",
                cone="pyramidal",
                jacobian="sparse",
                solver="newton",
                iterations=100,
                tolerance=1.0e-8,
                ls_iterations=50,
                ls_tolerance=0.01,
                ccd_iterations=35,
                gravity=(0.0, 0.0, -9.81),
            ),
        ),
        viewer=ViewerConfig(body_name="human/pelvis"),
    )


def make_runner_cfg(
    *,
    max_iterations: int = 3000,
    seed: int = 0,
) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.10,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=5.0e-4,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="proknee_mjlab_flat_walk",
        run_name="seed_0",
        logger="tensorboard",
        upload_model=False,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=max_iterations,
        clip_actions=1.0,
    )
