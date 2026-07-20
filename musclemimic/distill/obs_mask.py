"""Observation masks for full-muscle distillation with prosthesis-like sensing."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np

from loco_mujoco.core.utils.env import Box, MDPInfo
from musclemimic.prosthesis.constants import DEFAULT_DISABLED_MUSCLE_NAMES, PROSTHESIS_JOINT_NAMES


MUSCLE_OBS_PREFIXES = (
    "muscle_length_",
    "muscle_velocity_",
    "muscle_force_",
    "muscle_excitation_",
    "muscle_activation_",
)


@dataclass(frozen=True)
class MaskedObsSpec:
    keep_indices: tuple[int, ...]
    removed_indices: tuple[int, ...]
    removed_obs_names: tuple[str, ...]
    masked_muscle_names: tuple[str, ...]
    prosthesis_joint_names: tuple[str, ...]
    raw_obs_dim: int
    masked_obs_dim: int
    action_dim: int

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "MaskedObsSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in payload.items()})


def _obs_indices(obs) -> np.ndarray:
    return np.asarray(obs.obs_ind, dtype=np.int32).reshape(-1)


def build_masked_obs_spec(
    env,
    *,
    masked_muscle_names: tuple[str, ...] = DEFAULT_DISABLED_MUSCLE_NAMES,
    prosthesis_joint_names: tuple[str, ...] = PROSTHESIS_JOINT_NAMES,
) -> MaskedObsSpec:
    """Mask prosthesis-leg muscle observations while keeping full-muscle actions."""
    masked = {name.lower() for name in masked_muscle_names}
    removed: list[int] = []
    removed_names: list[str] = []
    for name, obs in env.obs_container.items():
        obs_name = str(name).lower()
        for prefix in MUSCLE_OBS_PREFIXES:
            if not obs_name.startswith(prefix):
                continue
            muscle_name = obs_name.removeprefix(prefix)
            if muscle_name in masked:
                removed.extend(_obs_indices(obs).tolist())
                removed_names.append(str(name))
            break

    raw_obs_dim = int(env.info.observation_space.shape[0])
    remove_mask = np.zeros(raw_obs_dim, dtype=bool)
    if removed:
        remove_mask[np.asarray(removed, dtype=np.int32)] = True
    keep = np.arange(raw_obs_dim, dtype=np.int32)[~remove_mask]
    removed_sorted = np.arange(raw_obs_dim, dtype=np.int32)[remove_mask]
    return MaskedObsSpec(
        keep_indices=tuple(int(x) for x in keep),
        removed_indices=tuple(int(x) for x in removed_sorted),
        removed_obs_names=tuple(removed_names),
        masked_muscle_names=tuple(masked_muscle_names),
        prosthesis_joint_names=tuple(prosthesis_joint_names),
        raw_obs_dim=raw_obs_dim,
        masked_obs_dim=int(keep.size),
        action_dim=int(env.info.action_space.shape[0]),
    )


def apply_obs_mask(obs: np.ndarray, spec: MaskedObsSpec) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)[np.asarray(spec.keep_indices, dtype=np.int32)]


class MaskedObservationEnvView:
    """Network-shape view: same action space, masked observation space."""

    def __init__(self, env, spec: MaskedObsSpec):
        self.env = env
        self.spec = spec
        low = np.asarray(env.info.observation_space.low, dtype=np.float32)[np.asarray(spec.keep_indices, dtype=np.int32)]
        high = np.asarray(env.info.observation_space.high, dtype=np.float32)[np.asarray(spec.keep_indices, dtype=np.int32)]
        self.info = MDPInfo(
            Box(low, high),
            env.info.action_space,
            env.info.gamma,
            env.info.horizon,
            getattr(env.info, "dt", getattr(env, "dt", 0.01)),
        )
        self.mdp_info = self.info
        self.obs_container = env.obs_container

    def __getattr__(self, name):
        return getattr(self.env, name)
