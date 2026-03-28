"""Configuration utilities for ProKnee-Hora."""

import os
import yaml
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        DictConfig object with loaded configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return OmegaConf.create(config)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations with override taking precedence.
    
    Args:
        base_config: Base configuration.
        override_config: Override configuration (takes precedence).
        
    Returns:
        Merged DictConfig object.
    """
    return OmegaConf.merge(base_config, override_config)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        save_path: Path to save the configuration.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_config_dir() -> str:
    """Get the configuration directory."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
