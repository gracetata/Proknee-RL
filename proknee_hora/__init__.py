"""
ProKnee-Hora: Teacher-Student Architecture for Prosthetic Knee Control

This package implements a three-stage training pipeline:
- Stage 0: Full body AMP walking policy
- Stage 1: Teacher policy with privileged information (knee control only)
- Stage 2: Student policy with limited sensor input (distillation)

NOTE: Isaac Gym requires being imported before torch.  Sub-modules that
depend on Isaac Gym (envs, algo) are imported lazily to avoid triggering
a premature ``import torch`` when this package is loaded by pytest or
other tools that import torch first.
"""

__version__ = "0.1.0"
__author__ = "ProKnee Team"


def __getattr__(name):
    """Lazy import of sub-modules to avoid Isaac Gym import-order issues."""
    if name == "envs":
        from . import envs
        return envs
    if name == "algo":
        from . import algo
        return algo
    if name == "utils":
        from . import utils
        return utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
