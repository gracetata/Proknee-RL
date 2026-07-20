"""External prosthesis controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ProsthesisController(ABC):
    """Evaluation-only controller for the four prosthesis DOFs."""

    def reset(self, env) -> None:
        del env

    @abstractmethod
    def update(self, obs: dict) -> np.ndarray:
        """Return torque for [knee, ankle, subtalar, mtp]."""
