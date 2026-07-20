"""Reference-motion PD prosthesis controller for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ProsthesisController


@dataclass
class ReferencePDConfig:
    kp: tuple[float, float, float, float] = (120.0, 80.0, 35.0, 20.0)
    kd: tuple[float, float, float, float] = (12.0, 8.0, 4.0, 2.0)
    torque_limits: tuple[float, float, float, float] = (120.0, 120.0, 30.0, 20.0)


class ReferencePDProsthesisController(ProsthesisController):
    def __init__(self, config: ReferencePDConfig | None = None):
        self.config = config or ReferencePDConfig()

    def update(self, obs: dict) -> np.ndarray:
        q = np.asarray(obs["q"], dtype=np.float64)
        qd = np.asarray(obs["qd"], dtype=np.float64)
        ref_q = np.asarray(obs["ref_q"], dtype=np.float64)
        ref_qd = np.asarray(obs["ref_qd"], dtype=np.float64)
        kp = np.asarray(self.config.kp, dtype=np.float64)
        kd = np.asarray(self.config.kd, dtype=np.float64)
        limits = np.asarray(self.config.torque_limits, dtype=np.float64)
        tau = kp * (ref_q - q) + kd * (ref_qd - qd)
        return np.clip(tau, -limits, limits).astype(np.float32)
