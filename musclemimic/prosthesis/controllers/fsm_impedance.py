"""Simple gait-state impedance prosthesis controller for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ProsthesisController


@dataclass
class FSMImpedanceConfig:
    stance_kp: tuple[float, float, float, float] = (120.0, 90.0, 25.0, 15.0)
    stance_kd: tuple[float, float, float, float] = (10.0, 8.0, 3.0, 2.0)
    swing_kp: tuple[float, float, float, float] = (60.0, 45.0, 20.0, 10.0)
    swing_kd: tuple[float, float, float, float] = (6.0, 5.0, 2.5, 1.5)
    torque_limits: tuple[float, float, float, float] = (120.0, 120.0, 30.0, 20.0)
    stance_grf_threshold: float = 30.0


class FSMImpedanceProsthesisController(ProsthesisController):
    """Reference-centered stance/swing impedance baseline."""

    def __init__(self, config: FSMImpedanceConfig | None = None):
        self.config = config or FSMImpedanceConfig()
        self.state = "swing"

    def reset(self, env) -> None:
        del env
        self.state = "swing"

    def update(self, obs: dict) -> np.ndarray:
        foot_contact = bool(obs.get("foot_contact", False))
        grf = float(obs.get("grf", 0.0))
        self.state = "stance" if foot_contact or grf > self.config.stance_grf_threshold else "swing"

        if self.state == "stance":
            kp = np.asarray(self.config.stance_kp, dtype=np.float64)
            kd = np.asarray(self.config.stance_kd, dtype=np.float64)
        else:
            kp = np.asarray(self.config.swing_kp, dtype=np.float64)
            kd = np.asarray(self.config.swing_kd, dtype=np.float64)

        q = np.asarray(obs["q"], dtype=np.float64)
        qd = np.asarray(obs["qd"], dtype=np.float64)
        ref_q = np.asarray(obs["ref_q"], dtype=np.float64)
        ref_qd = np.asarray(obs["ref_qd"], dtype=np.float64)
        limits = np.asarray(self.config.torque_limits, dtype=np.float64)
        tau = kp * (ref_q - q) + kd * (ref_qd - qd)
        return np.clip(tau, -limits, limits).astype(np.float32)
