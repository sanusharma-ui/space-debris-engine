# src/models/debris.py
import numpy as np
from src.config.settings import DEFAULT_POS_STD, DEFAULT_BALLISTIC_COEFF


class Debris:
    """
    Debris object for screening & high-fidelity engines.
    """
    def __init__(
        self,
        position,
        velocity,
        name: str = "Debris",
        cov_pos=None,
        cov_vel=None,
        ballistic_coeff: float = DEFAULT_BALLISTIC_COEFF,
    ):
        self.name = str(name)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        # Optional: used by Engine-2 force model + runner/pipeline shims
        self.ballistic_coeff = float(ballistic_coeff)

        # --- covariance ---
        self.cov_pos = (
            np.atleast_2d(cov_pos)
            if cov_pos is not None
            else np.eye(3, dtype=float) * float(DEFAULT_POS_STD) ** 2
        )
        self.cov_vel = (
            np.atleast_2d(cov_vel)
            if cov_vel is not None
            else np.eye(3, dtype=float) * 0.1**2
        )