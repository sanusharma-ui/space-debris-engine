# src/physics/entity.py
import numpy as np
from src.physics.state import State
from src.config.settings import DEFAULT_BALLISTIC_COEFF


class Entity:
    """
    Base class for satellites or debris, including position, velocity, ballistic coefficient,
    and optional covariances for Monte Carlo.
    """
    def __init__(self, position, velocity, ballistic_coeff=DEFAULT_BALLISTIC_COEFF, cov_pos=None, cov_vel=None):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.ballistic_coeff = float(ballistic_coeff)
        self.cov_pos = np.atleast_2d(cov_pos) if cov_pos is not None else np.eye(3) * 100.0**2
        self.cov_vel = np.atleast_2d(cov_vel) if cov_vel is not None else np.eye(3) * 0.1**2
