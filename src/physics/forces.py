# src/physics/forces.py
import numpy as np
from src.config.settings import GM, RE, J2, atmospheric_density

class ForceModel:
    """
    Base force model. Now takes full state for velocity-dependent forces.
    """
    def acceleration(self, state):
        raise NotImplementedError

class NewtonianGravity(ForceModel):
    """
    Newtonian central gravity.
    """
    def acceleration(self, state):
        position = state.r
        r = np.linalg.norm(position)
        if r == 0:
            return np.zeros(3)
        return -(GM / r**3) * position

class J2Perturbation(ForceModel):
    """
    J2 perturbation for Earth's oblateness.
    """
    def acceleration(self, state):
        position = state.r
        r = np.linalg.norm(position)
        if r == 0:
            return np.zeros(3)
        mu = GM
        z_r = position[2] / r
        z2_r2 = z_r**2
        factor = -(3/2) * J2 * (mu * (RE**2) / (r**5))
        a_j2 = factor * np.array([
            position[0] * (5 * z2_r2 - 1),
            position[1] * (5 * z2_r2 - 1),
            position[2] * (5 * z2_r2 - 3)
        ])
        return a_j2

class AtmosphericDrag(ForceModel):
    """
    Atmospheric drag model.
    Requires ballistic coefficient B = m / (C_d * A).
    """
    def __init__(self, ballistic_coeff):
        self.B = ballistic_coeff

    def acceleration(self, state):
        position = state.r
        velocity = state.v
        v = np.linalg.norm(velocity)
        if v == 0:
            return np.zeros(3)
        altitude = np.linalg.norm(position) - RE
        rho = atmospheric_density(altitude)
        if rho == 0:
            return np.zeros(3)
        # Acceleration: a = - (1/2) * rho * (1/B) * v^2 * (v / v)
        a_mag = -0.5 * rho * (1.0 / self.B) * v**2
        unit_v = velocity / v
        return a_mag * unit_v

class CompositeForce(ForceModel):
    """
    Combines multiple force models.
    """
    def __init__(self, *models):
        self.models = list(models)

    def acceleration(self, state):
        total_a = np.zeros(3)
        for model in self.models:
            total_a += model.acceleration(state)
        return total_a

    def has_drag(self):
        return any(isinstance(m, AtmosphericDrag) for m in self.models)