import numpy as np

from src.config.settings import (
    GM,
    RE,
    J2,
    J3,
    J4,
    OMEGA_EARTH,
    atmospheric_density,
)
from src.physics.ephemeris import AU, sun_position
from src.physics.third_body import ThirdBodyForce  # re-export for compatibility

P_SOLAR = 4.56e-6  # N/m^2 at 1 AU


class ForceModel:
    """
    Base force model. Acceleration signature accepts optional time t in seconds
    since J2000 for time-dependent perturbations.
    """
    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        raise NotImplementedError


class NewtonianGravity(ForceModel):
    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        r = state.r
        norm = np.linalg.norm(r)
        if norm <= 0.0:
            return np.zeros(3, dtype=float)
        return -(GM / norm**3) * r


class J2Perturbation(ForceModel):
    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        r = state.r
        norm = np.linalg.norm(r)
        if norm <= 0.0:
            return np.zeros(3, dtype=float)
        z_r = r[2] / norm
        z2_r2 = z_r * z_r
        factor = -(3.0 / 2.0) * J2 * (GM * RE**2 / norm**5)
        return factor * np.array(
            [
                r[0] * (5.0 * z2_r2 - 1.0),
                r[1] * (5.0 * z2_r2 - 1.0),
                r[2] * (5.0 * z2_r2 - 3.0),
            ],
            dtype=float,
        )


class J3Perturbation(ForceModel):
    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        r = state.r
        norm = np.linalg.norm(r)
        if norm <= 0.0:
            return np.zeros(3, dtype=float)
        x, y, z = r
        zr = z / norm
        zr2 = zr * zr
        factor = -0.5 * J3 * GM * RE**3 / norm**7
        ax = factor * x * (5.0 * zr * (7.0 * zr2 - 3.0))
        ay = factor * y * (5.0 * zr * (7.0 * zr2 - 3.0))
        az = factor * norm * (3.0 - 30.0 * zr2 + 35.0 * zr2 * zr2)
        return np.array([ax, ay, az], dtype=float)


class J4Perturbation(ForceModel):
    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        r = state.r
        norm = np.linalg.norm(r)
        if norm <= 0.0:
            return np.zeros(3, dtype=float)
        x, y, z = r
        zr = z / norm
        zr2 = zr * zr
        factor = (5.0 / 8.0) * J4 * GM * RE**4 / norm**9
        common = 35.0 * zr2 * zr2 - 30.0 * zr2 + 3.0
        ax = factor * x * common
        ay = factor * y * common
        az = factor * z * (35.0 * zr2 * zr2 - 42.0 * zr2 + 9.0)
        return np.array([ax, ay, az], dtype=float)


class AtmosphericDrag(ForceModel):
    def __init__(self, ballistic_coeff: float):
        self.B = max(float(ballistic_coeff), 1e-9)

    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        position = state.r
        velocity = state.v

        omega_vec = np.array([0.0, 0.0, float(OMEGA_EARTH)], dtype=float)
        v_atm = np.cross(omega_vec, position)
        v_rel_vec = velocity - v_atm
        v_rel = np.linalg.norm(v_rel_vec)
        if v_rel <= 0.0:
            return np.zeros(3, dtype=float)

        altitude = np.linalg.norm(position) - RE
        rho = atmospheric_density(altitude)
        if rho <= 0.0:
            return np.zeros(3, dtype=float)

        return -0.5 * rho * (v_rel / self.B) * v_rel_vec


def _earth_shadow_factor(r_sc: np.ndarray, r_sun: np.ndarray) -> float:
    """
    Cylindrical umbra approximation. Returns 0 in Earth shadow, 1 in sunlight.
    """
    sun_norm = np.linalg.norm(r_sun)
    if sun_norm <= 0.0:
        return 1.0

    sun_unit = r_sun / sun_norm
    along_sun = float(np.dot(r_sc, sun_unit))
    if along_sun >= 0.0:
        return 1.0

    cross_track = np.linalg.norm(r_sc - along_sun * sun_unit)
    return 0.0 if cross_track < RE else 1.0


class SolarRadiationPressure(ForceModel):
    def __init__(self, Cr: float = 1.2, area_mass_ratio: float = 0.02, use_shadow: bool = True):
        self.Cr = float(Cr)
        self.Am = max(float(area_mass_ratio), 0.0)
        self.use_shadow = bool(use_shadow)

    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        r_sc = state.r
        r_sun = sun_position(t)
        vec = r_sc - r_sun
        d = np.linalg.norm(vec)
        if d <= 0.0 or self.Am <= 0.0:
            return np.zeros(3, dtype=float)

        shadow = _earth_shadow_factor(r_sc, r_sun) if self.use_shadow else 1.0
        if shadow <= 0.0:
            return np.zeros(3, dtype=float)

        a_mag = shadow * (P_SOLAR * self.Cr * self.Am) * (AU / d) ** 2
        return a_mag * (vec / d)


class CompositeForce(ForceModel):
    def __init__(self, *models):
        self.models = list(models)
        self.model_names = tuple(type(model).__name__ for model in self.models)

    def acceleration(self, state, t: float = 0.0) -> np.ndarray:
        total_a = np.zeros(3, dtype=float)
        for model in self.models:
            try:
                total_a += model.acceleration(state, t)
            except TypeError:
                total_a += model.acceleration(state)
        return total_a

    def has_drag(self) -> bool:
        return any(isinstance(m, AtmosphericDrag) for m in self.models)
