# # src/models/satellite.py
# import numpy as np
# from src.config.settings import GM, EARTH_RADIUS

# class Satellite:
#     """
#     Satellite model using 2D cartesian coordinates (m).
#     update(dt) uses Velocity-Verlet integrator to preserve orbital energy.
#     """
#     def __init__(self, position, velocity, name="Satellite", radius=5.0):
#         self.position = np.array(position, dtype=float)
#         self.velocity = np.array(velocity, dtype=float)
#         self.name = name
#         self.radius = float(radius)

#     def update(self, dt):
#         r = np.linalg.norm(self.position)
#         if r == 0:
#             return

#         accel = - (GM / r**3) * self.position

#         # half-step velocity
#         v_half = self.velocity + 0.5 * accel * dt
#         # position update
#         self.position = self.position + v_half * dt

#         # new acceleration
#         r_new = np.linalg.norm(self.position)
#         if r_new == 0:
#             self.velocity = v_half
#             return
#         accel_new = - (GM / r_new**3) * self.position

#         # full velocity update
#         self.velocity = v_half + 0.5 * accel_new * dt

#     def __repr__(self):
#         return f"{self.name} at pos {self.position}, vel {self.velocity}"

# src/models/satellite.py
import numpy as np
from src.config.settings import DEFAULT_POS_STD

class Satellite:
    """
    Simple satellite model for Engine-1 & Engine-2 compatibility.
    """
    def __init__(self, position, velocity, cov_pos=None, cov_vel=None):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        # --- covariance (needed for screening / Chan probability) ---
        self.cov_pos = (
            np.atleast_2d(cov_pos)
            if cov_pos is not None
            else np.eye(3) * DEFAULT_POS_STD**2
        )
        self.cov_vel = (
            np.atleast_2d(cov_vel)
            if cov_vel is not None
            else np.eye(3) * 0.1**2
        )
