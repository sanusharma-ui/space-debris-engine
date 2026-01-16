# # src/models/debris.py
# import numpy as np
# from src.config.settings import GM

# class Debris:
#     """
#     Debris model (2D). Uses same update integrator as Satellite for consistency.
#     """
#     def __init__(self, position, velocity, name="Debris", radius=0.1):
#         self.position = np.array(position, dtype=float)
#         self.velocity = np.array(velocity, dtype=float)
#         self.name = name
#         self.radius = float(radius)

#     def update(self, dt):
#         r = np.linalg.norm(self.position)
#         if r == 0:
#             return

#         accel = - (GM / r**3) * self.position
#         v_half = self.velocity + 0.5 * accel * dt
#         self.position = self.position + v_half * dt

#         r_new = np.linalg.norm(self.position)
#         if r_new == 0:
#             self.velocity = v_half
#             return
#         accel_new = - (GM / r_new**3) * self.position
#         self.velocity = v_half + 0.5 * accel_new * dt

#     def __repr__(self):
#         return f"{self.name} at pos {self.position}, vel {self.velocity}"
# src/models/debris.py
import numpy as np
from src.config.settings import DEFAULT_POS_STD

class Debris:
    """
    Debris object for screening & high-fidelity engines.
    """
    def __init__(self, position, velocity, name="Debris", cov_pos=None, cov_vel=None):
        self.name = name
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        # --- covariance ---
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
