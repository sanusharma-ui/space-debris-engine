# # src/engine/engine2.py
# import numpy as np
# from typing import Optional, Dict, Any

# from src.physics.state import State
# from src.physics.entity import Entity
# from src.physics.forces import (
#     NewtonianGravity,
#     J2Perturbation,
#     AtmosphericDrag,
#     CompositeForce,
# )
# from src.physics.solver import RK4Solver
# from src.physics.utils import specific_energy
# from src.config.settings import COLLISION_RADIUS, DANGER_RADIUS

# from src.engine.engine1 import Engine1


# class Engine2:
#     """
#     High-fidelity physics-based collision confirmation engine.

#     Features:
#     - 3D RK4 propagation for satellite and debris with CompositeForce (gravity + J2 + optional drag)
#     - Adaptive timestep near close approach
#     - Energy-drift reporting (only when drag disabled)
#     - Covariance-based Monte Carlo using numpy
#     - Optional Engine-1 escalation to avoid unnecessary high-fidelity runs
#     """

#     def __init__(self, dt: float = 1.0, adaptive_threshold: float = 100000.0, enable_drag: bool = True):
#         """
#         dt: base timestep in seconds
#         adaptive_threshold: distance in meters below which timestep is refined
#         enable_drag: whether to include atmospheric drag in force model
#         """
#         self.dt_base = float(dt)
#         self.adaptive_threshold = float(adaptive_threshold)
#         self.enable_drag = bool(enable_drag)
#         self.gravity = NewtonianGravity()
#         self.j2 = J2Perturbation()
#         self.engine1 = Engine1()

#     def _get_force_model(self, ballistic_coeff: float) -> CompositeForce:
#         models = [self.gravity, self.j2]
#         if self.enable_drag:
#             models.append(AtmosphericDrag(ballistic_coeff))
#         return CompositeForce(*models)

#     def run(self, satellite: Entity, debris: Entity, duration: float, use_engine1_escalation: bool = True) -> Dict[str, Any]:
#         """
#         Run a single high-fidelity propagation for the given duration (seconds).

#         Returns dict:
#           {
#             "closest_time": float (seconds from start),
#             "miss_distance": float (meters),
#             "relative_velocity": float (m/s) at closest approach,
#             "collision": bool,
#             "conjunction": bool (within DANGER_RADIUS),
#             "energy_drift_sat_percent": float | None,
#             "energy_drift_deb_percent": float | None,
#             "note": optional str
#           }
#         """
#         # Engine-1 fast screening optional escalation
#         if use_engine1_escalation:
#             e1 = self.engine1.run(satellite, debris, duration)
#             if not e1.get("escalate", False):
#                 # No escalation necessary
#                 return {
#                     **e1,
#                     "note": "No escalation: Engine-1 indicates low risk",
#                     "energy_drift_sat_percent": None,
#                     "energy_drift_deb_percent": None,
#                 }

#         # Initialize states
#         sat_state = State(satellite.position, satellite.velocity)
#         deb_state = State(debris.position, debris.velocity)

#         # Force models and solvers
#         force_sat = self._get_force_model(satellite.ballistic_coeff)
#         force_deb = self._get_force_model(debris.ballistic_coeff)

#         solver_sat = RK4Solver(force_sat)
#         solver_deb = RK4Solver(force_deb)

#         # Determine if either composite has drag (for energy reporting)
#         has_drag = force_sat.has_drag() or force_deb.has_drag()

#         init_e_sat = specific_energy(sat_state) if not has_drag else None
#         init_e_deb = specific_energy(deb_state) if not has_drag else None

#         t = 0.0
#         min_dist = float("inf")
#         t_min: Optional[float] = 0.0
#         rel_v_at_min: Optional[float] = None

#         # Initial distance check (t = 0)
#         rel_pos = sat_state.r - deb_state.r
#         dist0 = float(np.linalg.norm(rel_pos))
#         if dist0 < min_dist:
#             min_dist = dist0
#             t_min = 0.0
#             rel_v_at_min = float(np.linalg.norm(sat_state.v - deb_state.v))

#         # Propagation loop with adaptive timestep
#         while t < duration:
#             # Choose dt based on proximity
#             rel_pos = sat_state.r - deb_state.r
#             dist = float(np.linalg.norm(rel_pos))
#             if dist < self.adaptive_threshold:
#                 current_dt = max(self.dt_base / 10.0, 1e-3)  # refine - but avoid extremely small dt
#             else:
#                 current_dt = self.dt_base
#             current_dt = min(current_dt, duration - t)

#             # Step both bodies
#             sat_state = solver_sat.step(sat_state, current_dt)
#             deb_state = solver_deb.step(deb_state, current_dt)

#             # Advance time and evaluate
#             t += current_dt
#             rel_pos = sat_state.r - deb_state.r
#             dist = float(np.linalg.norm(rel_pos))

#             if dist < min_dist:
#                 min_dist = dist
#                 t_min = float(t)
#                 rel_v_at_min = float(np.linalg.norm(sat_state.v - deb_state.v))

#         collision = bool(min_dist <= COLLISION_RADIUS)
#         conjunction = bool(min_dist <= DANGER_RADIUS)

#         # Energy drift if conservative (no drag)
#         if not has_drag:
#             final_e_sat = specific_energy(sat_state)
#             final_e_deb = specific_energy(deb_state)
#             drift_sat = (abs(final_e_sat - init_e_sat) / abs(init_e_sat) * 100.0) if init_e_sat not in (0, None) else 0.0
#             drift_deb = (abs(final_e_deb - init_e_deb) / abs(init_e_deb) * 100.0) if init_e_deb not in (0, None) else 0.0
#         else:
#             drift_sat = None
#             drift_deb = None

#         return {
#             "closest_time": t_min,
#             "miss_distance": min_dist,
#             "relative_velocity": rel_v_at_min,
#             "collision": collision,
#             "conjunction": conjunction,
#             "energy_drift_sat_percent": drift_sat,
#             "energy_drift_deb_percent": drift_deb,
#         }

#     def run_monte_carlo(self, satellite: Entity, debris: Entity, duration: float, N: int = 500, use_engine1_escalation: bool = False) -> Dict[str, Any]:
#         """
#         Monte Carlo: perturb initial states using the provided covariance matrices in Entity (pos & vel).
#         Returns collision & conjunction probabilities and some statistics.

#         N: number of Monte-Carlo samples (500 is a reasonable starting point for hackathon/demo)
#         """
#         collisions = 0
#         conjunctions = 0
#         min_dists = []

#         # Ensure covariance matrices are valid 3x3
#         cov_pos_sat = np.atleast_2d(satellite.cov_pos)
#         cov_vel_sat = np.atleast_2d(satellite.cov_vel)
#         cov_pos_deb = np.atleast_2d(debris.cov_pos)
#         cov_vel_deb = np.atleast_2d(debris.cov_vel)

#         # Precompute Cholesky if possible for speed and numerical stability
#         try:
#             L_pos_sat = np.linalg.cholesky(cov_pos_sat)
#             L_vel_sat = np.linalg.cholesky(cov_vel_sat)
#             L_pos_deb = np.linalg.cholesky(cov_pos_deb)
#             L_vel_deb = np.linalg.cholesky(cov_vel_deb)
#             use_cholesky = True
#         except np.linalg.LinAlgError:
#             use_cholesky = False

#         for i in range(N):
#             if use_cholesky:
#                 z = np.random.normal(size=3)
#                 pert_pos_sat = L_pos_sat @ z
#                 z = np.random.normal(size=3)
#                 pert_vel_sat = L_vel_sat @ z

#                 z = np.random.normal(size=3)
#                 pert_pos_deb = L_pos_deb @ z
#                 z = np.random.normal(size=3)
#                 pert_vel_deb = L_vel_deb @ z
#             else:
#                 pert_pos_sat = np.random.multivariate_normal(np.zeros(3), cov_pos_sat)
#                 pert_vel_sat = np.random.multivariate_normal(np.zeros(3), cov_vel_sat)
#                 pert_pos_deb = np.random.multivariate_normal(np.zeros(3), cov_pos_deb)
#                 pert_vel_deb = np.random.multivariate_normal(np.zeros(3), cov_vel_deb)

#             sat_pert = Entity(
#                 satellite.position + pert_pos_sat,
#                 satellite.velocity + pert_vel_sat,
#                 ballistic_coeff=satellite.ballistic_coeff,
#                 cov_pos=cov_pos_sat,
#                 cov_vel=cov_vel_sat,
#             )
#             deb_pert = Entity(
#                 debris.position + pert_pos_deb,
#                 debris.velocity + pert_vel_deb,
#                 ballistic_coeff=debris.ballistic_coeff,
#                 cov_pos=cov_pos_deb,
#                 cov_vel=cov_vel_deb,
#             )

#             result = self.run(sat_pert, deb_pert, duration, use_engine1_escalation=use_engine1_escalation)
#             min_dist = float(result["miss_distance"])
#             min_dists.append(min_dist)
#             if min_dist <= COLLISION_RADIUS:
#                 collisions += 1
#             if min_dist <= DANGER_RADIUS:
#                 conjunctions += 1

#         coll_prob = collisions / float(N)
#         conj_prob = conjunctions / float(N)
#         avg_min = float(np.mean(min_dists))
#         std_min = float(np.std(min_dists))

#         return {
#             "collision_probability": coll_prob,
#             "conjunction_probability": conj_prob,
#             "average_miss_distance": avg_min,
#             "std_miss_distance": std_min,
#             "num_simulations": int(N),
#         }

# new engine
# src/engine/engine2.py
import numpy as np
from typing import Optional, Dict, Any
from src.physics.state import State
from src.physics.entity import Entity
from src.physics.forces import (
    NewtonianGravity,
    J2Perturbation,
    AtmosphericDrag,
    CompositeForce,
)
from src.physics.solver import RK4Solver
from src.physics.utils import specific_energy
from src.config.settings import COLLISION_RADIUS, DANGER_RADIUS
from src.engine.engine1 import Engine1
class Engine2:
    """
    High-fidelity physics-based collision confirmation engine.
    Features:
    - 3D RK4 propagation for satellite and debris with CompositeForce (gravity + J2 + optional drag)
    - Adaptive timestep near close approach
    - Energy-drift reporting (only when drag disabled)
    - Covariance-based Monte Carlo using numpy
    - Optional Engine-1 escalation to avoid unnecessary high-fidelity runs
    """
    def __init__(self, dt: float = 1.0, adaptive_threshold: float = 100000.0, enable_drag: bool = True):
        """
        dt: base timestep in seconds
        adaptive_threshold: distance in meters below which timestep is refined
        enable_drag: whether to include atmospheric drag in force model
        """
        self.dt_base = float(dt)
        self.adaptive_threshold = float(adaptive_threshold)
        self.enable_drag = bool(enable_drag)
        self.gravity = NewtonianGravity()
        self.j2 = J2Perturbation()
        self.engine1 = Engine1()
    def _get_force_model(self, ballistic_coeff: float) -> CompositeForce:
        models = [self.gravity, self.j2]
        if self.enable_drag:
            models.append(AtmosphericDrag(ballistic_coeff))
        return CompositeForce(*models)
    def run(self, satellite: Entity, debris: Entity, duration: float, use_engine1_escalation: bool = True) -> Dict[str, Any]:
        """
        Run a single high-fidelity propagation for the given duration (seconds).
        Returns dict:
          {
            "closest_time": float (seconds from start),
            "miss_distance": float (meters),
            "relative_velocity": float (m/s) at closest approach,
            "collision": bool,
            "conjunction": bool (within DANGER_RADIUS),
            "energy_drift_sat_percent": float | None,
            "energy_drift_deb_percent": float | None,
            "note": optional str
          }
        """
        # Initialize states
        sat_state = State(satellite.position, satellite.velocity)
        deb_state = State(debris.position, debris.velocity)
        # Force models and solvers
        force_sat = self._get_force_model(satellite.ballistic_coeff)
        force_deb = self._get_force_model(debris.ballistic_coeff)
        solver_sat = RK4Solver(force_sat)
        solver_deb = RK4Solver(force_deb)
        # Determine if either composite has drag (for energy reporting)
        has_drag = force_sat.has_drag() or force_deb.has_drag()
        init_e_sat = specific_energy(sat_state) if not has_drag else None
        init_e_deb = specific_energy(deb_state) if not has_drag else None
        t = 0.0
        min_dist = float("inf")
        t_min: Optional[float] = 0.0
        rel_v_at_min: Optional[float] = None
        # Initial distance check (t = 0)
        rel_pos = sat_state.r - deb_state.r
        dist0 = float(np.linalg.norm(rel_pos))
        if dist0 < min_dist:
            min_dist = dist0
            t_min = 0.0
            rel_v_at_min = float(np.linalg.norm(sat_state.v - deb_state.v))
        # Propagation loop with adaptive timestep
        while t < duration:
            # Choose dt based on proximity
            rel_pos = sat_state.r - deb_state.r
            dist = float(np.linalg.norm(rel_pos))
            if dist < self.adaptive_threshold:
                current_dt = max(self.dt_base / 10.0, 1e-3) # refine - but avoid extremely small dt
            else:
                current_dt = self.dt_base
            current_dt = min(current_dt, duration - t)
            # Step both bodies
            sat_state = solver_sat.step(sat_state, current_dt)
            deb_state = solver_deb.step(deb_state, current_dt)
            # Advance time and evaluate
            t += current_dt
            rel_pos = sat_state.r - deb_state.r
            dist = float(np.linalg.norm(rel_pos))
            if dist < min_dist:
                min_dist = dist
                t_min = float(t)
                rel_v_at_min = float(np.linalg.norm(sat_state.v - deb_state.v))
        collision = bool(min_dist <= COLLISION_RADIUS)
        conjunction = bool(min_dist <= DANGER_RADIUS)
        # Energy drift if conservative (no drag)
        if not has_drag:
            final_e_sat = specific_energy(sat_state)
            final_e_deb = specific_energy(deb_state)
            drift_sat = (abs(final_e_sat - init_e_sat) / abs(init_e_sat) * 100.0) if init_e_sat not in (0, None) else 0.0
            drift_deb = (abs(final_e_deb - init_e_deb) / abs(init_e_deb) * 100.0) if init_e_deb not in (0, None) else 0.0
        else:
            drift_sat = None
            drift_deb = None
        return {
            "closest_time": t_min,
            "miss_distance": min_dist,
            "relative_velocity": rel_v_at_min,
            "collision": collision,
            "conjunction": conjunction,
            "energy_drift_sat_percent": drift_sat,
            "energy_drift_deb_percent": drift_deb,
        }
    def run_monte_carlo(self, satellite: Entity, debris: Entity, duration: float, N: int = 500, use_engine1_escalation: bool = False) -> Dict[str, Any]:
        """
        Monte Carlo: perturb initial states using the provided covariance matrices in Entity (pos & vel).
        Returns collision & conjunction probabilities and some statistics.
        N: number of Monte-Carlo samples (500 is a reasonable starting point for hackathon/demo)
        """
        collisions = 0
        conjunctions = 0
        min_dists = []
        # Ensure covariance matrices are valid 3x3
        cov_pos_sat = np.atleast_2d(satellite.cov_pos)
        cov_vel_sat = np.atleast_2d(satellite.cov_vel)
        cov_pos_deb = np.atleast_2d(debris.cov_pos)
        cov_vel_deb = np.atleast_2d(debris.cov_vel)
        # Precompute Cholesky if possible for speed and numerical stability
        try:
            L_pos_sat = np.linalg.cholesky(cov_pos_sat)
            L_vel_sat = np.linalg.cholesky(cov_vel_sat)
            L_pos_deb = np.linalg.cholesky(cov_pos_deb)
            L_vel_deb = np.linalg.cholesky(cov_vel_deb)
            use_cholesky = True
        except np.linalg.LinAlgError:
            use_cholesky = False
        for i in range(N):
            if use_cholesky:
                z = np.random.normal(size=3)
                pert_pos_sat = L_pos_sat @ z
                z = np.random.normal(size=3)
                pert_vel_sat = L_vel_sat @ z
                z = np.random.normal(size=3)
                pert_pos_deb = L_pos_deb @ z
                z = np.random.normal(size=3)
                pert_vel_deb = L_vel_deb @ z
            else:
                pert_pos_sat = np.random.multivariate_normal(np.zeros(3), cov_pos_sat)
                pert_vel_sat = np.random.multivariate_normal(np.zeros(3), cov_vel_sat)
                pert_pos_deb = np.random.multivariate_normal(np.zeros(3), cov_pos_deb)
                pert_vel_deb = np.random.multivariate_normal(np.zeros(3), cov_vel_deb)
            sat_pert = Entity(
                satellite.position + pert_pos_sat,
                satellite.velocity + pert_vel_sat,
                ballistic_coeff=satellite.ballistic_coeff,
                cov_pos=cov_pos_sat,
                cov_vel=cov_vel_sat,
            )
            deb_pert = Entity(
                debris.position + pert_pos_deb,
                debris.velocity + pert_vel_deb,
                ballistic_coeff=debris.ballistic_coeff,
                cov_pos=cov_pos_deb,
                cov_vel=cov_vel_deb,
            )
            result = self.run(sat_pert, deb_pert, duration, use_engine1_escalation=use_engine1_escalation)
            min_dist = float(result["miss_distance"])
            min_dists.append(min_dist)
            if min_dist <= COLLISION_RADIUS:
                collisions += 1
            if min_dist <= DANGER_RADIUS:
                conjunctions += 1
        coll_prob = collisions / float(N)
        conj_prob = conjunctions / float(N)
        avg_min = float(np.mean(min_dists))
        std_min = float(np.std(min_dists))
        return {
            "collision_probability": coll_prob,
            "conjunction_probability": conj_prob,
            "average_miss_distance": avg_min,
            "std_miss_distance": std_min,
            "num_simulations": int(N),
        }


# # impovements  

# # src/engine/engine2.py
# import numpy as np
# from typing import Optional, Dict, Any
# from src.physics.state import State
# from src.physics.entity import Entity
# from src.physics.forces import (
#     NewtonianGravity,
#     J2Perturbation,
#     AtmosphericDrag,
#     CompositeForce,
# )
# from src.physics.solver import RK4Solver
# from src.physics.utils import specific_energy
# from src.config.settings import COLLISION_RADIUS, DANGER_RADIUS
# from src.engine.engine1 import Engine1
# import math
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import cpu_count

# def _mc_worker(chunk_idx, sat_arr, deb_arr, covs, ballistic_coeffs, duration, n_samples_chunk):
#     """
#     Worker that runs n_samples_chunk Monte Carlo samples locally (in a process).
#     sat_arr/deb_arr: base vectors (position, velocity)
#     covs: tuple of (cov_pos_sat, cov_vel_sat, cov_pos_deb, cov_vel_deb)
#     Returns (collisions, conjunctions, min_dists_list)
#     """
#     import numpy as _np
#     collisions = 0
#     conjunctions = 0
#     min_dists = []
#     cov_pos_sat, cov_vel_sat, cov_pos_deb, cov_vel_deb = covs
#     # Precompute Cholesky if possible
#     try:
#         L_pos_sat = _np.linalg.cholesky(cov_pos_sat)
#         L_vel_sat = _np.linalg.cholesky(cov_vel_sat)
#         L_pos_deb = _np.linalg.cholesky(cov_pos_deb)
#         L_vel_deb = _np.linalg.cholesky(cov_vel_deb)
#         use_cholesky = True
#     except Exception:
#         use_cholesky = False
#     # Local Engine2.run usage (we re-create engine inside worker to avoid pickling heavy objects)
#     from src.engine.engine2 import Engine2 as Engine2_local
#     engine_local = Engine2_local(dt=1.0, adaptive_threshold=5000.0, enable_drag=True)
#     for i in range(n_samples_chunk):
#         if use_cholesky:
#             z = _np.random.normal(size=3)
#             pert_pos_sat = L_pos_sat @ z
#             z = _np.random.normal(size=3)
#             pert_vel_sat = L_vel_sat @ z
#             z = _np.random.normal(size=3)
#             pert_pos_deb = L_pos_deb @ z
#             z = _np.random.normal(size=3)
#             pert_vel_deb = L_vel_deb @ z
#         else:
#             pert_pos_sat = _np.random.multivariate_normal(_np.zeros(3), cov_pos_sat)
#             pert_vel_sat = _np.random.multivariate_normal(_np.zeros(3), cov_vel_sat)
#             pert_pos_deb = _np.random.multivariate_normal(_np.zeros(3), cov_pos_deb)
#             pert_vel_deb = _np.random.multivariate_normal(_np.zeros(3), cov_vel_deb)
#         # Build Entities and run local engine
#         from src.physics.entity import Entity as EntityLocal
#         sat_entity = EntityLocal(sat_arr[0] + pert_pos_sat, sat_arr[1] + pert_vel_sat,
#                                  ballistic_coeff=ballistic_coeffs[0],
#                                  cov_pos=cov_pos_sat, cov_vel=cov_vel_sat)
#         deb_entity = EntityLocal(deb_arr[0] + pert_pos_deb, deb_arr[1] + pert_vel_deb,
#                                  ballistic_coeff=ballistic_coeffs[1],
#                                  cov_pos=cov_pos_deb, cov_vel=cov_vel_deb)
#         res = engine_local.run(sat_entity, deb_entity, duration, use_engine1_escalation=False)
#         md = float(res["miss_distance"])
#         min_dists.append(md)
#         if md <= COLLISION_RADIUS:
#             collisions += 1
#         if md <= DANGER_RADIUS:
#             conjunctions += 1
#     return collisions, conjunctions, min_dists

# class Engine2:
#     """
#     High-fidelity physics-based collision confirmation engine.
#     Features:
#     - 3D RK4 propagation for satellite and debris with CompositeForce (gravity + J2 + optional drag)
#     - Adaptive timestep near close approach
#     - Energy-drift reporting (only when drag disabled)
#     - Covariance-based Monte Carlo using numpy
#     - Optional Engine-1 escalation to avoid unnecessary high-fidelity runs
#     """
#     def __init__(self, dt: float = 1.0, adaptive_threshold: float = 100000.0, enable_drag: bool = True):
#         """
#         dt: base timestep in seconds
#         adaptive_threshold: distance in meters below which timestep is refined
#         enable_drag: whether to include atmospheric drag in force model
#         """
#         self.dt_base = float(dt)
#         self.adaptive_threshold = float(adaptive_threshold)
#         self.enable_drag = bool(enable_drag)
#         self.gravity = NewtonianGravity()
#         self.j2 = J2Perturbation()
#         self.engine1 = Engine1()
#     def _get_force_model(self, ballistic_coeff: float) -> CompositeForce:
#         models = [self.gravity, self.j2]
#         if self.enable_drag:
#             models.append(AtmosphericDrag(ballistic_coeff))
#         return CompositeForce(*models)
#     def run(self, satellite: Entity, debris: Entity, duration: float, use_engine1_escalation: bool = True) -> Dict[str, Any]:
#         """
#         Run a single high-fidelity propagation for the given duration (seconds).
#         Returns dict:
#           {
#             "closest_time": float (seconds from start),
#             "miss_distance": float (meters),
#             "relative_velocity": float (m/s) at closest approach,
#             "collision": bool,
#             "conjunction": bool (within DANGER_RADIUS),
#             "energy_drift_sat_percent": float | None,
#             "energy_drift_deb_percent": float | None,
#             "note": optional str
#           }
#         """
#         # Initialize states
#         sat_state = State(satellite.position, satellite.velocity)
#         deb_state = State(debris.position, debris.velocity)
#         # Force models and solvers
#         force_sat = self._get_force_model(satellite.ballistic_coeff)
#         force_deb = self._get_force_model(debris.ballistic_coeff)
#         solver_sat = RK4Solver(force_sat)
#         solver_deb = RK4Solver(force_deb)
#         # Determine if either composite has drag (for energy reporting)
#         has_drag = force_sat.has_drag() or force_deb.has_drag()
#         init_e_sat = specific_energy(sat_state) if not has_drag else None
#         init_e_deb = specific_energy(deb_state) if not has_drag else None
#         t = 0.0
#         min_dist = float("inf")
#         t_min: Optional[float] = 0.0
#         rel_v_at_min: Optional[float] = None
#         # Initial distance check (t = 0)
#         rel_pos = sat_state.r - deb_state.r
#         dist0 = float(np.linalg.norm(rel_pos))
#         if dist0 < min_dist:
#             min_dist = dist0
#             t_min = 0.0
#             rel_v_at_min = float(np.linalg.norm(sat_state.v - deb_state.v))
#         # Propagation loop with adaptive timestep
#         while t < duration:
#             # Choose dt based on proximity
#             rel_pos = sat_state.r - deb_state.r
#             dist = float(np.linalg.norm(rel_pos))
#             if dist < self.adaptive_threshold:
#                 current_dt = max(self.dt_base / 10.0, 1e-3) # refine - but avoid extremely small dt
#             else:
#                 current_dt = self.dt_base
#             current_dt = min(current_dt, duration - t)
#             # Step both bodies
#             sat_state = solver_sat.step(sat_state, current_dt)
#             deb_state = solver_deb.step(deb_state, current_dt)
#             # Advance time and evaluate
#             t += current_dt
#             rel_pos = sat_state.r - deb_state.r
#             dist = float(np.linalg.norm(rel_pos))
#             if dist < min_dist:
#                 min_dist = dist
#                 t_min = float(t)
#                 rel_v_at_min = float(np.linalg.norm(sat_state.v - deb_state.v))
#         collision = bool(min_dist <= COLLISION_RADIUS)
#         conjunction = bool(min_dist <= DANGER_RADIUS)
#         # Energy drift if conservative (no drag)
#         if not has_drag:
#             final_e_sat = specific_energy(sat_state)
#             final_e_deb = specific_energy(deb_state)
#             drift_sat = (abs(final_e_sat - init_e_sat) / abs(init_e_sat) * 100.0) if init_e_sat not in (0, None) else 0.0
#             drift_deb = (abs(final_e_deb - init_e_deb) / abs(init_e_deb) * 100.0) if init_e_deb not in (0, None) else 0.0
#         else:
#             drift_sat = None
#             drift_deb = None
#         return {
#             "closest_time": t_min,
#             "miss_distance": min_dist,
#             "relative_velocity": rel_v_at_min,
#             "collision": collision,
#             "conjunction": conjunction,
#             "energy_drift_sat_percent": drift_sat,
#             "energy_drift_deb_percent": drift_deb,
#         }
#     def run_monte_carlo(self, satellite: Entity, debris: Entity, duration: float, N: int = 500, use_engine1_escalation: bool = False) -> dict:
#         """
#         Parallel chunked Monte Carlo.
#         """
#         import numpy as _np
#         from math import ceil
#         # Prepare covariance matrices
#         cov_pos_sat = _np.atleast_2d(satellite.cov_pos)
#         cov_vel_sat = _np.atleast_2d(satellite.cov_vel)
#         cov_pos_deb = _np.atleast_2d(debris.cov_pos)
#         cov_vel_deb = _np.atleast_2d(debris.cov_vel)
#         n_jobs = min(max(1, cpu_count() - 1), 6)
#         per_job = int(ceil(N / float(n_jobs)))
#         jobs = []
#         # Prepare base arrays for pickling
#         sat_arr = (satellite.position, satellite.velocity)
#         deb_arr = (debris.position, debris.velocity)
#         covs = (cov_pos_sat, cov_vel_sat, cov_pos_deb, cov_vel_deb)
#         ballistic_coeffs = (satellite.ballistic_coeff, debris.ballistic_coeff)
#         collisions = 0
#         conjunctions = 0
#         min_dists_all = []
#         with ProcessPoolExecutor(max_workers=n_jobs) as ex:
#             futures = []
#             for j in range(n_jobs):
#                 n_chunk = per_job if j < n_jobs - 1 else max(0, N - per_job * (n_jobs - 1))
#                 if n_chunk <= 0:
#                     continue
#                 futures.append(ex.submit(_mc_worker, j, sat_arr, deb_arr, covs, ballistic_coeffs, float(duration), n_chunk))
#             for fut in as_completed(futures):
#                 try:
#                     c, conj, mins = fut.result()
#                     collisions += c
#                     conjunctions += conj
#                     min_dists_all.extend(mins)
#                 except Exception as e:
#                     # one chunk failed – continue
#                     print("[MC] chunk failed:", e)
#         if len(min_dists_all) == 0:
#             avg_min = float("nan")
#             std_min = float("nan")
#         else:
#             avg_min = float(_np.mean(min_dists_all))
#             std_min = float(_np.std(min_dists_all))
#         coll_prob = collisions / float(N)
#         conj_prob = conjunctions / float(N)
#         return {
#             "collision_probability": coll_prob,
#             "conjunction_probability": conj_prob,
#             "average_miss_distance": avg_min,
#             "std_miss_distance": std_min,
#             "num_simulations": int(N),
#         }