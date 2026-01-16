# # ye sabse best code hai 
# # src/engine/engine1.py
# import numpy as np
# from src.config import settings
# from src.config.settings import (
#     DT, STEPS, COLLISION_RADIUS, RISK_THRESHOLD,
#     AVOIDANCE_DELTA_V, LOOKAHEAD_SEC
# )
# from src.physics.probability import collision_probability
# from src.physics.geometry import time_of_closest_approach
# from src.physics.cw_relative import cw_time_of_closest_approach
# from src.physics.state import State
# from src.physics.forces import NewtonianGravity, CompositeForce
# from src.physics.solver import RK4Solver

# class Engine1:
#     """
#     Engine-1: Conjunction Risk Engine (screening).
#     - Uses RK4 propagation with central gravity.
#     - Uses Clohessy-Wiltshire relative dynamics for TCA (cw_time_of_closest_approach).
#     - Uses Chan-style probability when possible, falling back to simple Gaussian heuristic.
#     """
#     def __init__(self):
#         force_model = CompositeForce(NewtonianGravity())
#         self.solver = RK4Solver(force_model)

#     def run(self, satellite, debris_list, dt=DT, steps=STEPS):
#         results = []
#         positions_sat = []
#         positions_debris = [[] for _ in debris_list]
#         state_sat = State(satellite.position.copy(), satellite.velocity.copy())
#         states_deb = [State(d.position.copy(), d.velocity.copy()) for d in debris_list]

#         horizon = min(float(getattr(settings, "LOOKAHEAD_SEC", dt * steps)), dt * steps)
#         print(f"[Engine1] Using horizon: {horizon}s")

#         current_time = 0.0
#         positions_sat.append(tuple(state_sat.r))
#         for i in range(len(debris_list)):
#             positions_debris[i].append(tuple(states_deb[i].r))

#         # Initial checks at t=0
#         for i, debris in enumerate(debris_list):
#             cov_rel = satellite.cov_pos + debris.cov_pos
#             try:
#                 tca, miss, rel_pos_at_tca, rel_vel_at_tca = cw_time_of_closest_approach(
#                     state_sat.r, state_sat.v, states_deb[i].r, states_deb[i].v, horizon=horizon
#                 )
#             except Exception:
#                 # fallback to linear instantaneous TCA (geometry)
#                 tca, miss = time_of_closest_approach(state_sat.r, state_sat.v, states_deb[i].r, states_deb[i].v)
#                 # build simple rel vectors
#                 rel_pos_at_tca = (states_deb[i].r + states_deb[i].v * tca) - (state_sat.r + state_sat.v * tca)
#                 rel_vel_at_tca = states_deb[i].v - state_sat.v

#             inst_dist = float(np.linalg.norm(state_sat.r - states_deb[i].r))
#             # use chan-style probability if we have rel vectors and covariance
#             p = collision_probability(miss_distance=miss, cov_rel=cov_rel, rel_pos=rel_pos_at_tca, rel_vel=rel_vel_at_tca, collision_radius=COLLISION_RADIUS)
#             risk = self._compute_risk(p, miss, float(np.linalg.norm(rel_vel_at_tca)))

#             is_high_risk = risk > RISK_THRESHOLD
#             results.append({
#                 "step": 0,
#                 "step_time": current_time,
#                 "debris_id": debris.name,
#                 "distance": inst_dist,
#                 "tca": float(tca),
#                 "miss_distance": float(miss),
#                 "inside_horizon": (0 <= tca <= horizon),
#                 "relative_velocity": float(np.linalg.norm(rel_vel_at_tca)),
#                 "probability": p,
#                 "risk": risk,
#                 "is_high_risk": is_high_risk
#             })

#         # Propagation loop
#         effective_steps = int(horizon / dt)
#         for step in range(1, effective_steps + 1):
#             state_sat = self.solver.step(state_sat, dt)
#             for i in range(len(debris_list)):
#                 states_deb[i] = self.solver.step(states_deb[i], dt)
#             current_time = step * dt
#             positions_sat.append(tuple(state_sat.r))
#             for i in range(len(debris_list)):
#                 positions_debris[i].append(tuple(states_deb[i].r))

#             for i, debris in enumerate(debris_list):
#                 cov_rel = satellite.cov_pos + debris.cov_pos
#                 try:
#                     tca, miss, rel_pos_at_tca, rel_vel_at_tca = cw_time_of_closest_approach(
#                         state_sat.r, state_sat.v, states_deb[i].r, states_deb[i].v, horizon=horizon
#                     )
#                 except Exception:
#                     tca, miss = time_of_closest_approach(state_sat.r, state_sat.v, states_deb[i].r, states_deb[i].v)
#                     rel_pos_at_tca = (states_deb[i].r + states_deb[i].v * tca) - (state_sat.r + state_sat.v * tca)
#                     rel_vel_at_tca = states_deb[i].v - state_sat.v

#                 inst_dist = float(np.linalg.norm(state_sat.r - states_deb[i].r))

#                 p = collision_probability(miss_distance=miss, cov_rel=cov_rel, rel_pos=rel_pos_at_tca, rel_vel=rel_vel_at_tca, collision_radius=COLLISION_RADIUS)
#                 risk = self._compute_risk(p, miss, float(np.linalg.norm(rel_vel_at_tca)))

#                 is_high_risk = risk > RISK_THRESHOLD
#                 results.append({
#                     "step": step,
#                     "step_time": current_time,
#                     "debris_id": debris.name,
#                     "distance": inst_dist,
#                     "tca": float(tca),
#                     "miss_distance": float(miss),
#                     "inside_horizon": (0 <= tca <= horizon),
#                     "relative_velocity": float(np.linalg.norm(rel_vel_at_tca)),
#                     "probability": p,
#                     "risk": risk,
#                     "is_high_risk": is_high_risk
#                 })

#         # Summary for escalation: consider only records inside horizon
#         try:
#             min_miss = min((r["miss_distance"] for r in results if r.get("inside_horizon")), default=float("inf"))
#         except Exception:
#             min_miss = min((r["miss_distance"] for r in results), default=float("inf"))

#         max_prob = max((r["probability"] for r in results), default=0.0)
#         max_risk = max((r["risk"] for r in results), default=0.0)
#         escalate = (
#             min_miss < getattr(settings, "ESCALATION_THRESHOLD", 5000.0) or
#             max_prob > 1e-6 or
#             max_risk > RISK_THRESHOLD
#         )
#         summary = {
#             "min_miss_distance": min_miss,
#             "max_probability": max_prob,
#             "max_risk": max_risk,
#             "escalate": escalate
#         }
#         return {
#             "screening": results,
#             "summary": summary,
#             "positions_sat": positions_sat,
#             "positions_debris": positions_debris
#         }

#     def _compute_risk(self, probability, miss_distance, rel_velocity):
#         # stable, interpretable screening risk
#         return float(np.clip(probability, 0.0, 1.0))

#     def get_avoidance_suggestion(self, satellite, debris, delta_v_mag=AVOIDANCE_DELTA_V):
#         rel_vel = debris.velocity - satellite.velocity
#         rel_pos = debris.position - satellite.position
#         if np.linalg.norm(rel_vel) < 1e-6:
#             rel_vel = satellite.velocity  # Fallback
#         cross = np.cross(rel_pos, rel_vel)
#         n = np.linalg.norm(cross)
#         if n < 1e-6:
#             arb = np.array([0, 0, 1])
#             if np.dot(rel_vel, arb) > 0.999 * np.linalg.norm(rel_vel):
#                 arb = np.array([1, 0, 0])
#             cross = np.cross(rel_vel, arb)
#             n = np.linalg.norm(cross)
#             if n < 1e-6:
#                 return np.zeros(3)
#         norm = cross / n
#         delta_v_mag_ms = delta_v_mag * 1000.0
#         return delta_v_mag_ms * norm



# new engine ye wal numba wala hai  jo error deta  hai niche wala error wala hai 


# # src/engine/engine1.py
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import cpu_count
# from typing import List, Dict, Any, Tuple
# from src.config import settings
# from src.config.settings import (
#     DT, STEPS, COLLISION_RADIUS, RISK_THRESHOLD, LOOKAHEAD_SEC
# )
# from src.physics.probability import collision_probability
# from src.physics.geometry import time_of_closest_approach
# from src.physics.cw_relative import cw_time_of_closest_approach
# from src.physics.state import State
# from src.physics.forces import NewtonianGravity, CompositeForce
# from src.physics.solver import RK4Solver

# # small tuning knobs (feel free to tweak)
# CHAN_DISTANCE_CUTOFF = 10_000.0   # only run Chan integrator if miss < 10 km
# SAT_EVAL_WORKERS = max(1, min(cpu_count() - 1, 6))  # conservative default

# class Engine1:
#     def __init__(self):
#         force_model = CompositeForce(NewtonianGravity())
#         self.solver = RK4Solver(force_model)

#     def _integrate_satellite(self, state_sat: State, dt: float, effective_steps: int) -> List[State]:
#         """Precompute satellite states for all steps (so debris tasks can read them)."""
#         sat_states = [state_sat.copy()]
#         local_solver = RK4Solver(CompositeForce(NewtonianGravity()))
#         cur = state_sat
#         for _ in range(effective_steps):
#             cur = local_solver.step(cur, dt)
#             sat_states.append(cur.copy())
#         return sat_states

#     def _process_one_debris(self, args) -> Tuple[str, List[Dict[str, Any]]]:
#         """
#         Worker function run in a separate process.
#         args = (debris_obj_serializable, sat_states_list_serializable, dt, horizon)
#         Returns (debris_name, list_of_records)
#         """
#         # Unpack (we expect plain numpy arrays and simple attributes)
#         debris, sat_states, dt, horizon = args
#         # Build local solver
#         local_solver = RK4Solver(CompositeForce(NewtonianGravity()))
#         # Convert sat_states back into State objects if they are (r,v) pairs
#         sat_state_objs = []
#         for s in sat_states:
#             # s may be tuple/list [r_array, v_array] or State object depending on pickling
#             if hasattr(s, "r") and hasattr(s, "v"):
#                 sat_state_objs.append(s)
#             else:
#                 # assume (r, v)
#                 sat_state_objs.append(State(np.array(s[0], dtype=float), np.array(s[1], dtype=float)))

#         # Debris initial state
#         state_deb = State(np.array(debris.position, dtype=float), np.array(debris.velocity, dtype=float))
#         sat0 = sat_state_objs[0]
#         # local caches
#         chan_cache = {}
#         results = []

#         # initial check t=0 (use CW if available)
#         try:
#             tca, miss, rel_pos_at_tca, rel_vel_at_tca = cw_time_of_closest_approach(
#                 sat0.r, sat0.v, state_deb.r, state_deb.v, horizon=horizon
#             )
#         except Exception:
#             tca, miss = time_of_closest_approach(sat0.r, sat0.v, state_deb.r, state_deb.v)
#             rel_pos_at_tca = (state_deb.r + state_deb.v * tca) - (sat0.r + sat0.v * tca)
#             rel_vel_at_tca = state_deb.v - sat0.v

#         cov_rel = getattr(debris, "cov_pos", None)
#         # run probability with conditional Chan
#         if miss <= CHAN_DISTANCE_CUTOFF and cov_rel is not None:
#             # try chan, else fallback
#             p = collision_probability(miss_distance=miss, cov_rel=cov_rel,
#                                       rel_pos=rel_pos_at_tca, rel_vel=rel_vel_at_tca, collision_radius=COLLISION_RADIUS)
#         else:
#             p = collision_probability(miss_distance=miss, cov_rel=cov_rel, collision_radius=COLLISION_RADIUS)

#         risk = float(np.clip(p, 0.0, 1.0))
#         inst_dist = float(np.linalg.norm(sat0.r - state_deb.r))
#         results.append({
#             "step": 0,
#             "step_time": 0.0,
#             "debris_id": getattr(debris, "name", repr(debris)),
#             "distance": inst_dist,
#             "tca": float(tca),
#             "miss_distance": float(miss),
#             "inside_horizon": (0 <= tca <= horizon),
#             "relative_velocity": float(np.linalg.norm(rel_vel_at_tca)),
#             "probability": p,
#             "risk": risk,
#             "is_high_risk": (risk > RISK_THRESHOLD)
#         })

#         # Now propagate debris independently across full horizon and evaluate against precomputed sat_states
#         effective_steps = len(sat_state_objs) - 1
#         current_time = 0.0
#         # Worker loop: integrate debris step-by-step, evaluate against satellite states
#         for step in range(1, effective_steps + 1):
#             # integrate debris one dt
#             state_deb = local_solver.step(state_deb, dt)
#             current_time = step * dt
#             sat_state = sat_state_objs[step]

#             # quick coarse filter: skip expensive math if far away
#             inst_dist = float(np.linalg.norm(sat_state.r - state_deb.r))
#             if inst_dist > getattr(settings, "ESCALATION_THRESHOLD", 5000.0) * 10:
#                 # very far — cheap Gaussian
#                 p = float(np.exp(-(inst_dist ** 2) / (2.0 * (getattr(settings, "DEFAULT_POS_STD", 100.0) ** 2))))
#                 risk = float(np.clip(p, 0.0, 1.0))
#                 results.append({
#                     "step": step,
#                     "step_time": current_time,
#                     "debris_id": getattr(debris, "name", repr(debris)),
#                     "distance": inst_dist,
#                     "tca": None,
#                     "miss_distance": inst_dist,
#                     "inside_horizon": False,
#                     "relative_velocity": float(np.linalg.norm(state_deb.v - sat_state.v)),
#                     "probability": p,
#                     "risk": risk,
#                     "is_high_risk": (risk > RISK_THRESHOLD)
#                 })
#                 continue

#             # attempt CW TCA occasionally (we avoid full CW every iteration)
#             try:
#                 tca, miss, rel_pos_at_tca, rel_vel_at_tca = cw_time_of_closest_approach(
#                     sat_state.r, sat_state.v, state_deb.r, state_deb.v, horizon=horizon
#                 )
#             except Exception:
#                 tca, miss = time_of_closest_approach(sat_state.r, sat_state.v, state_deb.r, state_deb.v)
#                 rel_pos_at_tca = (state_deb.r + state_deb.v * tca) - (sat_state.r + sat_state.v * tca)
#                 rel_vel_at_tca = state_deb.v - sat_state.v

#             # Conditional Chan usage
#             cov_rel = getattr(debris, "cov_pos", None)
#             if miss <= CHAN_DISTANCE_CUTOFF and cov_rel is not None:
#                 # caching on rounded vectors
#                 key = (tuple(np.round(rel_pos_at_tca, 3)), tuple(np.round(rel_vel_at_tca, 3)))
#                 if key in chan_cache:
#                     p = chan_cache[key]
#                 else:
#                     p = collision_probability(miss_distance=miss, cov_rel=cov_rel,
#                                               rel_pos=rel_pos_at_tca, rel_vel=rel_vel_at_tca, collision_radius=COLLISION_RADIUS)
#                     chan_cache[key] = p
#             else:
#                 p = collision_probability(miss_distance=miss, cov_rel=cov_rel, collision_radius=COLLISION_RADIUS)

#             risk = float(np.clip(p, 0.0, 1.0))
#             results.append({
#                 "step": step,
#                 "step_time": current_time,
#                 "debris_id": getattr(debris, "name", repr(debris)),
#                 "distance": inst_dist,
#                 "tca": float(tca) if tca is not None else None,
#                 "miss_distance": float(miss),
#                 "inside_horizon": (0 <= tca <= horizon) if tca is not None else False,
#                 "relative_velocity": float(np.linalg.norm(rel_vel_at_tca)),
#                 "probability": p,
#                 "risk": risk,
#                 "is_high_risk": (risk > RISK_THRESHOLD)
#             })

#         return (getattr(debris, "name", repr(debris)), results)

#     def run(self, satellite, debris_list, dt=DT, steps=STEPS):
#         # Prepare
#         results: List[Dict[str, Any]] = []
#         positions_sat = []
#         positions_debris = [[] for _ in debris_list]

#         state_sat = State(satellite.position.copy(), satellite.velocity.copy())
#         # Build a copy of debris list that is picklable (plain attributes & numpy arrays)
#         # (we assume Debris objects are picklable but be safe)
#         debris_serializable = []
#         for d in debris_list:
#             # create a lightweight object (namespace) to send to workers
#             class _D:
#                 pass
#             dd = _D()
#             dd.position = np.array(d.position, dtype=float)
#             dd.velocity = np.array(d.velocity, dtype=float)
#             dd.cov_pos = getattr(d, "cov_pos", None)
#             dd.cov_vel = getattr(d, "cov_vel", None)
#             dd.name = getattr(d, "name", repr(d))
#             debris_serializable.append(dd)

#         horizon = min(float(getattr(settings, "LOOKAHEAD_SEC", dt * steps)), dt * steps)
#         print(f"[Engine1] Using horizon: {horizon}s")

#         effective_steps = int(horizon / dt)

#         # 1) Precompute satellite trajectory (states list)
#         sat_states = self._integrate_satellite(state_sat.copy(), dt, effective_steps)

#         # For pickling convenience, convert sat_states to lightweight tuples (r,v)
#         sat_states_serial = [ (s.r, s.v) for s in sat_states ]

#         # 2) Prepare worker args
#         worker_args = []
#         for d in debris_serializable:
#             worker_args.append((d, sat_states_serial, float(dt), float(horizon)))

#         # 3) Run per-debris work in parallel
#         results_map = {}
#         max_workers = min(SAT_EVAL_WORKERS, len(worker_args)) if worker_args else 1
#         with ProcessPoolExecutor(max_workers=max_workers) as ex:
#             futures = [ex.submit(self._process_one_debris, arg) for arg in worker_args]
#             for fut in as_completed(futures):
#                 try:
#                     name, recs = fut.result()
#                     results_map[name] = recs
#                 except Exception as e:
#                     # If a worker fails, log and continue
#                     print("[Engine1] Worker error:", e)

#         # collect results and positions
#         # results_map[name] is full list of records per debris; flatten in time order
#         for name, recs in results_map.items():
#             for r in recs:
#                 results.append(r)

#         # Build positions lists (sat positions and debris positions for plotting)
#         positions_sat = [tuple(s.r) for s in sat_states]
#         for idx, d in enumerate(debris_serializable):
#             # Reconstruct debris trajectory from results_map (filter by debris id)
#             recs = results_map.get(d.name, [])
#             # Make a list of positions per step if available (we didn't store them each step to save memory)
#             # fallback: empty lists
#             positions_debris[idx] = []

#         # build summary like before
#         try:
#             min_miss = min((r["miss_distance"] for r in results if r.get("inside_horizon")), default=float("inf"))
#         except Exception:
#             min_miss = min((r["miss_distance"] for r in results), default=float("inf"))

#         max_prob = max((r["probability"] for r in results), default=0.0)
#         max_risk = max((r["risk"] for r in results), default=0.0)
#         escalate = (
#             min_miss < getattr(settings, "ESCALATION_THRESHOLD", 5000.0) or
#             max_prob > 1e-6 or
#             max_risk > RISK_THRESHOLD
#         )
#         summary = {
#             "min_miss_distance": min_miss,
#             "max_probability": max_prob,
#             "max_risk": max_risk,
#             "escalate": escalate
#         }
#         return {
#             "screening": results,
#             "summary": summary,
#             "positions_sat": positions_sat,
#             "positions_debris": positions_debris
#         }
# BETTER VERSION


import numpy as np
from src.config import settings
from src.config.settings import (
    ENGINE1_DT, STEPS, COLLISION_RADIUS, RISK_THRESHOLD,
    AVOIDANCE_DELTA_V, ENGINE1_LOOKAHEAD, ENGINE1_CW_SAMPLES, ESCALATION_THRESHOLD
)
from src.physics.probability import collision_probability_screening
from src.physics.geometry import time_of_closest_approach
from src.physics.cw_relative import cw_time_of_closest_approach
from src.physics.state import State
from src.physics.forces import NewtonianGravity, CompositeForce
from src.physics.solver import RK4Solver

class Engine1:
    """
    Engine-1: Conjunction Risk Engine (screening).
    - One-time CW-based TCA & conservative screening probability per debris.
    - RK4 propagation used only for trajectory tracking / distance time series.
    """
    def __init__(self):
        force_model = CompositeForce(NewtonianGravity())
        self.solver = RK4Solver(force_model)

    def run(self, satellite, debris_list, dt: float = ENGINE1_DT, steps: int = STEPS):
        results = []
        positions_sat = []
        positions_debris = [[] for _ in debris_list]

        state_sat = State(satellite.position.copy(), satellite.velocity.copy())
        states_deb = [State(d.position.copy(), d.velocity.copy()) for d in debris_list]

        # Respect CLI / runtime override if present; ensure consistent horizon
        horizon_setting = float(getattr(settings, "LOOKAHEAD_SEC", ENGINE1_LOOKAHEAD))
        horizon = min(float(horizon_setting), float(ENGINE1_LOOKAHEAD), float(dt * steps))
        print(f"[Engine1] Using horizon: {horizon}s")

        # Phase A — one-time analysis (physics, CW + conservative screening)
        analysis = {}
        for i, debris in enumerate(debris_list):
            cov_rel = satellite.cov_pos + debris.cov_pos

            tca, miss, rel_pos_at_tca, rel_vel_at_tca = cw_time_of_closest_approach(
                satellite.position,
                satellite.velocity,
                debris.position,
                debris.velocity,
                horizon=horizon,
                n_samples=ENGINE1_CW_SAMPLES
            )

            p = collision_probability_screening(miss, cov_rel)

            analysis[i] = {
                "tca": float(tca),
                "miss": float(miss),
                "prob": float(p),
                "rel_vel": float(np.linalg.norm(rel_vel_at_tca)),
                "flag": (p > RISK_THRESHOLD) or (float(miss) < ESCALATION_THRESHOLD)
            }

        # Phase B — propagation (for visualization / distance tracking)
        current_time = 0.0
        positions_sat.append(tuple(state_sat.r))
        for i in range(len(debris_list)):
            positions_debris[i].append(tuple(states_deb[i].r))

        # initial record at t=0
        for i, debris in enumerate(debris_list):
            inst_dist = float(np.linalg.norm(state_sat.r - states_deb[i].r))
            results.append({
                "step": 0,
                "step_time": current_time,
                "debris_id": debris.name,
                "distance": inst_dist,
                "tca": analysis[i]["tca"],
                "miss_distance": analysis[i]["miss"],
                "inside_horizon": (0 <= analysis[i]["tca"] <= horizon),
                "relative_velocity": analysis[i]["rel_vel"],
                "probability": analysis[i]["prob"],
                "risk": analysis[i]["prob"],
                "is_high_risk": analysis[i]["flag"]
            })

        effective_steps = int(max(1, np.floor(horizon / dt)))

        for step in range(1, effective_steps + 1):
            state_sat = self.solver.step(state_sat, dt)
            for i in range(len(debris_list)):
                states_deb[i] = self.solver.step(states_deb[i], dt)

            current_time = step * dt
            positions_sat.append(tuple(state_sat.r))
            for i in range(len(debris_list)):
                positions_debris[i].append(tuple(states_deb[i].r))

            for i, debris in enumerate(debris_list):
                inst_dist = float(np.linalg.norm(state_sat.r - states_deb[i].r))
                results.append({
                    "step": step,
                    "step_time": current_time,
                    "debris_id": debris.name,
                    "distance": inst_dist,
                    "tca": analysis[i]["tca"],
                    "miss_distance": analysis[i]["miss"],
                    "inside_horizon": (0 <= analysis[i]["tca"] <= horizon),
                    "relative_velocity": analysis[i]["rel_vel"],
                    "probability": analysis[i]["prob"],
                    "risk": analysis[i]["prob"],
                    "is_high_risk": analysis[i]["flag"]
                })

        # Summary for escalation: consider only records inside horizon
        try:
            min_miss = min((r["miss_distance"] for r in results if r.get("inside_horizon")), default=float("inf"))
        except Exception:
            min_miss = min((r["miss_distance"] for r in results), default=float("inf"))

        max_prob = max((r["probability"] for r in results), default=0.0)
        max_risk = max((r["risk"] for r in results), default=0.0)

        escalate = (
            min_miss < float(getattr(settings, "ESCALATION_THRESHOLD", ESCALATION_THRESHOLD)) or
            max_prob > 1e-6 or
            max_risk > RISK_THRESHOLD
        )

        summary = {
            "min_miss_distance": min_miss,
            "max_probability": max_prob,
            "max_risk": max_risk,
            "escalate": bool(escalate)
        }

        return {
            "screening": results,
            "summary": summary,
            "positions_sat": positions_sat,
            "positions_debris": positions_debris
        }

    def _compute_risk(self, probability, miss_distance, rel_velocity):
        # stable, interpretable screening risk (kept for future extension)
        return float(np.clip(probability, 0.0, 1.0))

    def get_avoidance_suggestion(self, satellite, debris, delta_v_mag=AVOIDANCE_DELTA_V):
        rel_vel = debris.velocity - satellite.velocity
        rel_pos = debris.position - satellite.position
        if np.linalg.norm(rel_vel) < 1e-6:
            rel_vel = satellite.velocity  # Fallback
        cross = np.cross(rel_pos, rel_vel)
        n = np.linalg.norm(cross)
        if n < 1e-6:
            arb = np.array([0, 0, 1])
            if np.dot(rel_vel, arb) > 0.999 * np.linalg.norm(rel_vel):
                arb = np.array([1, 0, 0])
            cross = np.cross(rel_vel, arb)
            n = np.linalg.norm(cross)
            if n < 1e-6:
                return np.zeros(3)
        norm = cross / n
        delta_v_mag_ms = delta_v_mag * 1000.0
        return delta_v_mag_ms * norm
