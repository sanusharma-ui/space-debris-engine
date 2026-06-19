"""
Engine-2: High-fidelity physics-based collision confirmation engine (Stage-2).

Features:
- RK45 (Dormand-Prince) adaptive integrator for accurate propagation.
- Composite force model with J2/J3/J4, atmospheric drag, SRP, and third-body (Sun/Moon).
- Adaptive timestep refinement near close approach.
- Energy-drift reporting only when the propagation is conservative (no drag/SRP/third-body).
- Covariance-aware Monte Carlo (uses Entity.cov_pos / cov_vel).
"""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import numpy as np

from src.config import settings
from src.physics.state import State
from src.physics.entity import Entity
from src.physics.forces import (
    NewtonianGravity,
    J2Perturbation,
    J3Perturbation,
    J4Perturbation, 
    AtmosphericDrag,
    SolarRadiationPressure,
    ThirdBodyForce,
    CompositeForce,
)
from src.physics.solver_rk45 import RK45Solver
from src.physics.utils import specific_energy
from src.physics.ephemeris import seconds_since_j2000
from src.physics.covariance import propagate_covariance
from src.physics.probability import collision_probability
from src.config.settings import COLLISION_RADIUS, DANGER_RADIUS
from src.engine.engine1 import Engine1

# small numeric tolerance
_MIN_DT = float(getattr(settings, "ENGINE2_DT_MIN", 1e-4))


def _composite_has_nonconservative(force: CompositeForce) -> bool:
    """
    Heuristic: consider a composite non-conservative if it contains AtmosphericDrag,
    SolarRadiationPressure, or ThirdBodyForce (third-body introduces multi-body energy
    exchange relative to two-body specific energy).
    """
    for m in getattr(force, "models", []):
        if isinstance(m, (AtmosphericDrag, SolarRadiationPressure, ThirdBodyForce)):
            return True
    return False


def _energy_drift_supported(force: CompositeForce) -> bool:
    """
    Only report energy drift when the force model matches the potential used in specific_energy().
    specific_energy() currently includes Newtonian + J2 only.
    """
    allowed = (NewtonianGravity, J2Perturbation)
    for m in getattr(force, "models", []):
        if not isinstance(m, allowed):
            return False
    return True


def _linear_segment_closest_approach(
    rel_r0: np.ndarray,
    rel_v0: np.ndarray,
    rel_r1: np.ndarray,
    rel_v1: np.ndarray,
    dt: float,
) -> tuple[float, float, float]:
    """
    Estimate closest approach inside a macro step using relative linear motion.
    Returns (tau, miss_distance, relative_speed) with tau in [0, dt].
    """
    dt = float(dt)
    if dt <= 0.0:
        return 0.0, float(np.linalg.norm(rel_r0)), float(np.linalg.norm(rel_v0))

    rel_v = 0.5 * (np.asarray(rel_v0, dtype=float) + np.asarray(rel_v1, dtype=float))
    denom = float(np.dot(rel_v, rel_v))
    if denom <= 1e-18:
        tau = 0.0
    else:
        tau = -float(np.dot(rel_r0, rel_v)) / denom
        tau = max(0.0, min(dt, tau))

    rel_r_tau = np.asarray(rel_r0, dtype=float) + rel_v * tau
    return tau, float(np.linalg.norm(rel_r_tau)), float(np.linalg.norm(rel_v))


class Engine2:
    """
    Engine-2: Confirmation engine. Focus on accuracy and diagnostics.
    """
    def __init__(
        self,
        dt: float = 1.0,
        adaptive_threshold: float = 5000.0,
        enable_drag: bool = True,
        enable_srp: Optional[bool] = None,
        enable_third_body: Optional[bool] = None,
        srp_Cr: float = 1.2,
        srp_area_mass_ratio: float = 0.02,
    ):
        """
        dt: base timestep in seconds (used as nominal step)
        adaptive_threshold: refine timestep when relative distance < this (meters)
        enable_drag: include atmospheric drag model
        enable_srp: include solar radiation pressure
        enable_third_body: include Sun/Moon third-body gravity
        srp_Cr: reflectivity coefficient
        srp_area_mass_ratio: spacecraft area-to-mass ratio (A/m) [m^2/kg]
        """
        self.dt_base = float(dt)
        self.adaptive_threshold = float(adaptive_threshold)
        self.enable_drag = bool(enable_drag)
        self.enable_srp = bool(
            getattr(settings, "ENGINE2_ENABLE_SRP_DEFAULT", True)
            if enable_srp is None
            else enable_srp
        )
        self.enable_third_body = bool(
            getattr(settings, "ENGINE2_ENABLE_THIRD_BODY_DEFAULT", True)
            if enable_third_body is None
            else enable_third_body
        )
        self.srp_Cr = float(srp_Cr)
        self.srp_area_mass_ratio = float(srp_area_mass_ratio)

        # Engine-1 instance for optional escalation checks (lightweight screening)
        self.engine1 = Engine1()

    def _get_force_model(self, ballistic_coeff: float):
        """
        Build CompositeForce for an object given ballistic coefficient.
        Order: central + J2/J3/J4, optional drag, optional SRP, optional third-body.
        """
        models = [
            NewtonianGravity(),
            J2Perturbation(),
            J3Perturbation(),
            J4Perturbation(),
        ]
        if self.enable_drag:
            models.append(AtmosphericDrag(ballistic_coeff))
        if self.enable_srp:
            models.append(
                SolarRadiationPressure(
                    Cr=self.srp_Cr,
                    area_mass_ratio=self.srp_area_mass_ratio,
                )
            )
        if self.enable_third_body:
            models.append(ThirdBodyForce(include_sun=True, include_moon=True))
        return CompositeForce(*models)

    def run(self, satellite: Entity, debris: Entity, duration: float, use_engine1_escalation: bool = True) -> Dict[str, Any]:
        """
        Run a single high-fidelity propagation for the given duration (seconds).

        Returns:
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
        # Optional quick escalation check using Engine-1 (helps skip unnecessary high-fidelity runs)
        if use_engine1_escalation:
            try:
                screening = self.engine1.run(
                    satellite,
                    [debris],
                    dt=self.dt_base,
                    steps=max(1, int(duration / self.dt_base)),
                )
                if isinstance(screening, dict) and "screening" in screening:
                    recs = screening["screening"]
                    if recs and isinstance(recs, list):
                        p = float(recs[0].get("probability", recs[0].get("prob", 0.0)))
                        if p <= 0.0:
                            return {
                                "closest_time": None,
                                "miss_distance": float(recs[0].get("miss_distance", float("inf"))),
                                "relative_velocity": None,
                                "collision": False,
                                "conjunction": False,
                                "energy_drift_sat_percent": None,
                                "energy_drift_deb_percent": None,
                                "note": "Skipped by Engine-1 escalation (screening indicated no risk)."
                            }
            except Exception:
                pass

        # Initialize states
        sat_state = State(satellite.position.copy(), satellite.velocity.copy())
        deb_state = State(debris.position.copy(), debris.velocity.copy())
        sat_initial = sat_state.copy()
        deb_initial = deb_state.copy()

        epoch_utc = (
            getattr(satellite, "epoch_utc", None)
            or getattr(debris, "epoch_utc", None)
            or datetime.now(timezone.utc)
        )
        epoch_seconds = seconds_since_j2000(epoch_utc)

        # Configure force models and RK45 solvers
        force_sat = self._get_force_model(satellite.ballistic_coeff)
        force_deb = self._get_force_model(debris.ballistic_coeff)

        solver_sat = RK45Solver(
            force_sat,
            rtol=float(getattr(settings, "ENGINE2_RK45_RTOL", 1e-9)),
            atol=float(getattr(settings, "ENGINE2_RK45_ATOL", 1e-11)),
            dt_min=_MIN_DT,
            dt_max=float(getattr(settings, "ENGINE2_DT_MAX", 10.0)),
        )
        solver_deb = RK45Solver(
            force_deb,
            rtol=float(getattr(settings, "ENGINE2_RK45_RTOL", 1e-9)),
            atol=float(getattr(settings, "ENGINE2_RK45_ATOL", 1e-11)),
            dt_min=_MIN_DT,
            dt_max=float(getattr(settings, "ENGINE2_DT_MAX", 10.0)),
        )

        # Determine whether propagation is conservative for energy-drift reporting
        nonconservative = _composite_has_nonconservative(force_sat) or _composite_has_nonconservative(force_deb)

        can_report_energy = (
            (not nonconservative)
            and _energy_drift_supported(force_sat)
            and _energy_drift_supported(force_deb)
        )

        init_e_sat = specific_energy(sat_state) if can_report_energy else None
        init_e_deb = specific_energy(deb_state) if can_report_energy else None

        t = 0.0
        min_dist = float("inf")
        t_min: Optional[float] = 0.0
        rel_v_at_min: Optional[float] = None
        rel_pos_at_min: Optional[np.ndarray] = None
        rel_vel_at_min_vec: Optional[np.ndarray] = None
        macro_steps = 0
        max_macro_steps = int(getattr(settings, "ENGINE2_MAX_MACRO_STEPS", 2_000_000))
        note = None

        # initial distance check
        rel_pos = sat_state.r - deb_state.r
        rel_vel = sat_state.v - deb_state.v
        dist0 = float(np.linalg.norm(rel_pos))
        if dist0 < min_dist:
            min_dist = dist0
            t_min = 0.0
            rel_v_at_min = float(np.linalg.norm(rel_vel))
            rel_pos_at_min = rel_pos.copy()
            rel_vel_at_min_vec = rel_vel.copy()

        # Propagation loop with adaptive timestep
        while t < duration and macro_steps < max_macro_steps:
            rel_pos = sat_state.r - deb_state.r
            rel_vel = sat_state.v - deb_state.v
            dist = float(np.linalg.norm(rel_pos))
            rel_speed = float(np.linalg.norm(rel_vel))

            # choose dt: refine when close
            if dist < self.adaptive_threshold:
                subdivisions = max(1, int(getattr(settings, "ENGINE2_NEAR_APPROACH_SUBDIVISIONS", 10)))
                current_dt = max(self.dt_base / float(subdivisions), _MIN_DT)
            else:
                current_dt = self.dt_base

            if rel_speed > 1e-9 and dist < (5.0 * self.adaptive_threshold):
                ttc = -float(np.dot(rel_pos, rel_vel)) / (rel_speed * rel_speed)
                if 0.0 < ttc < current_dt:
                    current_dt = max(_MIN_DT, min(current_dt, ttc / 2.0))

            # avoid overshoot
            current_dt = min(current_dt, duration - t)

            sat_prev = sat_state.copy()
            deb_prev = deb_state.copy()

            # Step both bodies (RK45 accepts (state, dt, t0))
            force_time = epoch_seconds + t
            sat_state = solver_sat.step(sat_state, current_dt, force_time)
            deb_state = solver_deb.step(deb_state, current_dt, force_time)

            rel_r0 = sat_prev.r - deb_prev.r
            rel_v0 = sat_prev.v - deb_prev.v
            rel_r1 = sat_state.r - deb_state.r
            rel_v1 = sat_state.v - deb_state.v
            tau, seg_dist, seg_rel_speed = _linear_segment_closest_approach(
                rel_r0,
                rel_v0,
                rel_r1,
                rel_v1,
                current_dt,
            )
            t += current_dt
            macro_steps += 1

            # evaluate miss
            if seg_dist < min_dist:
                rel_v_avg = 0.5 * (rel_v0 + rel_v1)
                min_dist = seg_dist
                t_min = float(t - current_dt + tau)
                rel_v_at_min = seg_rel_speed
                rel_pos_at_min = rel_r0 + rel_v_avg * tau
                rel_vel_at_min_vec = rel_v_avg

        if macro_steps >= max_macro_steps and t < duration:
            note = "Stopped at Engine-2 macro-step safety limit before full duration."

        collision = bool(min_dist <= COLLISION_RADIUS)
        conjunction = bool(min_dist <= DANGER_RADIUS)

        # Energy drift only when supported by the energy model
        if can_report_energy:
            final_e_sat = specific_energy(sat_state)
            final_e_deb = specific_energy(deb_state)
            drift_sat = (abs(final_e_sat - init_e_sat) / abs(init_e_sat) * 100.0) if init_e_sat not in (0, None) else 0.0
            drift_deb = (abs(final_e_deb - init_e_deb) / abs(init_e_deb) * 100.0) if init_e_deb not in (0, None) else 0.0
        else:
            drift_sat = None
            drift_deb = None

        covariance_probability = None
        try:
            if t_min is not None and rel_pos_at_min is not None and rel_vel_at_min_vec is not None:
                P_sat, _, _ = propagate_covariance(
                    sat_initial.r,
                    sat_initial.v,
                    satellite.cov_pos,
                    satellite.cov_vel,
                    float(t_min),
                )
                P_deb, _, _ = propagate_covariance(
                    deb_initial.r,
                    deb_initial.v,
                    debris.cov_pos,
                    debris.cov_vel,
                    float(t_min),
                )
                covariance_probability = float(
                    collision_probability(
                        miss_distance=min_dist,
                        cov_rel=P_sat + P_deb,
                        rel_pos=rel_pos_at_min,
                        rel_vel=rel_vel_at_min_vec,
                        collision_radius=COLLISION_RADIUS,
                    )
                )
        except Exception:
            covariance_probability = None

        out = {
            "closest_time": t_min,
            "miss_distance": min_dist,
            "relative_velocity": rel_v_at_min,
            "collision_probability": covariance_probability,
            "collision": collision,
            "conjunction": conjunction,
            "energy_drift_sat_percent": drift_sat,
            "energy_drift_deb_percent": drift_deb,
            "macro_steps": int(macro_steps),
            "solver_stats": {
                "satellite": dict(solver_sat.stats),
                "debris": dict(solver_deb.stats),
            },
            "force_models": {
                "satellite": list(getattr(force_sat, "model_names", ())),
                "debris": list(getattr(force_deb, "model_names", ())),
            },
            "epoch_j2000_seconds": float(epoch_seconds),
        }
        if note:
            out["note"] = note
        return out

    def run_monte_carlo(
        self,
        satellite: Entity,
        debris: Entity,
        duration: float,
        N: int = 500,
        use_engine1_escalation: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Monte Carlo: perturb initial states using the provided covariance matrices in Entity (pos & vel).
        Vectorized sampling for perturbations, reusing Cholesky factors when possible.
        """
        max_n = int(getattr(settings, "MC_MAX_N", 5000))
        N_requested = int(N)
        N = max(1, min(N_requested, max_n))
        if seed is None:
            seed = getattr(settings, "MC_RANDOM_SEED", None)
        rng = np.random.default_rng(seed)

        collisions = 0
        conjunctions = 0
        min_dists = []

        cov_pos_sat = np.atleast_2d(satellite.cov_pos)
        cov_vel_sat = np.atleast_2d(satellite.cov_vel)
        cov_pos_deb = np.atleast_2d(debris.cov_pos)
        cov_vel_deb = np.atleast_2d(debris.cov_vel)

        # Try Cholesky (regularize SPD if necessary) - use make_spd if available
        use_cholesky = False
        try:
            from src.physics.covariance import make_spd
            cov_pos_sat_spd = make_spd(cov_pos_sat)
            cov_vel_sat_spd = make_spd(cov_vel_sat)
            cov_pos_deb_spd = make_spd(cov_pos_deb)
            cov_vel_deb_spd = make_spd(cov_vel_deb)

            L_pos_sat = np.linalg.cholesky(cov_pos_sat_spd)
            L_vel_sat = np.linalg.cholesky(cov_vel_sat_spd)
            L_pos_deb = np.linalg.cholesky(cov_pos_deb_spd)
            L_vel_deb = np.linalg.cholesky(cov_vel_deb_spd)
            use_cholesky = True
        except Exception:
            use_cholesky = False

        if use_cholesky:
            Z_pos_sat = rng.normal(size=(N, 3))
            Z_vel_sat = rng.normal(size=(N, 3))
            Z_pos_deb = rng.normal(size=(N, 3))
            Z_vel_deb = rng.normal(size=(N, 3))

            pert_pos_sat_all = Z_pos_sat @ L_pos_sat.T
            pert_vel_sat_all = Z_vel_sat @ L_vel_sat.T
            pert_pos_deb_all = Z_pos_deb @ L_pos_deb.T
            pert_vel_deb_all = Z_vel_deb @ L_vel_deb.T

            for i in range(N):
                pert_pos_sat = pert_pos_sat_all[i]
                pert_vel_sat = pert_vel_sat_all[i]
                pert_pos_deb = pert_pos_deb_all[i]
                pert_vel_deb = pert_vel_deb_all[i]

                sat_pert = Entity(
                    position=satellite.position + pert_pos_sat,
                    velocity=satellite.velocity + pert_vel_sat,
                    ballistic_coeff=satellite.ballistic_coeff,
                    cov_pos=cov_pos_sat,
                    cov_vel=cov_vel_sat,
                )
                deb_pert = Entity(
                    position=debris.position + pert_pos_deb,
                    velocity=debris.velocity + pert_vel_deb,
                    ballistic_coeff=debris.ballistic_coeff,
                    cov_pos=cov_pos_deb,
                    cov_vel=cov_vel_deb,
                )

                res = self.run(sat_pert, deb_pert, duration, use_engine1_escalation=use_engine1_escalation)
                min_dist = float(res.get("miss_distance", float("inf")))
                min_dists.append(min_dist)
                if min_dist <= COLLISION_RADIUS:
                    collisions += 1
                if min_dist <= DANGER_RADIUS:
                    conjunctions += 1
        else:
            for i in range(N):
                pert_pos_sat = rng.multivariate_normal(np.zeros(3), cov_pos_sat)
                pert_vel_sat = rng.multivariate_normal(np.zeros(3), cov_vel_sat)
                pert_pos_deb = rng.multivariate_normal(np.zeros(3), cov_pos_deb)
                pert_vel_deb = rng.multivariate_normal(np.zeros(3), cov_vel_deb)

                sat_pert = Entity(
                    position=satellite.position + pert_pos_sat,
                    velocity=satellite.velocity + pert_vel_sat,
                    ballistic_coeff=satellite.ballistic_coeff,
                    cov_pos=cov_pos_sat,
                    cov_vel=cov_vel_sat,
                )
                deb_pert = Entity(
                    position=debris.position + pert_pos_deb,
                    velocity=debris.velocity + pert_vel_deb,
                    ballistic_coeff=debris.ballistic_coeff,
                    cov_pos=cov_pos_deb,
                    cov_vel=cov_vel_deb,
                )

                res = self.run(sat_pert, deb_pert, duration, use_engine1_escalation=use_engine1_escalation)
                min_dist = float(res.get("miss_distance", float("inf")))
                min_dists.append(min_dist)
                if min_dist <= COLLISION_RADIUS:
                    collisions += 1
                if min_dist <= DANGER_RADIUS:
                    conjunctions += 1

        coll_prob = collisions / float(N)
        conj_prob = conjunctions / float(N)
        avg_min = float(np.mean(min_dists)) if min_dists else float("inf")
        std_min = float(np.std(min_dists)) if min_dists else 0.0
        percentiles = (
            np.percentile(min_dists, [1.0, 5.0, 50.0, 95.0, 99.0])
            if min_dists
            else np.array([float("inf")] * 5)
        )

        return {
            "collision_probability": coll_prob,
            "conjunction_probability": conj_prob,
            "average_miss_distance": avg_min,
            "std_miss_distance": std_min,
            "miss_distance_percentiles": {
                "p01": float(percentiles[0]),
                "p05": float(percentiles[1]),
                "p50": float(percentiles[2]),
                "p95": float(percentiles[3]),
                "p99": float(percentiles[4]),
            },
            "num_simulations": int(N),
            "requested_simulations": int(N_requested),
            "random_seed": seed,
            "sampling": "cholesky" if use_cholesky else "multivariate_normal",
        }
