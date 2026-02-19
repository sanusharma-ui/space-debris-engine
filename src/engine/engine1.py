import numpy as np
from datetime import datetime, timezone, timedelta

from src.config import settings
from src.config.settings import (
    ENGINE1_DT,
    STEPS,
    COLLISION_RADIUS,
    AVOIDANCE_DELTA_V,
    ENGINE1_CW_SAMPLES,
)

from src.physics.probability import (
    collision_probability,
    collision_probability_screening,
)
from src.physics.cw_relative import cw_time_of_closest_approach
from src.physics.state import State
from src.physics.forces import NewtonianGravity, CompositeForce
from src.physics.solver import RK4Solver
from src.physics.covariance import propagate_covariance

# Phase-2 SGP4 helper
from src.propagation.sgp4_propagator import satrec_from_tle, propagate_teme_m


def _has_tle(obj) -> bool:
    return (
        hasattr(obj, "tle1")
        and hasattr(obj, "tle2")
        and bool(getattr(obj, "tle1", None))
        and bool(getattr(obj, "tle2", None))
    )


def _parabolic_tca_refine(times: np.ndarray, dists: np.ndarray, idx: int) -> tuple[float, float]:
    """
    3-point parabolic refinement around idx (idx-1, idx, idx+1).
    Returns (t_refined, d_refined). If not possible, returns sample time/dist.
    """
    if idx <= 0 or idx >= len(times) - 1:
        return float(times[idx]), float(dists[idx])

    t0, t1, t2 = float(times[idx - 1]), float(times[idx]), float(times[idx + 1])
    y0, y1, y2 = float(dists[idx - 1]), float(dists[idx]), float(dists[idx + 1])

    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return t1, y1

    # vertex location (assumes uniform dt)
    dt = t1 - t0
    delta = 0.5 * (y0 - y2) / denom
    t_star = t1 + delta * dt
    t_star = max(t0, min(t2, t_star))

    # approximate distance at vertex
    y_star = y1 - 0.25 * (y0 - y2) * delta
    return float(t_star), float(max(0.0, y_star))


class Engine1:
    """
    ENGINE-1 (STAGE-1): Fast probabilistic screening engine.
    Conservative, physics-aware screening (CW + covariance propagation to TCA).
    Phase-2 adds SGP4 screening if TLE is present (TEME frame).
    """

    def __init__(self):
        force_model = CompositeForce(NewtonianGravity())
        self.solver = RK4Solver(force_model)

    def _screening_risk_threshold(self) -> float:
        return float(
            getattr(
                settings,
                "SCREENING_RISK_THRESHOLD",
                getattr(settings, "RISK_THRESHOLD", 1e-6),
            )
        )

    def _screening_escalation_threshold(self) -> float:
        return float(
            getattr(
                settings,
                "SCREENING_ESCALATION_THRESHOLD",
                getattr(settings, "ESCALATION_THRESHOLD", 5000.0),
            )
        )

    def _screening_process_noise_sigma(self, prop_t: float) -> float:
        """
        Conservative process noise inflation (meters, 1-sigma).
        Uses settings knobs introduced in Phase-1.
        """
        growth = float(getattr(settings, "SCREENING_POS_GROWTH_M_PER_S", 0.01))
        floor = float(getattr(settings, "SCREENING_POS_GROWTH_FLOOR_M", 1.0))
        cap = float(getattr(settings, "SCREENING_POS_GROWTH_CAP_M", 250.0))
        q = max(floor, growth * float(prop_t))
        q = min(q, cap)
        return float(q)

    # -------------------------
    # Phase-2: SGP4 screening
    # -------------------------
    def _run_sgp4_screening(self, satellite, debris_list, horizon: float):
        sample_dt = float(getattr(settings, "ENGINE1_SGP4_SAMPLE_DT", 10.0))
        sample_dt = max(1.0, sample_dt)

        lookback = float(getattr(settings, "ENGINE1_LOOKBACK_SEC", 300.0))
        lookback = max(0.0, lookback)

        sigma0 = float(getattr(settings, "ENGINE1_SIGMA0", getattr(settings, "DEFAULT_POS_STD", 100.0)))
        k = float(getattr(settings, "ENGINE1_COV_GROWTH_RATE", 0.02))

        risk_thr = self._screening_risk_threshold()
        esc_thr = self._screening_escalation_threshold()

        # Use a fixed epoch for this run (prefer attached epoch_utc)
        t0 = getattr(satellite, "epoch_utc", None)
        if t0 is None:
            t0 = datetime.now(timezone.utc)
        else:
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=timezone.utc)
            else:
                t0 = t0.astimezone(timezone.utc)

        satrec_sat = getattr(satellite, "satrec", None) or satrec_from_tle(satellite.tle1, satellite.tle2)
        satrecs_deb = [
            (getattr(d, "satrec", None) or satrec_from_tle(d.tle1, d.tle2))
            for d in debris_list
        ]

        # Sample window [-lookback, +horizon]. For output/viz, keep only [0..horizon].
        times = np.arange(-lookback, float(horizon) + float(sample_dt), float(sample_dt), dtype=float)
        future_mask = (times >= 0.0) & (times <= float(horizon) + 1e-9)
        future_times = times[future_mask]

        # Precompute satellite samples
        rs_sat = np.empty((len(times), 3), dtype=float)
        vs_sat = np.empty((len(times), 3), dtype=float)
        for j, t in enumerate(times):
            st = propagate_teme_m(satrec_sat, t0 + timedelta(seconds=float(t)))
            rs_sat[j, :] = st.r_m
            vs_sat[j, :] = st.v_ms

        positions_sat = [tuple(r) for r in rs_sat[future_mask]]
        positions_debris = [[] for _ in debris_list]

        results: list[dict] = []
        per_debris: list[dict] = []

        use_chan = bool(getattr(settings, "ENGINE1_USE_CHAN", False))
        do_refine = bool(getattr(settings, "ENGINE1_TCA_REFINE", True))

        for i, debris in enumerate(debris_list):
            satrec_deb = satrecs_deb[i]

            rs_deb = np.empty((len(times), 3), dtype=float)
            vs_deb = np.empty((len(times), 3), dtype=float)
            for j, t in enumerate(times):
                st = propagate_teme_m(satrec_deb, t0 + timedelta(seconds=float(t)))
                rs_deb[j, :] = st.r_m
                vs_deb[j, :] = st.v_ms

            positions_debris[i] = [tuple(r) for r in rs_deb[future_mask]]

            rel_r = rs_deb - rs_sat
            rel_v = vs_deb - vs_sat
            dists = np.linalg.norm(rel_r, axis=1)

            idx = int(np.argmin(dists))
            tca_sample = float(times[idx])
            miss_sample = float(dists[idx])
            tca_at_boundary = (idx == 0) or (idx == len(times) - 1)

            # refine + recompute state at refined time (consistency)
            if do_refine:
                t_ref, _ = _parabolic_tca_refine(times, dists, idx)
                tca = float(t_ref)
                st_sat = propagate_teme_m(satrec_sat, t0 + timedelta(seconds=tca))
                st_deb = propagate_teme_m(satrec_deb, t0 + timedelta(seconds=tca))
                rel_pos_at_tca = st_deb.r_m - st_sat.r_m
                rel_vel_at_tca = st_deb.v_ms - st_sat.v_ms
                miss = float(np.linalg.norm(rel_pos_at_tca))
            else:
                tca = tca_sample
                miss = miss_sample
                rel_pos_at_tca = rel_r[idx]
                rel_vel_at_tca = rel_v[idx]

            rel_speed = float(np.linalg.norm(rel_vel_at_tca))

            sigma = max(1.0, sigma0 + k * max(0.0, float(tca)))
            cov_rel = np.eye(3, dtype=float) * (sigma ** 2)

            try:
                if use_chan:
                    p = float(
                        collision_probability(
                            miss_distance=miss,
                            cov_rel=cov_rel,
                            rel_pos=rel_pos_at_tca,
                            rel_vel=rel_vel_at_tca,
                            collision_radius=COLLISION_RADIUS,
                        )
                    )
                else:
                    p = float(
                        collision_probability_screening(
                            miss_distance=miss,
                            cov_rel=cov_rel,
                            collision_radius=COLLISION_RADIUS,
                        )
                    )
            except Exception:
                p = float(np.exp(-(miss ** 2) / (2.0 * sigma ** 2)))

            inside_horizon = (0.0 <= float(tca) <= float(horizon))

            # CRITICAL FIX: do not flag/escalate if closest approach is outside future horizon
            flag = bool(inside_horizon and ((p > risk_thr) or (miss < esc_thr and p > 1e-7)))

            debris_id = getattr(debris, "name", f"debris_{i}")
            per_debris.append(
                {
                    "debris_id": debris_id,
                    "tca": float(tca),
                    "miss_distance": float(miss),
                    "probability": float(p),
                    "relative_velocity": float(rel_speed),
                    "inside_horizon": bool(inside_horizon),
                    "tca_at_boundary": bool(tca_at_boundary),
                    "is_high_risk": bool(flag),
                }
            )

            future_dists = dists[future_mask]
            for step_idx, t in enumerate(future_times):
                results.append(
                    {
                        "step": int(step_idx),
                        "step_time": float(t),
                        "debris_id": debris_id,
                        "distance": float(future_dists[step_idx]),
                        "tca": float(tca),
                        "miss_distance": float(miss),
                        "inside_horizon": bool(inside_horizon),
                        "tca_at_boundary": bool(tca_at_boundary),
                        "tca_window": {
                            "lookback": float(lookback),
                            "horizon": float(horizon),
                            "sample_dt": float(sample_dt),
                        },
                        "relative_velocity": float(rel_speed),
                        "probability": float(p),
                        "risk": float(p),
                        "is_high_risk": bool(flag),
                        "frame": "TEME",
                        "method": "SGP4",
                    }
                )

        in_h = [d for d in per_debris if d.get("inside_horizon")]
        min_miss = min((d["miss_distance"] for d in in_h), default=float("inf"))
        max_prob = max((d["probability"] for d in in_h), default=0.0)
        escalate = (min_miss < esc_thr) or (max_prob > risk_thr)

        summary = {
            "min_miss_distance": float(min_miss),
            "max_probability": float(max_prob),
            "max_risk": float(max_prob),
            "escalate": bool(escalate),
            "method": "SGP4",
            "frame": "TEME",
            "sample_dt": float(sample_dt),
            "lookback": float(lookback),
            "horizon": float(horizon),
        }

        return {
            "screening": results,
            "summary": summary,
            "positions_sat": positions_sat,
            "positions_debris": positions_debris,
            "per_debris": per_debris,  # extra (non-breaking) but very useful
        }

    # -------------------------
    # main run
    # -------------------------
    def run(self, satellite, debris_list, dt: float = ENGINE1_DT, steps: int = STEPS):
        # Respect CLI / runtime override if present
        horizon_setting = float(getattr(settings, "LOOKAHEAD_SEC", dt * steps))
        horizon = min(horizon_setting, dt * steps)

        # Phase-2: if TLE exists on objects, use SGP4 screening
        if (
            bool(getattr(settings, "ENGINE1_USE_SGP4", False))
            and _has_tle(satellite)
            and all(_has_tle(d) for d in debris_list)
        ):
            print(f"[Engine1] Using horizon: {horizon}s (SGP4/TEME)")
            try:
                return self._run_sgp4_screening(satellite, debris_list, horizon=float(horizon))
            except Exception as e:
                # fail-safe fallback: CW path
                print(f"[Engine1] SGP4 screening failed, falling back to CW screening: {e}")

        print(f"[Engine1] Using horizon: {horizon}s (CW)")

        # ---- CW screening ----
        results = []
        positions_sat = []
        positions_debris = [[] for _ in debris_list]

        state_sat = State(satellite.position.copy(), satellite.velocity.copy())
        states_deb = [State(d.position.copy(), d.velocity.copy()) for d in debris_list]

        risk_thr = self._screening_risk_threshold()
        esc_thr = self._screening_escalation_threshold()

        use_chan = bool(getattr(settings, "ENGINE1_USE_CHAN", False))

        # Phase A — one-time analysis
        analysis = {}
        for i, debris in enumerate(debris_list):
            tca, miss, rel_pos_at_tca, rel_vel_at_tca = cw_time_of_closest_approach(
                satellite.position,
                satellite.velocity,
                debris.position,
                debris.velocity,
                horizon=horizon,
                n_samples=ENGINE1_CW_SAMPLES,
            )

            if tca < 0.0 or tca > horizon:
                analysis[i] = {
                    "tca": float(tca),
                    "miss": float(miss),
                    "prob": 0.0,
                    "rel_vel": float(np.linalg.norm(rel_vel_at_tca)),
                    "flag": False,
                }
                continue

            prop_t = float(max(0.0, tca))

            # Covariance propagation (best-effort)
            try:
                Ppos_sat_t, _, _ = propagate_covariance(
                    satellite.position,
                    satellite.velocity,
                    satellite.cov_pos,
                    satellite.cov_vel,
                    prop_t,
                )
            except Exception:
                Ppos_sat_t = np.atleast_2d(satellite.cov_pos)

            try:
                Ppos_deb_t, _, _ = propagate_covariance(
                    debris.position,
                    debris.velocity,
                    debris.cov_pos,
                    debris.cov_vel,
                    prop_t,
                )
            except Exception:
                Ppos_deb_t = np.atleast_2d(debris.cov_pos)

            cov_rel = Ppos_sat_t + Ppos_deb_t

            # process noise inflation (settings-driven)
            q_sigma = self._screening_process_noise_sigma(prop_t)
            cov_rel = cov_rel + (np.eye(3) * (q_sigma ** 2))

            # Probability: Chan if enabled, else fast conservative screening
            try:
                if use_chan:
                    p = float(
                        collision_probability(
                            miss_distance=miss,
                            cov_rel=cov_rel,
                            rel_pos=rel_pos_at_tca,
                            rel_vel=rel_vel_at_tca,
                            collision_radius=COLLISION_RADIUS,
                        )
                    )
                else:
                    p = float(
                        collision_probability_screening(
                            miss_distance=miss,
                            cov_rel=cov_rel,
                            collision_radius=COLLISION_RADIUS,
                        )
                    )
            except Exception:
                try:
                    sigma = np.sqrt(np.trace(cov_rel) / 3.0)
                    sigma = max(sigma, 1.0)
                    p = float(np.exp(-(miss ** 2) / (2.0 * sigma ** 2)))
                except Exception:
                    p = 0.0

            flag = (p > risk_thr) or (miss < esc_thr and p > 1e-7)

            analysis[i] = {
                "tca": float(tca),
                "miss": float(miss),
                "prob": float(p),
                "rel_vel": float(np.linalg.norm(rel_vel_at_tca)),
                "flag": bool(flag),
            }

        # Phase B — propagation for viz / distance tracking
        current_time = 0.0
        positions_sat.append(tuple(state_sat.r))
        for i in range(len(debris_list)):
            positions_debris[i].append(tuple(states_deb[i].r))

        # record at t=0
        for i, debris in enumerate(debris_list):
            inst_dist = float(np.linalg.norm(state_sat.r - states_deb[i].r))
            rec = analysis.get(i, {"tca": None, "miss": None, "prob": 0.0, "rel_vel": 0.0, "flag": False})
            results.append(
                {
                    "step": 0,
                    "step_time": current_time,
                    "debris_id": getattr(debris, "name", f"debris_{i}"),
                    "distance": inst_dist,
                    "tca": rec["tca"],
                    "miss_distance": rec["miss"],
                    "inside_horizon": (rec["tca"] is not None and 0 <= rec["tca"] <= horizon),
                    "relative_velocity": rec["rel_vel"],
                    "probability": rec["prob"],
                    "risk": rec["prob"],
                    "is_high_risk": rec["flag"],
                }
            )

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
                rec = analysis.get(i, {"tca": None, "miss": None, "prob": 0.0, "rel_vel": 0.0, "flag": False})
                results.append(
                    {
                        "step": step,
                        "step_time": current_time,
                        "debris_id": getattr(debris, "name", f"debris_{i}"),
                        "distance": inst_dist,
                        "tca": rec["tca"],
                        "miss_distance": rec["miss"],
                        "inside_horizon": (rec["tca"] is not None and 0 <= rec["tca"] <= horizon),
                        "relative_velocity": rec["rel_vel"],
                        "probability": rec["prob"],
                        "risk": rec["prob"],
                        "is_high_risk": rec["flag"],
                    }
                )

        # Summary
        try:
            min_miss = min((r["miss_distance"] for r in results if r.get("inside_horizon")), default=float("inf"))
        except Exception:
            min_miss = min((r.get("miss_distance", float("inf")) for r in results), default=float("inf"))

        max_prob = max((r.get("probability", 0.0) for r in results), default=0.0)
        max_risk = max((r.get("risk", 0.0) for r in results), default=0.0)

        escalate = (min_miss < esc_thr) or (max_prob > risk_thr)

        summary = {
            "min_miss_distance": float(min_miss),
            "max_probability": float(max_prob),
            "max_risk": float(max_risk),
            "escalate": bool(escalate),
        }

        return {
            "screening": results,
            "summary": summary,
            "positions_sat": positions_sat,
            "positions_debris": positions_debris,
        }

    def _compute_risk(self, probability, miss_distance, rel_velocity):
        return float(np.clip(probability, 0.0, 1.0))

    def get_avoidance_suggestion(self, satellite, debris, delta_v_mag=AVOIDANCE_DELTA_V):
        rel_vel = debris.velocity - satellite.velocity
        rel_pos = debris.position - satellite.position
        if np.linalg.norm(rel_vel) < 1e-6:
            rel_vel = satellite.velocity

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
        # delta_v_mag is now defined in m/s (see settings.AVOIDANCE_DELTA_V_MS)
        delta_v_mag_ms = float(delta_v_mag)
        return delta_v_mag_ms * norm
