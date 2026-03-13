from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timezone

from src.config import settings
from src.config.settings import clamp_lookahead
from src.simulation.runner import run_simulation
from src.engine.risk_filter import shortlist
from src.engine.engine2 import Engine2
from src.physics.entity import Entity


def _vec3_to_np(v: Any) -> np.ndarray:
    if v is None:
        raise ValueError("Missing vector")
    if isinstance(v, dict) and {"x", "y", "z"} <= set(v.keys()):
        return np.array([v["x"], v["y"], v["z"]], dtype=float)
    arr = np.array(v, dtype=float)
    if arr.shape == (3,):
        return arr
    if arr.shape == (2,):
        return np.array([arr[0], arr[1], 0.0], dtype=float)
    raise ValueError(f"Invalid vector: {v!r}")


def _as_entity(obj: Any) -> Entity:
    if isinstance(obj, Entity):
        return obj
    bc = float(getattr(obj, "ballistic_coeff", getattr(settings, "DEFAULT_BALLISTIC_COEFF", 50.0)))
    return Entity(
        position=getattr(obj, "position"),
        velocity=getattr(obj, "velocity"),
        ballistic_coeff=bc,
        cov_pos=getattr(obj, "cov_pos", None),
        cov_vel=getattr(obj, "cov_vel", None),
    )


def _build_entities_from_manual(payload: Dict[str, Any]) -> Tuple[Entity, List[Entity]]:
    sat_pos = _vec3_to_np(payload.get("sat_pos"))
    sat_vel = _vec3_to_np(payload.get("sat_vel"))

    sat = Entity(
        position=sat_pos,
        velocity=sat_vel,
        ballistic_coeff=float(payload.get("sat_ballistic_coeff", getattr(settings, "DEFAULT_BALLISTIC_COEFF", 50.0))),
        cov_pos=payload.get("sat_cov_pos", None),
        cov_vel=payload.get("sat_cov_vel", None),
    )
    setattr(sat, "name", payload.get("sat_name", "SAT"))

    debris_states = payload.get("debris_states") or []
    debris_list: List[Entity] = []

    for i, d in enumerate(debris_states):
        pos = _vec3_to_np(d.get("pos") or d.get("position"))
        vel = _vec3_to_np(d.get("vel") or d.get("velocity"))
        bc = float(d.get("ballistic_coeff", getattr(settings, "DEFAULT_BALLISTIC_COEFF", 50.0)))

        ent = Entity(
            position=pos,
            velocity=vel,
            ballistic_coeff=bc,
            cov_pos=d.get("cov_pos", None),
            cov_vel=d.get("cov_vel", None),
        )
        setattr(ent, "name", d.get("name", f"DEB-{i+1}"))
        debris_list.append(ent)

    return sat, debris_list


def _build_entities_from_tle(payload: Dict[str, Any]) -> Tuple[Any, List[Any]]:
    from src.data.tle_to_state import tle_to_state
    from src.data.tle_entity_factory import entity_from_tle
    from src.propagation.sgp4_propagator import satrec_from_tle

    sat_tle = payload.get("satellite_tle")
    deb_tles = payload.get("debris_tles") or []

    if not sat_tle or not sat_tle.get("tle1") or not sat_tle.get("tle2"):
        raise ValueError("satellite_tle with tle1/tle2 required for TLE mode")

    epoch0 = payload.get("epoch_utc")
    if epoch0 is None:
        epoch0 = datetime.now(timezone.utc)
    else:
        if isinstance(epoch0, str):
            epoch0 = datetime.fromisoformat(epoch0)
        if epoch0.tzinfo is None:
            epoch0 = epoch0.replace(tzinfo=timezone.utc)

    r_s, v_s = tle_to_state(sat_tle["tle1"], sat_tle["tle2"], epoch=epoch0)
    satellite = entity_from_tle(r_s, v_s)

    satellite.name = sat_tle.get("name", "SAT")
    satellite.tle1 = sat_tle["tle1"]
    satellite.tle2 = sat_tle["tle2"]
    satellite.epoch_utc = epoch0
    satellite.satrec = satrec_from_tle(sat_tle["tle1"], sat_tle["tle2"])
    satellite.frame = "TEME"

    debris_list: List[Any] = []
    for i, d in enumerate(deb_tles):
        if not d.get("tle1") or not d.get("tle2"):
            continue
        r_d, v_d = tle_to_state(d["tle1"], d["tle2"], epoch=epoch0)
        debris = entity_from_tle(r_d, v_d)

        debris.name = d.get("name", f"DEB-{i+1}")
        debris.tle1 = d["tle1"]
        debris.tle2 = d["tle2"]
        debris.epoch_utc = epoch0
        debris.satrec = satrec_from_tle(d["tle1"], d["tle2"])
        debris.frame = "TEME"
        debris_list.append(debris)

    return satellite, debris_list


def _compute_summary(screening: List[Dict[str, Any]]) -> Dict[str, Any]:
    risk_thr = float(getattr(settings, "SCREENING_RISK_THRESHOLD", getattr(settings, "RISK_THRESHOLD", 1e-6)))
    esc_thr = float(getattr(settings, "SCREENING_ESCALATION_THRESHOLD", getattr(settings, "ESCALATION_THRESHOLD", 5000.0)))

    min_miss = float("inf")
    max_prob = 0.0
    max_risk = 0.0
    high_ids = set()

    for r in screening or []:
        if r.get("inside_horizon") is False:
            continue

        md = r.get("miss_distance", r.get("distance", None))
        if md is not None:
            try:
                min_miss = min(min_miss, float(md))
            except Exception:
                pass

        p = r.get("probability", r.get("prob", r.get("risk", 0.0)))
        try:
            p = float(p)
        except Exception:
            p = 0.0

        max_prob = max(max_prob, p)
        max_risk = max(max_risk, float(r.get("risk", p) or p))

        if r.get("is_high_risk"):
            did = r.get("debris_id") or r.get("name")
            if did:
                high_ids.add(str(did))

    escalate = (min_miss < esc_thr) or (max_prob > risk_thr)

    return {
        "min_miss_distance": (float(min_miss) if np.isfinite(min_miss) else None),
        "max_probability": float(max_prob),
        "max_risk": float(max_risk),
        "high_risk_count": int(len(high_ids)),
        "high_risk_ids": sorted(list(high_ids)),
        "risk_threshold": float(risk_thr),
        "escalation_threshold": float(esc_thr),
        "escalate": bool(escalate),
    }


def _build_meta(mode: str, dt: float, steps: int, lookahead: float) -> Dict[str, Any]:
    return {
        "mode": mode,
        "dt": float(dt),
        "steps": int(steps),
        "lookahead": float(lookahead),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "settings_snapshot": {
            "DT": getattr(settings, "DT", None),
            "STEPS": getattr(settings, "STEPS", None),
            "LOOKAHEAD_SEC": getattr(settings, "LOOKAHEAD_SEC", None),
            "DANGER_RADIUS": getattr(settings, "DANGER_RADIUS", None),
            "SCREENING_ESCALATION_THRESHOLD": getattr(settings, "SCREENING_ESCALATION_THRESHOLD", None),
            "SCREENING_RISK_THRESHOLD": getattr(settings, "SCREENING_RISK_THRESHOLD", None),
        }
    }


def run_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(payload.get("mode", "auto")).lower().strip()
    dt = float(payload.get("dt", getattr(settings, "DT", 1.0)))
    lookahead_raw = float(payload.get("lookahead", getattr(settings, "LOOKAHEAD_SEC", dt * getattr(settings, "STEPS", 600))))
    lookahead = clamp_lookahead(lookahead_raw)

    setattr(settings, "LOOKAHEAD_SEC", float(lookahead))

    use_tle = ("satellite_tle" in payload)
    if use_tle:
        satellite, debris_list = _build_entities_from_tle(payload)
    else:
        satellite, debris_list = _build_entities_from_manual(payload)

    steps = int(max(1, np.ceil(float(lookahead) / float(dt))))

    # ---------- engine2 only ----------
    if mode == "engine2":
        adaptive_threshold = float(getattr(settings, "CONFIRM_ESCALATION_THRESHOLD", 5000.0))
        engine2 = Engine2(dt=dt, adaptive_threshold=adaptive_threshold, enable_drag=True)

        sat_ent = _as_entity(satellite)
        out: List[Dict[str, Any]] = []
        for d in debris_list:
            deb_ent = _as_entity(d)
            res = engine2.run(sat_ent, deb_ent, duration=float(lookahead), use_engine1_escalation=False)
            res.setdefault("debris_id", getattr(d, "name", getattr(d, "id", "debris")))
            out.append(res)

        meta = _build_meta("engine2", dt, steps, lookahead)

        return {
            "meta": meta,
            "results": out
        }

    # ---------- engine1 screening (auto/pipeline/engine1) ----------
    results, pos_sat, pos_deb = run_simulation("engine1", satellite, debris_list, dt=dt, steps=steps)

    meta = _build_meta(mode, dt, steps, lookahead)

    # Clean per-step format
    formatted_results: List[Dict[str, Any]] = []
    for r in results or []:
        step = int(r.get("step", 0))
        formatted_results.append({
            "step": step,
            "step_time": float(step * dt),
            "debris_id": r.get("debris_id") or r.get("name"),
            "distance": r.get("distance"),
            "tca": r.get("tca"),
            "miss_distance": r.get("miss_distance"),
            "inside_horizon": r.get("inside_horizon"),
            "relative_velocity": r.get("relative_velocity"),
            "probability": r.get("probability", r.get("prob")),
            "risk": r.get("risk"),
            "is_high_risk": r.get("is_high_risk"),
        })

    resp: Dict[str, Any] = {
        "meta": meta,
        "results": formatted_results,
        "summary": _compute_summary(results),

        # optional: useful for debugging / trajectory visualization (can be hidden/collapsed in UI)
        "trajectories": {
            "satellite": pos_sat,
            "debris": pos_deb,
        }
    }

    if mode in ("pipeline", "auto"):
        risky = shortlist(results, max_candidates=int(payload.get("max_candidates", 10)))
        resp["shortlist"] = risky

        if mode == "pipeline":
            adaptive_threshold = float(getattr(settings, "CONFIRM_ESCALATION_THRESHOLD", 5000.0))
            engine2 = Engine2(dt=dt, adaptive_threshold=adaptive_threshold, enable_drag=True)

            sat_ent = _as_entity(satellite)
            idset = {r.get("debris_id") for r in risky}
            confirmations: List[Dict[str, Any]] = []

            for d in debris_list:
                did = getattr(d, "name", None)
                if did in idset:
                    deb_ent = _as_entity(d)
                    res = engine2.run(sat_ent, deb_ent, duration=float(lookahead), use_engine1_escalation=False)
                    res.setdefault("debris_id", did or "debris")
                    confirmations.append(res)

            resp["confirmations"] = confirmations

    return resp