from typing import List, Tuple, Any, Dict
from src.engine.engine1 import Engine1
from src.engine.engine2 import Engine2
from src.config import settings
from src.physics.entity import Entity


def _ensure_positions_debris_shape(positions_debris, debris_list):
    """
    Ensure positions_debris is a list of length len(debris_list),
    where each element is a list of (x,y,z) tuples.
    """
    n = len(debris_list)
    if not isinstance(positions_debris, list):
        return [[] for _ in range(n)]
    if len(positions_debris) == n:
        return positions_debris
    # If it's empty or wrong length, fix shape
    if len(positions_debris) == 0:
        return [[] for _ in range(n)]
    # If shorter, pad; if longer, trim
    if len(positions_debris) < n:
        return positions_debris + ([[]] * (n - len(positions_debris)))
    return positions_debris[:n]


def _normalize_engine1_output(raw, debris_list):
    """
    Engine-1 may return:
      - dict: {"screening": [...], "positions_sat": [...], "positions_debris": [...]}
      - list of dicts
      - tuple (results, positions_sat, positions_debris)
      - tuple (results, positions_debris)
      - single dict
    Return (results, positions_sat, positions_debris)
    """
    results = []
    positions_sat = []
    positions_debris = [[] for _ in debris_list]

    if isinstance(raw, dict):
        # preferred shape
        if "screening" in raw:
            results = raw.get("screening") or []
            positions_sat = raw.get("positions_sat") or []
            positions_debris = raw.get("positions_debris") or [[] for _ in debris_list]
        else:
            results = [raw]

    elif isinstance(raw, tuple):
        if len(raw) == 3:
            results = raw[0] or []
            positions_sat = raw[1] or []
            positions_debris = raw[2] or [[] for _ in debris_list]
        elif len(raw) == 2:
            results = raw[0] or []
            positions_debris = raw[1] or [[] for _ in debris_list]
        else:
            # unexpected tuple — best effort
            try:
                results = list(raw)
            except Exception:
                results = []

    elif isinstance(raw, list):
        results = raw

    else:
        try:
            results = list(raw)
        except Exception:
            results = []

    # normalize positions shape
    positions_debris = _ensure_positions_debris_shape(positions_debris, debris_list)

    # Ensure results is list[dict]
    normalized = []
    for idx, r in enumerate(results):
        if not isinstance(r, dict):
            try:
                r = dict(r)
            except Exception:
                r = {"raw": repr(r)}

        # Ensure debris_id exists
        r.setdefault("debris_id", r.get("name", r.get("debris", f"debris_{idx}")))
        normalized.append(r)

    return normalized, positions_sat, positions_debris


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


def run_simulation(
    engine_name: str,
    satellite: Any,
    debris_list: List[Any],
    dt: float,
    steps: int
) -> Tuple[List[Dict], List, List]:
    """
    Engine selector and simulation runner.
    engine1 → fast analytical screening (vectorized; expects debris_list)
    engine2 → high-fidelity physics propagation (pairwise; expects Entities)
    """
    engine_name = engine_name.lower()

    # =========================
    # ENGINE 1: FAST SCREENING
    # =========================
    if engine_name == "engine1":
        engine = Engine1()
        raw = engine.run(satellite, debris_list, dt=dt, steps=steps)
        results, positions_sat, positions_debris = _normalize_engine1_output(raw, debris_list)
        return results, positions_sat, positions_debris

    # =========================
    # ENGINE 2: HIGH FIDELITY
    # =========================
    if engine_name == "engine2":
        # config from settings (safe defaults)
        adaptive_threshold = float(getattr(settings, "CONFIRM_ESCALATION_THRESHOLD", 5000.0))

        engine = Engine2(dt=dt, adaptive_threshold=adaptive_threshold, enable_drag=True)

        results: List[Dict] = []

        # Convert Satellite -> Entity (PRESERVE cov/vel/bc if present)
        sat_entity = _as_entity(satellite)

        duration = float(getattr(settings, "LOOKAHEAD_SEC", steps * dt))

        for debris in debris_list:
            deb_entity = _as_entity(debris)

            r = engine.run(
                sat_entity,
                deb_entity,
                duration=duration,
                use_engine1_escalation=False
            )

            if not isinstance(r, dict):
                try:
                    r = dict(r)
                except Exception:
                    r = {"raw": repr(r)}

            r.setdefault("debris_id", getattr(debris, "name", getattr(debris, "id", "debris")))
            results.append(r)

        return results, [], []

    raise ValueError(f"Unknown engine: {engine_name}")
