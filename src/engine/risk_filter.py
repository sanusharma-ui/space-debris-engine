def _as_float(x, default=float("inf")):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def shortlist(screening_results, max_candidates=10):
    """
    Pick best candidate per debris_id (avoid duplicates from time-series records),
    ignore records outside horizon when inside_horizon key exists.
    """
    candidates = [
        r for r in (screening_results or [])
        if bool(r.get("is_high_risk", False))
        and bool(r.get("inside_horizon", True))
    ]

    best_by_id = {}
    for r in candidates:
        did = r.get("debris_id") or r.get("name") or "unknown"
        md = _as_float(r.get("miss_distance", r.get("distance", None)), default=float("inf"))
        p = _as_float(r.get("probability", r.get("prob", 0.0)), default=0.0)

        cur = best_by_id.get(did)
        key = (md, -p)  # prefer smaller miss, then higher probability
        if cur is None or key < cur[0]:
            best_by_id[did] = (key, r)

    out = [v[1] for v in best_by_id.values()]
    out.sort(
        key=lambda r: _as_float(r.get("miss_distance", r.get("distance", None)), default=float("inf"))
    )
    return out[: int(max_candidates)]
