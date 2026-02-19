from src.engine.engine1 import Engine1
from src.engine.engine2 import Engine2
from src.engine.risk_filter import shortlist
from src.physics.entity import Entity
from src.config import settings


def _as_entity(obj):
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


def run_pipeline(satellite, debris_list, lookahead):
    engine1 = Engine1()
    engine2 = Engine2()

    # Stage 1: Fast screening
    screening = engine1.run(
        satellite=satellite,
        debris_list=debris_list,
        dt=10.0,
        steps=int(lookahead / 10.0),
    )

    # Stage 2: shortlist risky debris
    risky = shortlist(screening["screening"])

    # Stage 3: High-fidelity confirmation
    results = []
    for d in debris_list:
        for r in risky:
            if getattr(d, "name", None) == r.get("debris_id"):
                res = engine2.run(
                    satellite=_as_entity(satellite),
                    debris=_as_entity(d),
                    duration=lookahead,
                    use_engine1_escalation=False,
                )
                results.append(res)

    return results
