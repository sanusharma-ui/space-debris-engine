from src.engine.engine1 import Engine1
from src.engine.engine2 import Engine2
from src.engine.risk_filter import shortlist

def run_pipeline(satellite, debris_list, lookahead):
    engine1 = Engine1()
    engine2 = Engine2()

    # Stage 1: Fast screening
    screening = engine1.run(
        satellite=satellite,
        debris_list=debris_list,
        dt=10.0,
        steps=int(lookahead / 10.0)
    )

    # Stage 2: shortlist risky debris
    risky = shortlist(screening["screening"])

    # Stage 3: High-fidelity confirmation
    results = []
    for d in debris_list:
        for r in risky:
            if d.name == r["debris_id"]:
                res = engine2.run(
                    satellite=satellite,
                    debris=d,
                    duration=lookahead,
                    use_engine1_escalation=False
                )
                results.append(res)

    return results
