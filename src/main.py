# src/main.py
import os
import json
import time
import math
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from src.cli import run_cli
from src.simulation.runner import run_simulation
from src.visualization.plots import plot_probability_over_time, plot_risk_ranking
from src.visualization.animation import animate_simulation
from src.config import settings
from src.config.settings import DT, STEPS, OUTPUT_DIR, ESCALATION_THRESHOLD, DANGER_RADIUS
from src.engine.engine2 import Engine2
from src.physics.entity import Entity

# --- Setup logger ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("main")

# Ensure OUTPUT_DIR exists (some visualization modules expect it)
os.makedirs(getattr(settings, "OUTPUT_DIR", "output"), exist_ok=True)


def save_json(obj: Any, name_prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(getattr(settings, "OUTPUT_DIR", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{name_prefix}_{ts}.json"
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: repr(o))
    return str(filename)


def _should_escalate_from_record(rec: Dict[str, Any]) -> bool:
    """
    Heuristic to decide escalation from a single Engine-1 record.
    Supports different keys that various engine1 implementations may produce.
    """
    try:
        if rec.get("escalate") is True:
            return True
        if rec.get("is_high_risk") is True:
            return True
        # risk numeric threshold if provided
        if "risk" in rec:
            try:
                if float(rec["risk"]) > getattr(settings, "RISK_THRESHOLD", 0.001):
                    return True
            except Exception:
                pass
        # miss_distance numeric check
        md = rec.get("miss_distance", rec.get("distance", None))
        if md is not None:
            try:
                if float(md) <= getattr(settings, "ESCALATION_THRESHOLD", ESCALATION_THRESHOLD):
                    return True
            except Exception:
                pass
    except Exception:
        log.debug("escalation heuristic failed on record: %s", rec)
    return False


def summarize_screening(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build summary per debris_id from Engine-1 screening results.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    for r in results:
        did = r.get("debris_id") or r.get("debris") or r.get("name") or "unknown"
        if did not in summary:
            summary[did] = {
                "min_miss": float("inf"),
                "max_prob": 0.0,
                "any_escalate": False,
                "latest_record": None
            }
        try:
            md = float(r.get("miss_distance", r.get("distance", float("inf"))))
            summary[did]["min_miss"] = min(summary[did]["min_miss"], md)
        except Exception:
            pass
        try:
            p = float(r.get("probability", r.get("prob", 0.0)))
            summary[did]["max_prob"] = max(summary[did]["max_prob"], p)
        except Exception:
            pass
        if _should_escalate_from_record(r):
            summary[did]["any_escalate"] = True
        summary[did]["latest_record"] = r
    return summary


def main():
    try:
        # 1) Get inputs from CLI (CLI sets and clamps settings.LOOKAHEAD_SEC)
        satellite, debris_list, mode, lookahead = run_cli()
        log.info("Starting simulation: mode=%s, lookahead=%ss", mode, lookahead)

        # Ensure we have a canonical desired lookahead (settings may also have been set by CLI)
        desired_lookahead = float(getattr(settings, "LOOKAHEAD_SEC", lookahead))

        # 2) Ensure DT * STEPS covers the desired lookahead for the simulation runs.
        #    If not, compute a local steps_to_use that does — do NOT overwrite global STEPS permanently.
        if DT * STEPS < desired_lookahead:
            needed_steps = int(math.ceil(desired_lookahead / DT))
            log.warning(
                "Configured DT * STEPS (%.2f s) < LOOKAHEAD_SEC (%.2f s). "
                "Adjusting local steps to %d (DT=%.4f) to cover lookahead.",
                DT * STEPS, desired_lookahead, needed_steps, DT
            )
            steps_to_use = needed_steps
        else:
            steps_to_use = STEPS

        # =========================
        # MODE DISPATCH
        # =========================

        if mode == "engine2":
            # direct high-fidelity run (already correct), but use steps_to_use
            results, pos_sat, pos_deb = run_simulation(
                "engine2",
                satellite,
                debris_list,
                dt=DT,
                steps=steps_to_use
            )

        elif mode == "engine1":
            # screening only
            results, pos_sat, pos_deb = run_simulation(
                "engine1",
                satellite,
                debris_list,
                dt=DT,
                steps=steps_to_use
            )

        elif mode == "auto":
            # AUTO = Engine-1 screening FIRST
            results, pos_sat, pos_deb = run_simulation(
                "engine1",
                satellite,
                debris_list,
                dt=DT,
                steps=steps_to_use
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # =========================
        # ENGINE-2 MODE: DIRECT OUTPUT & EXIT
        # =========================
        if mode == "engine2":
            print("\n================ ENGINE-2 RESULTS ================\n")
            for r in results:
                # use .get where appropriate but assume typical engine2 keys exist
                print(f"Debris ID          : {r.get('debris_id')}")
                # safe printing with fallback if keys missing
                try:
                    print(f"Closest Time (s)   : {r['closest_time']:.2f}")
                except Exception:
                    print(f"Closest Time (s)   : {r.get('closest_time', 'N/A')}")
                try:
                    print(f"Miss Distance (m)  : {r['miss_distance']:.2f}")
                except Exception:
                    print(f"Miss Distance (m)  : {r.get('miss_distance', 'N/A')}")
                print(f"Relative Velocity : {r.get('relative_velocity', 'N/A')}")
                print(f"Collision          : {r.get('collision', False)}")
                print(f"Conjunction        : {r.get('conjunction', False)}")
                print("-" * 55)

            # Save clean Engine-2 output
            e2_file = save_json(
                {
                    "meta": {
                        "mode": "engine2",
                        "lookahead": desired_lookahead,
                        "timestamp_utc": datetime.utcnow().isoformat()
                    },
                    "results": results
                },
                "engine2_results"
            )
            log.info("Saved Engine-2 results: %s", e2_file)

            print("\n[OK] Engine-2 run complete. Plots, animation & escalation skipped by design.\n")
            return

        # 3) Save screening results & positions
        out_screen = {
            "meta": {
                "mode": mode,
                "dt": DT,
                "steps": steps_to_use,
                "lookahead": desired_lookahead,
                "timestamp_utc": datetime.utcnow().isoformat()
            },
            "results": results
        }
        screen_file = save_json(out_screen, "screening_results")
        log.info("Saved screening results: %s", screen_file)

        # 4) Produce quick plots
        try:
            plot_probability_over_time(results)
            plot_risk_ranking(results)
            log.info("Plots generated.")
        except Exception as e:
            log.warning("Plotting failed: %s", e)

        # 5) Animate (best-effort)
        try:
            animate_simulation(pos_sat, pos_deb, results)
        except Exception as e:
            log.warning("Animation failed or running headless: %s", e)

        # 6) Summarize screening and detect escalation candidates
        summary = summarize_screening(results)
        candidates = [did for did, s in summary.items() if s["any_escalate"] or s["min_miss"] <= getattr(settings, "ESCALATION_THRESHOLD", ESCALATION_THRESHOLD)]
        if not candidates:
            log.info("No candidates flagged for escalation (Engine-2). Done.")
            return

        log.info("Candidates for Engine-2 confirmation (%d): %s", len(candidates), ", ".join(candidates))

        # Ask user whether to run Engine-2 for candidates
        run_e2_input = input(f"Run Engine-2 confirmation for {len(candidates)} object(s)? [Y/n]: ").strip().lower()
        do_run_e2 = (run_e2_input == "" or run_e2_input.startswith("y"))

        if not do_run_e2:
            log.info("Skipping Engine-2. You can re-run later.")
            return

        # Build map of debris name -> object instance from original debris_list
        name_to_obj = {}
        for d in debris_list:
            # try common attributes
            name = getattr(d, "name", None) or getattr(d, "id", None) or repr(d)
            name_to_obj[name] = d

        # If names not match keys in summary, try fallback by order
        engine2 = Engine2(dt=1.0, adaptive_threshold=5000.0, enable_drag=True)

        engine2_results = {}

        # Use the canonical lookahead duration for Engine-2 confirmation runs
        duration = float(getattr(settings, "LOOKAHEAD_SEC", lookahead))

        for did in candidates:
            debris_obj = name_to_obj.get(did)
            if debris_obj is None:
                # fallback: if only one debris, use that; else skip
                log.warning("Could not find object instance for '%s' (skipping)", did)
                continue

            log.info("Running Engine-2 for %s ...", did)
            try:
                # Convert to Entities before passing to Engine-2 (runner enforces this earlier, but be safe here)
                sat_entity = Entity(position=satellite.position, velocity=satellite.velocity)
                deb_entity = Entity(position=debris_obj.position, velocity=debris_obj.velocity)

                # Use explicit duration variable derived from settings / CLI lookahead
                e2r = engine2.run(
                    sat_entity,
                    deb_entity,
                    duration=duration,
                    use_engine1_escalation=False
                )
                engine2_results[did] = e2r
                log.info("Engine-2 result for %s: miss=%.2f m, collision=%s", did, e2r.get("miss_distance", float("nan")), e2r.get("collision", False))
            except Exception as e:
                log.exception("Engine-2 failed for %s: %s", did, e)

        # 7) Optionally run Monte Carlo for candidates
        run_mc_input = input("Run Monte Carlo for flagged objects? (may be slow) [y/N]: ").strip().lower()
        do_mc = run_mc_input.startswith("y")

        mc_results = {}
        if do_mc:
            N = input("Monte Carlo sample count [default 200]: ").strip()
            try:
                Nval = int(N) if N else 200
            except Exception:
                Nval = 200
            log.info("Running Monte Carlo N=%d ...", Nval)
            for did in engine2_results.keys():
                debris_obj = name_to_obj.get(did)
                if debris_obj is None:
                    continue
                try:
                    sat_entity = Entity(position=satellite.position, velocity=satellite.velocity)
                    deb_entity = Entity(position=debris_obj.position, velocity=debris_obj.velocity)
                    # pass same explicit duration to Monte Carlo run
                    mc = engine2.run_monte_carlo(sat_entity, deb_entity, duration=duration, N=Nval)
                    mc_results[did] = mc
                    log.info("MC %s -> collision_prob=%.4f, conj_prob=%.4f", did, mc.get("collision_probability", 0.0), mc.get("conjunction_probability", 0.0))
                except Exception as e:
                    log.exception("Monte Carlo failed for %s: %s", did, e)

        # 8) Save Engine-2 and MC results
        e2_file = save_json({"engine2": engine2_results, "timestamp_utc": datetime.utcnow().isoformat()}, "engine2_results")
        log.info("Saved Engine-2 results: %s", e2_file)
        if mc_results:
            mc_file = save_json({"monte_carlo": mc_results, "timestamp_utc": datetime.utcnow().isoformat()}, "mc_results")
            log.info("Saved Monte Carlo results: %s", mc_file)

        log.info("All done. Final artifacts:\n  screening=%s\n  engine2=%s\n  monte_carlo=%s", screen_file, e2_file, mc_file if mc_results else "N/A")

    except Exception:
        log.error("Fatal exception during run:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
