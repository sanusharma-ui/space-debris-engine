import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

matplotlib.use('TkAgg')          

from src.config.settings import OUTPUT_DIR, COLLISION_RADIUS


def _xy(p):
    try:
        return float(p[0]), float(p[1])
    except Exception:
        return 0.0, 0.0


def animate_simulation(positions_sat, positions_debris, results):
    """
    Animate satellite + debris trajectories with TCA/high-risk highlighting.
    Uses block=True + explicit close to prevent hanging in VS Code.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if isinstance(positions_debris, dict):
        debris_ids = list(positions_debris.keys())
        debris_trajs = [positions_debris[k] for k in debris_ids]
    else:
        debris_trajs = positions_debris
        debris_ids = [f"Debris-{i+1}" for i in range(len(debris_trajs))]

    result_lookup = {}
    for r in results:
        key = (r.get("step"), r.get("debris_id"))
        if key[0] is None or key[1] is None:
            continue
        if key not in result_lookup or r.get("risk", 0.0) > result_lookup[key].get("risk", 0.0):
            result_lookup[key] = r

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title("Conjunction Screening – TCA & Risk Visualization")

    # Collect bounds for nice view
    all_x, all_y = [], []
    for p in positions_sat:
        x, y = _xy(p)
        all_x.append(x)
        all_y.append(y)

    for traj in debris_trajs:
        for p in traj:
            x, y = _xy(p)
            all_x.append(x)
            all_y.append(y)

    if not all_x:
        print("[WARNING] No trajectory points available for animation.")
        plt.close(fig)
        return

    margin = 2500
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    sx0, sy0 = _xy(positions_sat[0])
    collision_circle = plt.Circle((sx0, sy0), COLLISION_RADIUS, color="red",
                                  fill=False, linestyle="--", alpha=0.5, linewidth=1.5)
    ax.add_patch(collision_circle)

    # Artists
    sat_plot, = ax.plot([], [], "bo", ms=8, label="Satellite")
    debris_plots = [ax.plot([], [], "go", ms=5, alpha=0.7)[0] for _ in debris_trajs]
    tca_markers = [ax.plot([], [], "rx", ms=10, mew=2.5)[0] for _ in debris_trajs]
    tca_texts   = [ax.text(0, 0, "", fontsize=9, color="darkred", ha="center", va="bottom") for _ in debris_trajs]

    ax.legend(loc="upper right", fontsize=10)

    frames = min(len(positions_sat), *(len(t) for t in debris_trajs)) if debris_trajs else len(positions_sat)

    def update(frame):
        sx, sy = _xy(positions_sat[frame])
        sat_plot.set_data([sx], [sy])
        collision_circle.center = (sx, sy)

        artists = [sat_plot, collision_circle]

        for i, dp in enumerate(debris_plots):
            dx, dy = _xy(debris_trajs[i][frame])
            dp.set_data([dx], [dy])

            did = debris_ids[i]
            rec = result_lookup.get((frame, did))

            dp.set_color("green")
            dp.set_markersize(5)
            tca_markers[i].set_data([], [])
            tca_texts[i].set_text("")

            if rec and rec.get("is_high_risk") and rec.get("tca") is not None:
                dp.set_color("red")
                dp.set_markersize(9)

                tca = float(rec["tca"])
                miss = float(rec.get("miss_distance", 0.0))

                rel_vec = np.array([sx - dx, sy - dy])
                n = np.linalg.norm(rel_vec)
                if n > 0:
                    rel_unit = rel_vec / n
                    ca_x = sx - rel_unit[0] * miss
                    ca_y = sy - rel_unit[1] * miss

                    tca_markers[i].set_data([ca_x], [ca_y])
                    tca_texts[i].set_position((ca_x + 300, ca_y + 300))  
                    tca_texts[i].set_text(f"TCA {tca:.0f}s\nMiss {miss:.0f}m")

            artists.extend([dp, tca_markers[i], tca_texts[i]])

        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

    try:
        plt.tight_layout()
        plt.show(block=True)
    finally:
        plt.close('all')
        print("Animation window closed → program exiting cleanly")

    