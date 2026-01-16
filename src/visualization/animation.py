# src/visualization/animation.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.config.settings import OUTPUT_DIR, COLLISION_RADIUS

def animate_simulation(positions_sat, positions_debris, results):
    """
    Animate satellite & debris motion with TCA / miss-distance overlay.
    High-risk debris get predicted closest-approach markers.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build lookup by (step, debris_id)
    result_lookup = {}
    for r in results:
        key = (r["step"], r["debris_id"])
        # keep the most risky record per step/debris
        if key not in result_lookup or r["risk"] > result_lookup[key]["risk"]:
            result_lookup[key] = r

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Conjunction Screening with TCA Prediction")

    # Collect bounds
    all_x, all_y = [], []
    for p in positions_sat:
        all_x.append(p[0])
        all_y.append(p[1])
    for debris in positions_debris:
        for p in debris:
            all_x.append(p[0])
            all_y.append(p[1])

    margin = 2000
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")

    # Collision radius circle (around satellite)
    collision_circle = plt.Circle(
        (positions_sat[0][0], positions_sat[0][1]),
        COLLISION_RADIUS,
        color="red",
        fill=False,
        linestyle="--",
        alpha=0.5
    )
    ax.add_patch(collision_circle)

    # Satellite plot
    sat_plot, = ax.plot([], [], "bo", label="Satellite")

    # Debris plots
    debris_plots = [
        ax.plot([], [], "go", alpha=0.6)[0]
        for _ in positions_debris
    ]

    # TCA markers (one per debris)
    tca_markers = [
        ax.plot([], [], "rx", markersize=8, mew=2)[0]
        for _ in positions_debris
    ]

    # Text annotations
    tca_texts = [
        ax.text(0, 0, "", fontsize=8, color="red")
        for _ in positions_debris
    ]

    ax.legend(loc="upper right")

    def update(frame):
        # Satellite
        sx, sy = positions_sat[frame]
        sat_plot.set_data([sx], [sy])
        collision_circle.center = (sx, sy)

        artists = [sat_plot, collision_circle]

        for i, dp in enumerate(debris_plots):
            dx, dy = positions_debris[i][frame]
            dp.set_data([dx], [dy])

            debris_id = f"Debris-{i+1}"
            rec = result_lookup.get((frame, debris_id))

            # default visuals
            dp.set_color("green")
            dp.set_markersize(4)
            tca_markers[i].set_data([], [])
            tca_texts[i].set_text("")

            if rec and rec["is_high_risk"] and rec["tca"] is not None:
                dp.set_color("red")
                dp.set_markersize(7)

                # Predict closest-approach point (linear approx)
                # r_ca = r_now + v_rel * tca
                # We approximate using current debris velocity
                vx, vy = (
                    (positions_debris[i][frame][0] - dx),
                    (positions_debris[i][frame][1] - dy)
                )

                # Use analytical relation instead (simpler & stable):
                # project relative motion forward
                tca = rec["tca"]
                miss = rec["miss_distance"]

                # Direction from debris to satellite at closest approach
                rel_vec = np.array([sx - dx, sy - dy])
                n = np.linalg.norm(rel_vec)
                if n > 0:
                    rel_unit = rel_vec / n
                    ca_x = sx - rel_unit[0] * miss
                    ca_y = sy - rel_unit[1] * miss

                    tca_markers[i].set_data([ca_x], [ca_y])
                    tca_texts[i].set_position((ca_x, ca_y))
                    tca_texts[i].set_text(
                        f"TCA={tca:.0f}s\nMiss={miss:.1f}m"
                    )

            artists.extend([dp, tca_markers[i], tca_texts[i]])

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(positions_sat),
        interval=50,
        blit=True
    )

    plt.show()
