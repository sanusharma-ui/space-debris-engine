import numpy as np

from src.models.satellite import Satellite
from src.models.debris import Debris
from src.engine.engine1 import Engine1
from src.config import settings

# -----------------------------
# CONSTANTS
# -----------------------------
EARTH_RADIUS = 6378137.0     # meters
ALTITUDE = 400000.0          # 400 km
SATELLITE_SPEED = 7660.0     # m/s

# -----------------------------
# FORCED COLLISION TEST
# -----------------------------
def main():
    print("======================================")
    print("  FORCED COLLISION HONESTY TEST")
    print("======================================")

    # Satellite
    sat_pos = np.array([EARTH_RADIUS + ALTITUDE, 0.0, 0.0])
    sat_vel = np.array([0.0, SATELLITE_SPEED, 0.0])

    satellite = Satellite(
        position=sat_pos,
        velocity=sat_vel
    )

    # Debris: ONLY 2 METERS AWAY, SAME VELOCITY
    debris_pos = sat_pos + np.array([2.0, 0.0, 0.0])
    debris_vel = sat_vel.copy()

    debris = Debris(
        position=debris_pos,
        velocity=debris_vel,
        name="Forced-Collision"
    )

    debris_list = [debris]

    print("\nSatellite:")
    print(" Position:", satellite.position)
    print(" Velocity:", satellite.velocity)

    print("\nDebris:")
    print(" Position:", debris.position)
    print(" Velocity:", debris.velocity)
    print(" Separation:", np.linalg.norm(debris.position - satellite.position), "m")

    # Lookahead
    lookahead = 60.0
    settings.LOOKAHEAD_SEC = lookahead

    # -----------------------------
    # RUN ENGINE-1
    # -----------------------------
    engine1 = Engine1()

    result = engine1.run(
        satellite=satellite,
        debris_list=debris_list,
        dt=2.0,
        steps=int(lookahead / 2.0)
    )

    print("\n========== ENGINE-1 SUMMARY ==========")
    print(result["summary"])

    print("\n========== LAST SCREENING RECORD ==========")
    print(result["screening"][-1])

    print("\nTEST COMPLETE.")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
