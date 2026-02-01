import numpy as np

from src.physics.entity import Entity
from src.engine.engine2 import Engine2
from src.config.settings import EARTH_RADIUS

ALTITUDE = 400000.0       # 400 km
SAT_SPEED = 7670.0        # m/s

def main():
    print("======================================")
    print("  FORCED CONJUNCTION TEST (ENGINE-2)")
    print("======================================")

    r = EARTH_RADIUS + ALTITUDE

    # SATELLITE (REFERENCE)
    sat_pos = np.array([r, 0.0, 0.0])
    sat_vel = np.array([0.0, SAT_SPEED, 0.0])

    satellite = Entity(
        position=sat_pos,
        velocity=sat_vel
    )

    # DEBRIS (INTENTIONALLY CLOSE)
    debris_pos = np.array([r + 500.0, 0.0, 0.0])      # 500 m radial offset
    debris_vel = np.array([0.0, SAT_SPEED + 15.0, 0.0])  # slight along-track drift

    debris = Entity(
        position=debris_pos,
        velocity=debris_vel
    )

    print("\nInitial Separation:",
          np.linalg.norm(debris_pos - sat_pos), "m")

    # ENGINE-2 RUN
    engine2 = Engine2(
        dt=1.0,
        adaptive_threshold=2000.0,
        enable_drag=False,
        enable_srp=False,
        enable_third_body=False
    )

    result = engine2.run(
        satellite,
        debris,
        duration=600.0,
        use_engine1_escalation=False
    )

    print("\n========== ENGINE-2 RESULT ==========")
    for k, v in result.items():
        print(f"{k:25s}: {v}")

    print("\nTEST COMPLETE.")

if __name__ == "__main__":
    main()
