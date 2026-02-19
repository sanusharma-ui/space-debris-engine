import numpy as np


def time_of_closest_approach(r1, v1, r2, v2):
    r1 = np.array(r1, dtype=float)
    v1 = np.array(v1, dtype=float)
    r2 = np.array(r2, dtype=float)
    v2 = np.array(v2, dtype=float)

    dr = r1 - r2
    dv = v1 - v2

    dv2 = float(np.dot(dv, dv))
    if dv2 == 0.0:
        return 0.0, float(np.linalg.norm(dr))

    tca = -float(np.dot(dr, dv)) / dv2
    miss_vec = dr + dv * tca
    miss = float(np.linalg.norm(miss_vec))
    return float(tca), float(miss)
