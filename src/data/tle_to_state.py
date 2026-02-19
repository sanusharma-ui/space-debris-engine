import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

def tle_to_state(tle1, tle2, epoch=None):
    """
    Convert TLE to TEME position & velocity (meters, m/s).
    Note: SGP4 returns TEME, not GCRF/ECI. We keep it consistent (Phase-2).
    """
    sat = Satrec.twoline2rv(tle1, tle2)

    if epoch is None:
        now = datetime.now(timezone.utc)
    else:
        # accept naive -> treat as UTC
        now = epoch
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

    jd, fr = jday(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second + now.microsecond * 1e-6
    )

    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 propagation failed (code={e})")

    r = np.array(r, dtype=float) * 1000.0  # km → m
    v = np.array(v, dtype=float) * 1000.0  # km/s → m/s
    return r, v
