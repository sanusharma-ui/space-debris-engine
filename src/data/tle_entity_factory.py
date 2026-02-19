import numpy as np
from sgp4.api import Satrec
from datetime import datetime, timezone
from src.physics.entity import Entity

def entity_from_tle(r, v, pos_sigma=100.0, vel_sigma=0.1, tle1=None, tle2=None, epoch_utc=None):
    """
    Create Entity with reasonable covariance from TLE.
    Also attaches:
      - tle1, tle2
      - satrec (SGP4 object)
      - epoch_utc (timezone-aware UTC datetime)
    """
    cov_pos = np.eye(3) * float(pos_sigma) ** 2
    cov_vel = np.eye(3) * float(vel_sigma) ** 2

    ent = Entity(
        position=np.array(r, dtype=float),
        velocity=np.array(v, dtype=float),
        cov_pos=cov_pos,
        cov_vel=cov_vel
    )

    # Attach SGP4 metadata (non-breaking; Entity is still same)
    if epoch_utc is None:
        epoch_utc = datetime.now(timezone.utc)
    else:
        if epoch_utc.tzinfo is None:
            epoch_utc = epoch_utc.replace(tzinfo=timezone.utc)
        else:
            epoch_utc = epoch_utc.astimezone(timezone.utc)

    setattr(ent, "epoch_utc", epoch_utc)

    if tle1 and tle2:
        setattr(ent, "tle1", tle1)
        setattr(ent, "tle2", tle2)
        try:
            setattr(ent, "satrec", Satrec.twoline2rv(tle1, tle2))
        except Exception:
            # if satrec fails, leave absent -> Engine1 will fallback
            pass

    return ent
