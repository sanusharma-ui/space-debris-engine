from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
from sgp4.api import Satrec, jday


@dataclass(frozen=True)
class Sgp4State:
    r_m: np.ndarray   # position in meters (TEME)
    v_ms: np.ndarray  # velocity in m/s (TEME)


def satrec_from_tle(tle1: str, tle2: str) -> Satrec:
    return Satrec.twoline2rv(tle1, tle2)


def propagate_teme_m(sat: Satrec, t_utc: datetime) -> Sgp4State:
    """
    Propagate using SGP4 to time t_utc (timezone-aware UTC).
    Returns TEME state in meters and m/s (sgp4 returns km and km/s).
    """
    if t_utc.tzinfo is None:
        t_utc = t_utc.replace(tzinfo=timezone.utc)
    else:
        t_utc = t_utc.astimezone(timezone.utc)

    jd, fr = jday(
        t_utc.year, t_utc.month, t_utc.day,
        t_utc.hour, t_utc.minute,
        t_utc.second + t_utc.microsecond * 1e-6,
    )

    e, r_km, v_kms = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 error code={e}")

    r_m = np.array(r_km, dtype=float) * 1000.0
    v_ms = np.array(v_kms, dtype=float) * 1000.0
    return Sgp4State(r_m=r_m, v_ms=v_ms)


def propagate_entity_teme(entity, t_offset_s: float) -> Sgp4State:
    """
    Convenience wrapper: propagate an Entity-like object that has .tle1/.tle2 and optional .epoch_utc.
    If .satrec is already attached, uses it; else builds satrec from tle.
    """
    tle1 = getattr(entity, "tle1", None)
    tle2 = getattr(entity, "tle2", None)
    if not tle1 or not tle2:
        raise RuntimeError("Entity missing tle1/tle2 for SGP4 propagation")

    sat = getattr(entity, "satrec", None)
    if sat is None:
        sat = satrec_from_tle(tle1, tle2)

    epoch = getattr(entity, "epoch_utc", None)
    if epoch is None:
        epoch = datetime.now(timezone.utc)
    elif epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    else:
        epoch = epoch.astimezone(timezone.utc)

    return propagate_teme_m(sat, epoch + timedelta(seconds=float(t_offset_s)))
