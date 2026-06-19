import numpy as np
from datetime import datetime, timezone

AU = 1.495978707e11
J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def seconds_since_j2000(epoch_utc=None) -> float:
    """
    Convert a datetime-like epoch to seconds since J2000.
    If no epoch is provided, use current UTC time.
    """
    if epoch_utc is None:
        epoch_utc = datetime.now(timezone.utc)
    if isinstance(epoch_utc, str):
        epoch_utc = datetime.fromisoformat(epoch_utc)
    if epoch_utc.tzinfo is None:
        epoch_utc = epoch_utc.replace(tzinfo=timezone.utc)
    epoch_utc = epoch_utc.astimezone(timezone.utc)
    return float((epoch_utc - J2000).total_seconds())


def _julian_centuries_from_j2000(t: float) -> float:
    return float(t) / (36525.0 * 86400.0)


def _wrap_degrees(angle: float) -> float:
    return angle % 360.0


def sun_position(t: float) -> np.ndarray:
    """
    Low-cost analytical Sun position in an Earth-centered inertial frame.
    Input t is seconds since J2000. Accuracy is suitable for SRP and
    third-body perturbation screening, not precision orbit determination.
    """
    T = _julian_centuries_from_j2000(t)
    mean_long = _wrap_degrees(280.460 + 36000.771 * T)
    mean_anomaly = np.deg2rad(_wrap_degrees(357.5291092 + 35999.05034 * T))
    ecliptic_long = np.deg2rad(
        _wrap_degrees(
            mean_long
            + 1.914666471 * np.sin(mean_anomaly)
            + 0.019994643 * np.sin(2.0 * mean_anomaly)
        )
    )
    obliquity = np.deg2rad(23.439291 - 0.0130042 * T)
    radius_au = (
        1.000140612
        - 0.016708617 * np.cos(mean_anomaly)
        - 0.000139589 * np.cos(2.0 * mean_anomaly)
    )
    radius = radius_au * AU

    return radius * np.array(
        [
            np.cos(ecliptic_long),
            np.cos(obliquity) * np.sin(ecliptic_long),
            np.sin(obliquity) * np.sin(ecliptic_long),
        ],
        dtype=float,
    )


def moon_position(t: float) -> np.ndarray:
    """
    Compact analytical Moon position in ECI (meters).
    Includes the dominant lunar longitude, anomaly, node and inclination terms.
    """
    days = float(t) / 86400.0
    T = _julian_centuries_from_j2000(t)

    L = np.deg2rad(_wrap_degrees(218.316 + 13.176396 * days))
    M_moon = np.deg2rad(_wrap_degrees(134.963 + 13.064993 * days))
    F = np.deg2rad(_wrap_degrees(93.272 + 13.229350 * days))

    lon = L + np.deg2rad(6.289) * np.sin(M_moon)
    lat = np.deg2rad(5.128) * np.sin(F)
    distance = (385001.0 - 20905.0 * np.cos(M_moon)) * 1000.0
    obliquity = np.deg2rad(23.439291 - 0.0130042 * T)

    x_ecl = distance * np.cos(lat) * np.cos(lon)
    y_ecl = distance * np.cos(lat) * np.sin(lon)
    z_ecl = distance * np.sin(lat)

    return np.array(
        [
            x_ecl,
            y_ecl * np.cos(obliquity) - z_ecl * np.sin(obliquity),
            y_ecl * np.sin(obliquity) + z_ecl * np.cos(obliquity),
        ],
        dtype=float,
    )
