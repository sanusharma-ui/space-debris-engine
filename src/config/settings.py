"""
Project settings (constants + small helpers).
Units: meters (m), seconds (s), kilograms (kg), meters/second (m/s).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Run
DEFAULT_RANDOM_SEED: Optional[int] = None
RUN_ID_PREFIX = "run"
VALIDATE_ON_IMPORT = False

# Earth
GM = 3.986004418e14
RE = 6378137.0
EARTH_RADIUS = RE
J2 = 1.08262668e-3
J3 = -2.53215306e-6
J4 = -1.61962159137e-6
OMEGA_EARTH = 7.2921150e-5  # rad/s (Earth rotation rate)

# Geometry
SATELLITE_RADIUS = 5.0
DEBRIS_RADIUS = 1.0
COLLISION_RADIUS = SATELLITE_RADIUS + DEBRIS_RADIUS
DANGER_RADIUS = 1000.0

# Thresholds (profiles)
SCREENING_ESCALATION_THRESHOLD = 10_000.0
SCREENING_RISK_THRESHOLD = 1e-6

CONFIRM_ESCALATION_THRESHOLD = 5_000.0
CONFIRM_RISK_THRESHOLD = 1e-3

# Backward-compatible names
ESCALATION_THRESHOLD = SCREENING_ESCALATION_THRESHOLD
RISK_THRESHOLD = SCREENING_RISK_THRESHOLD

# Uncertainty defaults
SIGMA = 50.0
DEFAULT_POS_STD = 100.0

# Screening process-noise growth (Engine-1 CW path)
SCREENING_POS_GROWTH_M_PER_S = 0.01
SCREENING_POS_GROWTH_FLOOR_M = 1.0
SCREENING_POS_GROWTH_CAP_M = 250.0

# Simulation
DT = 1.0
STEPS = 600
LOOKAHEAD_SEC = 78.0

LOOKAHEAD_MIN = 1.0
LOOKAHEAD_MAX = 3600.0

SCREENING_LOOKAHEAD_MAX = LOOKAHEAD_MAX
CONFIRM_LOOKAHEAD_MAX = LOOKAHEAD_MAX

# CLI/random debris generator
MAX_DEBRIS = 500
SPREAD = 5000.0
REL_VEL_SPREAD = 20.0

DEFAULT_ALTITUDE = 400_000.0
SATELLITE_SPEED = 7500.0

# ✅ AVOIDANCE: now explicitly m/s (5 cm/s)
AVOIDANCE_DELTA_V_MS = 0.05  # m/s (5 cm/s) — sensible demo avoidance burn
# Backward-compatible name (now explicitly in m/s)
AVOIDANCE_DELTA_V = AVOIDANCE_DELTA_V_MS

# Atmosphere
RHO0 = 2.4e-12
H0 = 400_000.0
SCALE_HEIGHT = 60_000.0

# Static density table used for fast log-linear interpolation. Values are
# representative SI densities for quiet thermospheric screening; for operational
# work this should be replaced by NRLMSISE-00/JB2008 with space-weather inputs.
ATMOSPHERE_DENSITY_TABLE = (
    (0.0, 1.225),
    (25_000.0, 3.899e-2),
    (30_000.0, 1.774e-2),
    (40_000.0, 3.972e-3),
    (50_000.0, 1.057e-3),
    (60_000.0, 3.206e-4),
    (70_000.0, 8.770e-5),
    (80_000.0, 1.905e-5),
    (90_000.0, 3.396e-6),
    (100_000.0, 5.297e-7),
    (110_000.0, 9.661e-8),
    (120_000.0, 2.438e-8),
    (130_000.0, 8.484e-9),
    (140_000.0, 3.845e-9),
    (150_000.0, 2.070e-9),
    (180_000.0, 5.464e-10),
    (200_000.0, 2.789e-10),
    (250_000.0, 7.248e-11),
    (300_000.0, 2.418e-11),
    (350_000.0, 9.518e-12),
    (400_000.0, 3.725e-12),
    (450_000.0, 1.585e-12),
    (500_000.0, 6.967e-13),
    (600_000.0, 1.454e-13),
    (700_000.0, 3.614e-14),
    (800_000.0, 1.170e-14),
    (900_000.0, 5.245e-15),
    (1_000_000.0, 3.019e-15),
)


def atmospheric_density(altitude: float) -> float:
    altitude = float(altitude)
    if altitude < 0.0 or altitude > 1_000_000.0:
        return 0.0

    table = ATMOSPHERE_DENSITY_TABLE
    if altitude <= table[0][0]:
        return float(table[0][1])

    for (h0, rho0), (h1, rho1) in zip(table[:-1], table[1:]):
        if h0 <= altitude <= h1:
            if rho0 <= 0.0 or rho1 <= 0.0:
                return 0.0
            w = (altitude - h0) / (h1 - h0)
            log_rho = (1.0 - w) * np.log(rho0) + w * np.log(rho1)
            return float(np.exp(log_rho))

    return float(table[-1][1])


DEFAULT_BALLISTIC_COEFF = 50.0

# Engine-1 tuning
ENGINE1_DT = 2.0
ENGINE1_LOOKAHEAD = 2000.0
ENGINE1_CW_SAMPLES = 120
ENGINE1_USE_CHAN = False  # keep false for screening
ENGINE1_LOOKBACK_SEC = 300.0

# Phase-2: SGP4 screening (TEME) if TLE exists
ENGINE1_USE_SGP4 = True
ENGINE1_SGP4_SAMPLE_DT = 10.0
ENGINE1_TCA_REFINE = True
ENGINE1_COV_GROWTH_RATE = 0.02   # sigma(t) = sigma0 + k*t
ENGINE1_SIGMA0 = 100.0
ENGINE1_FRAME = "TEME"

# Engine-2 tuning
ENGINE2_DT = 1.0
ENGINE2_USE_CHAN = True
ENGINE2_RK45_RTOL = 1e-9
ENGINE2_RK45_ATOL = 1e-11
ENGINE2_DT_MIN = 1e-4
ENGINE2_DT_MAX = 10.0
ENGINE2_NEAR_APPROACH_SUBDIVISIONS = 10
ENGINE2_MAX_MACRO_STEPS = 2_000_000
ENGINE2_ENABLE_SRP_DEFAULT = True
ENGINE2_ENABLE_THIRD_BODY_DEFAULT = True

# Monte Carlo
MC_DEFAULT_N = 300
MC_MAX_N = 5000
MC_RANDOM_SEED: Optional[int] = None


def clamp_lookahead(val: Optional[float]) -> float:
    out = float(LOOKAHEAD_SEC if val is None else val)
    return max(float(LOOKAHEAD_MIN), min(float(LOOKAHEAD_MAX), out))


def get_collision_radius(
    sat_radius: Optional[float] = None,
    deb_radius: Optional[float] = None,
) -> float:
    sr = SATELLITE_RADIUS if sat_radius is None else float(sat_radius)
    dr = DEBRIS_RADIUS if deb_radius is None else float(deb_radius)
    return sr + dr


def validate_settings() -> None:
    if DT <= 0:
        raise ValueError("DT must be > 0")
    if STEPS <= 0:
        raise ValueError("STEPS must be > 0")
    if LOOKAHEAD_SEC <= 0:
        raise ValueError("LOOKAHEAD_SEC must be > 0")
    if LOOKAHEAD_MIN <= 0:
        raise ValueError("LOOKAHEAD_MIN must be > 0")
    if LOOKAHEAD_MAX < LOOKAHEAD_MIN:
        raise ValueError("LOOKAHEAD_MAX must be >= LOOKAHEAD_MIN")
    if COLLISION_RADIUS <= 0:
        raise ValueError("COLLISION_RADIUS must be > 0")
    if DEFAULT_POS_STD <= 0:
        raise ValueError("DEFAULT_POS_STD must be > 0")
    if DEFAULT_BALLISTIC_COEFF <= 0:
        raise ValueError("DEFAULT_BALLISTIC_COEFF must be > 0")

    if SCREENING_RISK_THRESHOLD <= 0:
        raise ValueError("SCREENING_RISK_THRESHOLD must be > 0")
    if CONFIRM_RISK_THRESHOLD <= 0:
        raise ValueError("CONFIRM_RISK_THRESHOLD must be > 0")
    if SCREENING_ESCALATION_THRESHOLD <= 0:
        raise ValueError("SCREENING_ESCALATION_THRESHOLD must be > 0")
    if CONFIRM_ESCALATION_THRESHOLD <= 0:
        raise ValueError("CONFIRM_ESCALATION_THRESHOLD must be > 0")

    if SCREENING_POS_GROWTH_M_PER_S < 0:
        raise ValueError("SCREENING_POS_GROWTH_M_PER_S must be >= 0")
    if SCREENING_POS_GROWTH_FLOOR_M <= 0:
        raise ValueError("SCREENING_POS_GROWTH_FLOOR_M must be > 0")
    if SCREENING_POS_GROWTH_CAP_M < SCREENING_POS_GROWTH_FLOOR_M:
        raise ValueError("SCREENING_POS_GROWTH_CAP_M must be >= SCREENING_POS_GROWTH_FLOOR_M")


if VALIDATE_ON_IMPORT:
    validate_settings()
