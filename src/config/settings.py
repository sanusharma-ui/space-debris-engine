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

# Atmosphere (simple demo model)
RHO0 = 2.4e-12
H0 = 400_000.0
SCALE_HEIGHT = 60_000.0


def atmospheric_density(altitude: float) -> float:
    if altitude <= 0.0 or altitude >= 1_000_000.0:
        return 0.0
    return float(RHO0 * np.exp(-(altitude - H0) / SCALE_HEIGHT))


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

# Monte Carlo
MC_DEFAULT_N = 300


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
