# src/cli.py
import math
import numpy as np
from src.models.satellite import Satellite
from src.models.debris import Debris
from src.config import settings

# bring in useful defaults from settings for CLI defaults
from src.config.settings import (
    MAX_DEBRIS,
    DEFAULT_ALTITUDE,
    EARTH_RADIUS,
    SATELLITE_SPEED,
    SPREAD,
    REL_VEL_SPREAD,
    clamp_lookahead,
)

def _to_3d_array(v):
    """Ensure the input is a numpy array of shape (3,).

    - Scalars will be promoted to (scalar, 0.0, 0.0)
    - 2-element lists/arrays will get a trailing 0.0 appended
    - 3-element lists/arrays become numpy arrays
    - Anything else will raise ValueError
    """
    arr = np.array(v, dtype=float)
    if arr.ndim == 0:
        return np.array([float(arr), 0.0, 0.0])
    if arr.shape == (2,):
        return np.append(arr, 0.0)
    if arr.shape == (3,):
        return arr
    raise ValueError(f"Cannot coerce {v!r} to 3D vector")


def get_float(prompt, default=None):
    """
    Safe float input with optional default. Non-interactive (EOF) returns default.
    """
    while True:
        try:
            user = input(prompt)
        except EOFError:
            return float(default) if default is not None else None
        if user.strip() == "" and default is not None:
            return float(default)
        try:
            return float(user)
        except (ValueError, TypeError):
            print("❌ Please enter a valid number.")


def get_int(prompt, default=None, min_val=None, max_val=None):
    """
    Safe integer input with limits. Non-interactive (EOF) returns default.
    """
    while True:
        try:
            user = input(prompt)
        except EOFError:
            return int(default) if default is not None else None
        if user.strip() == "" and default is not None:
            return int(default)
        try:
            val = int(user)
            if min_val is not None and val < min_val:
                raise ValueError
            if max_val is not None and val > max_val:
                raise ValueError
            return val
        except (ValueError, TypeError):
            print("❌ Invalid integer input.")


def choose_mode():
    """
    Choose simulation mode.
      1 -> AUTO (Engine-1 screening + Engine-2 confirmation) [recommended]
      2 -> FAST (Engine-1 only)
      3 -> ACCURATE (Engine-2 only)
    """
    print("\n⚙️  Simulation Mode")
    print("  1) AUTO (Recommended) — Screening + Confirmation")
    print("  2) FAST — Engine-1 only")
    print("  3) ACCURATE — Engine-2 only")

    choice = input("Select mode [1]: ").strip()

    if choice == "2":
        return "engine1"
    if choice == "3":
        return "engine2"
    return "auto"


def create_satellite():
    print("\n🛰️ Satellite Configuration")

    altitude = get_float(
        f"Altitude above Earth (m) [default {DEFAULT_ALTITUDE}]: ",
        default=DEFAULT_ALTITUDE
    )

    r = EARTH_RADIUS + altitude

    # Put the satellite on the x-axis and include z-axis (0.0)
    position = np.array([r, 0.0, 0.0], dtype=float)

    # Velocity: assume prograde circular velocity along +y at given magnitude
    velocity = np.array([0.0, float(SATELLITE_SPEED), 0.0], dtype=float)

    print(f"✔ Satellite initialized at r = {r:.1f} m (altitude {altitude:.1f} m)")

    return Satellite(position=position, velocity=velocity)


def _sample_debris_orbit(base_radius, sat_speed, spread=SPREAD, rel_vel_spread=REL_VEL_SPREAD,
                          allow_inclination=False):
    """
    Generate a realistic 3D debris initial condition near the reference circular orbit.

    Strategy:
      - pick a random true anomaly theta
      - radial perturbation ~ Uniform(-spread, spread)
      - small z-offset (optionally non-zero if allow_inclination)
      - velocity vector is set approximately perpendicular to radius (prograde)
        with small random perturbations in all three components.
    Returns (pos_3d, vel_3d)
    """
    theta = float(np.random.uniform(0.0, 2 * math.pi))

    # radial/altitude perturbations
    radial = float(np.random.uniform(-spread, spread))
    z_perturb = float(np.random.uniform(-spread / 10.0, spread / 10.0)) if allow_inclination else 0.0

    r = base_radius + radial

    # position in orbital plane (x,y) and small z
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = z_perturb

    pos = np.array([x, y, z], dtype=float)

    # velocity magnitude roughly sat_speed (can be perturbed)
    v_mag = float(sat_speed + np.random.uniform(-rel_vel_spread, rel_vel_spread))

    # velocity direction: perpendicular to radius in the orbital plane (prograde)
    vx = -v_mag * math.sin(theta) + float(np.random.uniform(-rel_vel_spread, rel_vel_spread))
    vy = v_mag * math.cos(theta) + float(np.random.uniform(-rel_vel_spread, rel_vel_spread))
    vz = float(np.random.uniform(-rel_vel_spread / 10.0, rel_vel_spread / 10.0)) if allow_inclination else 0.0

    vel = np.array([vx, vy, vz], dtype=float)

    return pos, vel


def create_debris_list():
    print("\n☄️ Debris Configuration")

    n = get_int(
        f"Number of debris (1–{MAX_DEBRIS}) [default 3]: ",
        default=3,
        min_val=1,
        max_val=MAX_DEBRIS
    )

    # Ask whether to allow small inclinations for debris (makes them truly 3D)
    try:
        incl_choice = input("Allow small random inclinations for debris? (y/N): ").strip().lower()
    except EOFError:
        incl_choice = "n"
    allow_inclination = incl_choice == "y"

    debris_list = []

    # reference radius = Earth's radius + default altitude (but we might want satellite altitude later)
    base_alt = getattr(settings, "DEFAULT_ALTITUDE", DEFAULT_ALTITUDE)
    base_radius = EARTH_RADIUS + float(base_alt)

    for i in range(n):
        print(f"\nDebris-{i+1}")

        try:
            name = input("Name (press Enter for auto): ").strip()
        except EOFError:
            name = ""
        if name == "":
            name = f"Debris-{i+1}"

        pos, vel = _sample_debris_orbit(base_radius, SATELLITE_SPEED,
                                        spread=SPREAD, rel_vel_spread=REL_VEL_SPREAD,
                                        allow_inclination=allow_inclination)

        # safety: coerce to 3D (this will also raise cleanly if something is wrong)
        pos = _to_3d_array(pos)
        vel = _to_3d_array(vel)

        debris = Debris(position=pos, velocity=vel, name=name)
        debris_list.append(debris)

        # print full 3D vector for clarity
        print(f"✔ {name} initialized at pos ~[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] m")

    return debris_list


def ask_lookahead(default=None):
    """
    Ask user for a lookahead horizon in seconds.
    If user presses Enter, returns default (clamped).
    """
    if default is None:
        # fallback sensible default: use settings LOOKAHEAD_SEC if present
        default = getattr(settings, "LOOKAHEAD_SEC", None)
        if default is None:
            # try derive from DT * STEPS if available
            default = getattr(settings, "DT", 1.0) * getattr(settings, "STEPS", 3600)

    val = get_float(f"\nLookahead horizon in seconds [default {int(default)}]: ", default=default)
    return float(val)


def run_cli():
    print("======================================")
    print("  SPACE DEBRIS COLLISION ENGINE (CLI)  ")
    print("======================================")

    satellite = create_satellite()
    debris_list = create_debris_list()

    # Choose simulation mode (engine1 / engine2)
    mode = choose_mode()

    # Ask lookahead horizon (seconds), clamp it, and set into settings module for runtime use
    lookahead_raw = ask_lookahead(default=getattr(settings, "LOOKAHEAD_SEC", None))
    lookahead = clamp_lookahead(lookahead_raw)
    # Assign into settings so engine reads the runtime value (single place of truth)
    setattr(settings, "LOOKAHEAD_SEC", float(lookahead))

    print("\n✅ CLI input complete.")
    print(f"→ Mode: {mode}")
    print(f"→ Lookahead horizon set to: {int(lookahead)} seconds")

    return satellite, debris_list, mode, float(lookahead)


