Markdown# Space Debris Collision Engine

Physics-based close-approach detection & collision prediction system for spacecraft and space debris.

**Two-stage architecture** — fast screening + high-fidelity confirmation — built for reproducibility, auditability, and scientific validation.

Designed as a strong, well-documented entry for space-tech hackathons and further research.

---

## Key Features

- **Engine-1 (Fast Screening)**  
  RK4 orbit propagation + Clohessy–Wiltshire equations for quick Time-of-Closest-Approach (TCA) estimation  
  Very fast → suitable for screening thousands of debris objects

- **Engine-2 (High-Fidelity Confirmation)**  
  Full numerical integration (RK4) with J₂ perturbation  
  Optional atmospheric drag model  
  Monte-Carlo uncertainty propagation support

- **Validation**  
  Forced very-close collision test case (detects collision within seconds)  
  → `src/test_forced_collision.py`

- Clean modular structure, type hints, configuration-driven, testable

---

## Project Structure
space-debris-engine/
├── src/
│   ├── engine/
│   │   ├── engine1.py        # Fast screening (CW + RK4)
│   │   └── engine2.py        # High-fidelity confirmation
│   ├── physics/              # Propagators, force models, CW equations
│   ├── models/               # Satellite & Debris classes
│   ├── config/               # Settings, constants, clamping functions
│   ├── cli.py                # Interactive & test input interface
│   ├── main.py               # Entry point — runs simulation
│   └── test_forced_collision.py  # Validation: guaranteed collision detection
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
text---

## Quick Start (Windows / Linux / macOS)

# 2. Virtual environment (strongly recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
Recommended requirements.txt:
txtnumpy>=1.24
scipy>=1.10
matplotlib>=3.7    # optional — for future visualization
joblib>=1.2        # optional — parallel processing
numba>=0.57        # optional — significant speedup if used

Run the Validation Test (most important for judges)
Bash# Extremely close debris → should detect collision in < 1 second
python -m src.test_forced_collision
Expected output:
Collision detected! (with miss distance ≈ 0–few meters)

Run Full Interactive Simulation
Bashpython -m src.main
Features:

Choose altitude, number of debris objects
Allow small inclinations (true 3D orbits)
Select mode: AUTO (recommended), FAST, ACCURATE
Custom lookahead time


Philosophy & Goals

Physics-first approach over black-box ML
Deterministic & reproducible results
Easy to extend (add drag, SRP, third-body, etc.)
Designed to be verifiable and auditable

Perfect foundation for real mission analysis tools, student research, or space situational awareness prototypes.