# King SpaceDeb — Space‑Debris Screening & Confirmation Engine

**Status:** Prototype → research‑grade toolchain  
**License:** GNU Affero General Public License v3.0 (AGPL‑3.0) — updated January 31, 2026  
**In one sentence:** A practical two‑stage system that quickly screens thousands of potential conjunctions with conservative filtering (Engine‑1) and then applies high‑fidelity propagation and Monte‑Carlo analysis only to the real candidates (Engine‑2), together with solid TLE fetching, CLI control, visualization, and JSON outputs.

---

## Why I built this

I wanted a collision‑assessment toolchain that actually runs on real hardware under realistic compute constraints — not just something neat in a paper. The core insight was simple:

- Propagating every object in the catalog with a high‑order integrator for days ahead is computationally insane.  
- Most potential events (~99.8% for typical scenarios) are obviously safe.  
- Only a tiny fraction deserve expensive, careful attention.

So I designed a two‑stage architecture:

1. **Engine‑1 — fast, conservative screening** (cheap physics, covariance propagation, conservative inflation).  
2. **Engine‑2 — precise, physics‑rich confirmation** (adaptive RK45, extended force models, covariance‑aware Monte‑Carlo).

This split usually reduces expensive propagations by orders of magnitude while keeping a very low probability of missed conjunctions.

---

## What’s inside

- **Engine‑1** (`src/engine/engine1.py`) — Clohessy‑Wiltshire relative motion sampling, covariance propagation, conservative noise inflation, Chan / B‑plane style probability where applicable, and Gaussian fallback.
- **Engine‑2** (`src/engine/engine2.py`) — RK45 (Dormand–Prince) adaptive integrator, composite force models (central + J2/J3/J4, optional drag, SRP, third‑body), adaptive timestep refinement, energy‑drift diagnostics, and Monte‑Carlo confirmation.
- **TLE fetcher** (`src/data/tle_fetcher.py`) — Space‑Track primary (cookie login), CelesTrak fallback, robust parser for messy responses, and TTL disk cache (`tle_cache.json`).
- **CLI & orchestration** (`src/cli.py`, `src/main.py`, `src/simulation/runner.py`) — interactive satellite/debris setup (TLE vs synthetic), mode chooser (AUTO/FAST/ACCURATE), runtime lookahead override, and save/visualize pipelines.
- **Utilities**: covariance helpers, simple Sun/Moon ephemeris, Monte‑Carlo runner, plotting & animation helpers.
- **Outputs**: timestamped JSON artifacts and optional plot/animation files.

---

## High‑level design choices

- **Separation of concerns:** screening, confirmation and avoidance suggestion are separate stages with clear inputs/outputs.  
- **Conservative first:** Engine‑1 biases toward false positives to avoid missed conjunctions.  
- **Adaptive compute:** Engine‑2 spends compute only when needed (close approaches).  
- **Covariance‑aware Monte‑Carlo:** when covariances exist, MC is the honest way to estimate tail risk.  
- **Practical I/O:** the TLE fetcher caches results (6‑hour TTL) and disables Space‑Track for the run if HTML/redirects are detected to avoid repeated failures.

---

## Engine details

### Engine‑1 (screening)
**Purpose:** quick, conservative elimination of obvious non‑threats.

Key behaviors:
- CW sampling to get a cheap estimate of TCA.  
- Covariance propagation to TCA where available; conservative inflation for unmodeled effects.  
- Collision probability via Chan / B‑plane when applicable, otherwise Gaussian fallback.  
- Emits a timeline of screening records and a summary used to escalate candidates to Engine‑2.

Typical use: run Engine‑1 across many debris objects and escalate a small subset for confirmation.

### Engine‑2 (confirmation + Monte‑Carlo)
**Purpose:** accurate, diagnostic propagation and defensible collision estimates.

Key behaviors:
- RK45 adaptive integrator with embedded error control.  
- Composite force models: central gravity + J2/J3/J4, optional drag, SRP and Sun/Moon third‑body.  
- Adaptive timestep refinement near close approach.  
- Energy‑drift checks for conservative runs.  
- `run_monte_carlo(...)` perturbs initial states using covariances (vectorized draws when possible) and returns collision/conjunction statistics.

Engine‑2 is intentionally pairwise and diagnostic‑focused.

---

## TLE fetcher & cache (practical notes)

- **Primary:** Space‑Track (cookie‑based login; prompts once per process run). If Space‑Track returns HTML/redirects or rate‑limit pages, the fetcher disables it for the run.  
- **Fallback:** CelesTrak GP endpoint.  
- **Parser:** tolerant — finds consecutive `1 ` / `2 ` TLE lines anywhere in the response and optionally uses a preceding line as the name.  
- **Cache:** `tle_cache.json` with TTL (default 6 hours). Cache entries include `timestamp`, `name`, `line1`, `line2`, and `source`.

---

## Quick start

**Requirements:** Python 3.10+ recommended. Add a `requirements.txt` (I provided one): core deps include `numpy`, `requests`, `matplotlib`, `pillow`, and optionally `scipy`.

```bash
# create & activate venv
python -m venv .venv
source .venv/bin/activate   # mac / linux
.venv/ Scripts/activate     # windows (PowerShell: run .venv/Scripts/Activate.ps1)

pip install -r requirements.txt

# run the interactive CLI
python -m src.main
```

### CLI flow (short)
1. Choose real TLEs? (y/N). If yes, provide NORAD ID(s); Space‑Track login may be prompted.  
2. Mode: AUTO (Engine‑1 → Engine‑2 optionally), FAST (Engine‑1), ACCURATE (Engine‑2).  
3. Choose lookahead horizon (seconds). CLI clamps/validates and stores it in `settings.LOOKAHEAD_SEC`.  
4. Runner writes JSON outputs to `output/` and attempts plotting/animation if visualization libs are present.

Example: to run accurate confirmation, choose `ACCURATE` when prompted.

---

## Configuration / settings
All runtime knobs live in `src/config/settings.py`. Important variables to tune:
- `DT`, `STEPS` — base timestep and number of steps (Engine‑1 horizon default = `DT * STEPS`).
- `LOOKAHEAD_SEC` — prediction horizon (CLI writes this at runtime).
- `ENGINE1_CW_SAMPLES`, `ENGINE1_LOOKAHEAD`, `ESCALATION_THRESHOLD`, `RISK_THRESHOLD`, `COLLISION_RADIUS`, `DANGER_RADIUS`, `AVOIDANCE_DELTA_V`.
- `OUTPUT_DIR` — output artifact directory.

Tune `settings.py` for experiments or override at runtime via the CLI.

---

## Outputs & artifacts
- `output/screening_results_<timestamp>.json` — Engine‑1 timeline.  
- `output/engine2_results_<timestamp>.json` — Engine‑2 confirmations.  
- `output/mc_results_<timestamp>.json` — Monte‑Carlo aggregate results.  
- Optional plots and animations (if plotting modules are available).

Timestamps use UTC format: `YYYYMMDDTHHMMSSZ`.

---

## Security / licensing / legal
- **License:** AGPL‑3.0 (added/updated on January 31, 2026). This repo is copyleft: network service redistributors must disclose source under AGPL terms.  
- **Disclaimer:** Research prototype — **NOT** certified for operational collision avoidance. Do not use for live maneuver decisions without independent validation and authorization.  
- **Data sources:** TLEs are public (CelesTrak, Space‑Track). The fetcher logs the source per cached entry.  
- **Responsible disclosure:** If you find a bug that can cause incorrect risk reporting, open an issue and mark it `security` or `safety`.

---
**Author:** Sanu Sharma  
**Date:** January 31, 2026
