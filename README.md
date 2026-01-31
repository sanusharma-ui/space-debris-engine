# King SpaceDeb — Space‑Debris Screening & Confirmation Engine

**Status:** Prototype / hackathon → research‑grade
**License:** GNU Affero General Public License v3.0 (AGPL‑3.0) — updated: **January 31, 2026**
**Short:** A pragmatic, multi‑stage system for fast probabilistic screening of conjunctions (Engine‑1) and high‑fidelity confirmation (Engine‑2), with utilities for TLE fetching, CLI driving, visualization and Monte‑Carlo analysis.

---

## Quick elevator pitch

I built King SpaceDeb as an engineering‑first toolchain for real space‑safety problems:

* Fast, conservative filtering (Engine‑1) to cheaply remove the noisy majority.
* High‑fidelity confirmation (Engine‑2) using an adaptive RK45 integrator, extended force models and Monte‑Carlo sampling.
* Practical tooling: a robust TLE fetcher (Space‑Track primary, CelesTrak fallback) with a local disk cache; an interactive CLI; plotting and animation helpers; and JSON output for downstream use.

> **Important:** This repository is for research and development and **NOT** certified for operational collision‑avoidance. Use at your own risk.

---

## Table of contents

1. [Why multi‑stage?](#why-multi-stage)
2. [What’s included / features](#whats-included--features)
3. [Architecture & design decisions](#architecture--design-decisions)
4. [Engine details](#engine-details)

   * [Engine‑1 (screening)](#engine-1---screening)
   * [Engine‑2 (confirmation + Monte‑Carlo)](#engine-2---confirmation--monte-carlo)
5. [TLE fetcher & cache](#tle-fetcher--cache)
6. [Running (CLI & examples)](#running-cli--examples)
7. [Configuration / settings](#configuration--settings)
8. [Outputs & artifacts](#outputs--artifacts)
9. [Development notes & best practices](#development-notes--best-practices)
10. [Security / licensing / legal](#security--licensing--legal)
11. [Contributing / contact](#contributing--contact)

---

## Why multi‑stage?

Real conjunction systems must operate at scale under constrained compute budgets. Propagating every object with a heavy integrator is wasteful and slow.

**My strategy:**

* **Stage‑1 (cheap, conservative):** filter out obvious non‑threats using analytic/linearized tools (Clohessy‑Wiltshire sampling, conservative covariance inflation, Chan/B‑plane style metrics).
* **Stage‑2 (expensive, precise):** apply accurate propagation and diagnostics only to candidates flagged by Stage‑1.

This separation keeps the system practical, auditable, and scalable.

---

## What’s included / features

* `src/engine/engine1.py` — Fast probabilistic screening engine (CW sampling, covariance propagation, conservative noise inflation).
* `src/engine/engine2.py` — High‑fidelity confirmation engine (RK45 Dormand–Prince, composite force models: central + J2/J3/J4, drag, SRP, third‑body).
* `src/data/tle_fetcher.py` — Robust TLE fetcher (Space‑Track cookie login primary, CelesTrak fallback, TTL disk cache).
* `src/cli.py` — Interactive CLI for satellite/debris setup, TLE mode, and mode selection (AUTO/FAST/ACCURATE).
* `src/simulation/runner.py` — Orchestration glue between engines and system I/O.
* `src/main.py` — Full run orchestration, result saving, plotting and animation hooks.
* Utilities: covariance helpers, simple Sun/Moon ephemeris, Monte‑Carlo runner, plotting & animation modules.
* Output: timestamped JSON artifacts, optional plots and animation files.

---

## Architecture & design decisions (high level)

* **Separation of concerns:** Screening, confirmation and avoidance suggestion are split into distinct components with clear responsibilities and outputs.
* **Conservative-first:** Engine‑1 is tuned to prefer false positives over false negatives (covariance inflation, conservative heuristics). This helps avoid missed conjunctions.
* **Adaptive compute:** Engine‑2 refines timesteps only when and where precision is needed (close approaches).
* **Covariance‑aware Monte‑Carlo:** When covariance is available, MC sampling gives a defensible estimate of tail risk.
* **Practical I/O:** TLE fetcher caches results with a TTL (default 6 hours) and disables Space‑Track for a run if non‑TLE HTML or redirects are detected, to avoid repeated broken attempts.

---

## Engine details

### Engine‑1 — Fast screening

**Goals:** conservative elimination, speed, physics awareness.

Key behavior:

* Analytic CW sampling to estimate Time‑of‑Closest‑Approach (TCA) quickly.
* Covariance propagation to TCA where possible, with conservative fallback heuristics.
* Screening noise inflation to account for unmodeled effects (drag, SRP, execution uncertainty).
* Probability computation via a `collision_probability` helper (Chan / B‑plane where applicable) with Gaussian fallback.
* Produces a `results` timeline, animation positions, and a summary object used for escalation decisions.

Typical workflow: run Engine‑1 over many candidate debris objects and escalate a compact subset to Engine‑2.

---

### Engine‑2 — High‑fidelity confirmation

**Goals:** accuracy, diagnostics, energy‑drift reporting, Monte‑Carlo confirmation.

Key behavior:

* RK45 (Dormand–Prince) adaptive integrator with embedded error control.
* Composite force model: Newtonian central gravity + J2/J3/J4, optional atmospheric drag, SRP, and third‑body (Sun/Moon).
* Adaptive timestep refinement when relative distance is below a configurable threshold.
* Energy‑drift reporting when propagation is conservative (no drag/SRP/third‑body) to detect integrator problems.
* `run_monte_carlo(...)` for covariance‑aware sampling; supports vectorized draws and Cholesky reuse when available.
* Engine‑2 is designed for pairwise confirmation (satellite vs. debris) and to return defensible diagnostics: miss distance, time of closest approach, relative velocity, collision/conjunction flags, and optional energy drift.

---

## TLE fetcher & cache

`src/data/tle_fetcher.py` highlights (practical behavior):

* **Primary:** Space‑Track (cookie login). The process prompts for credentials once per run. If Space‑Track responds with HTML (login, redirect or rate limit), the fetcher disables Space‑Track for that run and falls back to CelesTrak.
* **Fallback:** CelesTrak GP endpoint (`gp.php`).
* **Robust parser:** finds consecutive TLE lines (`1 ` / `2 `) anywhere in the response and optionally uses a preceding line as the name.
* **Disk cache:** `tle_cache.json` with TTL (default 6 hours). Cache entries include `timestamp`, `name`, `line1`, `line2` and `source`.
* **Runtime flow:** cache → Space‑Track (if allowed) → CelesTrak → clear error if both fail.

**Important:** TLE sources are public, but **do not commit cache files**. Treat `tle_cache.json` as a runtime artifact and add it to `.gitignore`.

---

## Running (quick start)

**Requirements:** Python 3.10+ recommended.
Suggested dependencies (add to `requirements.txt`): `numpy`, `requests`, `matplotlib`, `pillow` (if animations), etc.

```bash
# create & activate venv
python -m venv .venv
source .venv/bin/activate   # mac / linux
.venv\Scripts\activate     # windows

pip install -r requirements.txt

# run interactive CLI
python -m src.main
```

### CLI flow summary

1. Choose real TLEs? (y/N). If yes, provide NORAD ID(s); Space‑Track login may be prompted.
2. Choose mode:

   * `AUTO`: Engine‑1 screening (recommended) → optional Engine‑2 confirmation
   * `FAST`: Engine‑1 only
   * `ACCURATE`: Engine‑2 only
3. Choose lookahead horizon (seconds). The CLI clamps/validates the value and stores it in `settings.LOOKAHEAD_SEC` for the run.
4. Runner saves JSON artifacts in `output/`, and tries to create plots/animations if plotting modules are available.

**Example:**

```bash
python -m src.main   # choose ACCURATE in the CLI
```

---

## Configuration / settings

All runtime knobs live in `src/config/settings.py`. Important variables you'll likely tune:

* `DT`, `STEPS` — base timestep & number of steps (Engine‑1 horizon = DT * STEPS unless overridden)
* `LOOKAHEAD_SEC` — lookahead horizon (CLI writes this into settings at runtime)
* `ENGINE1_DT`, `ENGINE1_CW_SAMPLES`, `ENGINE1_LOOKAHEAD`, `ESCALATION_THRESHOLD`, `RISK_THRESHOLD`, `COLLISION_RADIUS`, `DANGER_RADIUS`, `AVOIDANCE_DELTA_V`
* `OUTPUT_DIR` — output artifact directory

Adjust `settings.py` for experiments or set values programmatically at runtime (the CLI does this for `LOOKAHEAD_SEC`).

---

## Outputs & artifacts

* `output/screening_results_<timestamp>.json` — Engine‑1 screening timeline and records.
* `output/engine2_results_<timestamp>.json` — pairwise Engine‑2 confirmations.
* `output/mc_results_<timestamp>.json` — Monte‑Carlo aggregate statistics.
* Optional plots and animation files if plotting/animation modules succeed.

Timestamps use UTC format: `YYYYMMDDTHHMMSSZ`.

---


## Security / licensing / legal

* **License:** AGPL‑3.0 (added/updated on **January 31, 2026**). This repository is copyleft; network service redistributors must disclose source under AGPL terms.
* **Disclaimer:** This is a research prototype. **NOT** certified for operational collision avoidance. Do not use for live maneuver decisions without rigorous engineering verification and authorization.
* **Data sources:** TLEs are fetched from public sources (CelesTrak, Space‑Track). The code logs the `source` for cached TLEs.
* **Responsible disclosure:** If you find a bug that could cause incorrect risk reporting, please open an issue and tag it `security` or `safety`.

---

## Contributing

* Open a clear PR with a short description, a reproducible test, and a note on engineering trade‑offs.
* Keep numerical experiments reproducible by saving seed values or configuration snapshots to `output/` (do **not** commit `output/`).
* If adding force models or numerical methods, include unit tests and reference comparisons where feasible.

---
## Final note (straight talk)

I built this stack to be engineering‑forward and defensible. Share the design and reasoning; keep operational knobs controlled until you’re ready. That balance builds credibility — and credibility is what opens doors for partnerships, funding and hardware later.