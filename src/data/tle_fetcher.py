"""
Robust TLE fetcher (Space-Track primary, CelesTrak fallback, disk cache).

Phase-1 hardening upgrades:
 - Treat Space-Track 204 as "no TLE" (do NOT disable provider globally)
 - Provider disable is per-NORAD (not global) to avoid hurting other IDs
 - Cache only SUCCESS payloads (no caching failures)
 - Robust HTML/error detection for CelesTrak
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from getpass import getpass
from pathlib import Path
from typing import Optional, Tuple

import requests

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# -----------------------
# Config
# -----------------------
CACHE_FILE = Path("tle_cache.json")  # local project cache
CACHE_TTL = timedelta(hours=6)

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

SPACE_TRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACE_TRACK_TLE_URL = (
    "https://www.space-track.org/basicspacedata/query/"
    "class/gp_latest/"
    "NORAD_CAT_ID/{norad}/"
    "orderby/EPOCH desc/limit/1/format/tle"
)

# -----------------------
# Runtime globals
# -----------------------
_SPACE_TRACK_SESSION: Optional[requests.Session] = None

# Space-Track should NOT be globally disabled for one bad NORAD.
# Instead maintain per-NORAD skip within this process run.
_SPACE_TRACK_SKIP_NORAD: set[int] = set()

# -----------------------
# Cache helpers
# -----------------------
def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        txt = CACHE_FILE.read_text(encoding="utf-8")
        data = json.loads(txt)
        if isinstance(data, dict):
            return data
        logger.warning("TLE cache content not a dict; starting fresh.")
        return {}
    except Exception:
        logger.warning("TLE cache file unreadable or corrupt, starting fresh.")
        return {}

def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write TLE cache: %s", e)

def _cache_get(cache: dict, norad_str: str, now: datetime) -> Optional[Tuple[str, str, str]]:
    """
    Return cached (name, line1, line2) if valid & within TTL else None.
    """
    if norad_str not in cache:
        return None
    entry = cache.get(norad_str, {})
    try:
        ts = datetime.fromisoformat(entry["timestamp"])
        if now - ts >= CACHE_TTL:
            return None
        name = entry["name"]
        l1 = entry["line1"]
        l2 = entry["line2"]
        if not (isinstance(name, str) and isinstance(l1, str) and isinstance(l2, str)):
            return None
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            return None
        logger.info("TLE cache hit for NORAD %s", norad_str)
        return name, l1, l2
    except Exception:
        return None

def _cache_put_success(cache: dict, norad_str: str, now: datetime, name: str, l1: str, l2: str, source: str) -> None:
    """
    Cache only successful TLE fetches.
    """
    cache[norad_str] = {
        "timestamp": now.isoformat(),
        "name": name,
        "line1": l1,
        "line2": l2,
        "source": source,
        "status": "ok",
    }
    _save_cache(cache)

# -----------------------
# Small helpers
# -----------------------
def _looks_like_html(text: str) -> bool:
    t = (text or "").lower()
    return ("<html" in t) or ("<!doctype html" in t) or ("</html>" in t)

def _looks_like_celestrak_error(text: str) -> bool:
    t = (text or "").lower()
    needles = [
        "no gp data", "no data", "not found", "invalid", "error",
        "forbidden", "access denied", "cloudflare", "attention required",
    ]
    return any(n in t for n in needles)

# -----------------------
# Space-Track login
# -----------------------
def _get_space_track_session() -> requests.Session:
    """
    Authenticated Space-Track session. Prompts once per run.
    """
    global _SPACE_TRACK_SESSION

    if _SPACE_TRACK_SESSION is not None:
        return _SPACE_TRACK_SESSION

    print("\n=== Space-Track login required (one-time for this run) ===")
    print("If you don't have an account, register at https://www.space-track.org")
    username = input("Space-Track username/email: ").strip()
    password = getpass("Space-Track password: ")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "SpaceDebrisEngine/1.0 (+https://example.invalid)",
        "Accept": "text/plain, */*; q=0.01",
    })

    try:
        resp = session.post(
            SPACE_TRACK_LOGIN_URL,
            data={"identity": username, "password": password},
            timeout=20,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Space-Track login network error: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"Space-Track login HTTP error: {resp.status_code}")

    if not session.cookies:
        raise RuntimeError("Space-Track login failed (no session cookies received)")

    logger.info("Space-Track login succeeded (cookies present).")
    _SPACE_TRACK_SESSION = session
    return session

# -----------------------
# Robust TLE parser
# -----------------------
def _parse_tle(text: str, norad_id: int) -> Tuple[str, str, str]:
    """
    Find consecutive '1 ' and '2 ' lines anywhere in the response.
    If a name line exists immediately before the '1' line, use it as name.
    Returns (name, line1, line2).
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for i in range(len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            return f"NORAD-{norad_id}", lines[i], lines[i + 1]
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = lines[i]
            return name, lines[i + 1], lines[i + 2]
    raise ValueError("Invalid TLE response (no '1 '/'2 ' pair found)")

# -----------------------
# Space-Track fetch (per-NORAD safe)
# -----------------------
def _fetch_space_track(norad_id: int) -> Tuple[str, str, str]:
    """
    Fetch latest TLE from Space-Track for norad_id.

    IMPORTANT:
    - 204 => no content/TLE for that NORAD => mark skip for this NORAD only
    - HTML => likely auth/redirect/rate-limit => skip this NORAD for this run (do NOT global-disable)
    """
    if norad_id in _SPACE_TRACK_SKIP_NORAD:
        raise RuntimeError(f"Space-Track skipped for NORAD {norad_id} in this run")

    session = _get_space_track_session()
    url = SPACE_TRACK_TLE_URL.format(norad=norad_id)

    try:
        resp = session.get(url, timeout=20)
    except requests.RequestException as e:
        # transient network errors shouldn't poison this NORAD permanently
        raise RuntimeError(f"Space-Track GET failed: {e}") from e

    # Key fix: Space-Track returns 204 for "no data"
    if resp.status_code == 204:
        _SPACE_TRACK_SKIP_NORAD.add(norad_id)
        raise RuntimeError(f"Space-Track: no TLE available (204) for NORAD {norad_id}")

    if resp.status_code != 200:
        # don't skip forever; but if it's a consistent non-200 for this NORAD,
        # we skip it for this run to avoid repeated failing calls.
        _SPACE_TRACK_SKIP_NORAD.add(norad_id)
        raise RuntimeError(f"Space-Track GET HTTP error: {resp.status_code}")

    ct = (resp.headers.get("Content-Type", "") or "").lower()
    text = resp.text or ""

    if "text/html" in ct or _looks_like_html(text):
        _SPACE_TRACK_SKIP_NORAD.add(norad_id)
        raise RuntimeError(
            f"Space-Track returned HTML for NORAD {norad_id} (redirect/login/rate-limit?). Skipping ST for this NORAD."
        )

    if "1 " not in text or "2 " not in text:
        _SPACE_TRACK_SKIP_NORAD.add(norad_id)
        raise RuntimeError(f"Space-Track did not return TLE lines for NORAD {norad_id}")

    return _parse_tle(text, norad_id)

# -----------------------
# CelesTrak fetch (retry + html/error detection)
# -----------------------
def _fetch_celestrak(norad_id: int) -> Tuple[str, str, str]:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/plain, text/html;q=0.9, */*;q=0.8",
    })
    params = {"CATNR": str(norad_id), "FORMAT": "TLE"}

    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = session.get(CELESTRAK_GP_URL, params=params, timeout=15)
            resp.raise_for_status()

            ct = (resp.headers.get("Content-Type", "") or "").lower()
            text = resp.text or ""

            if "text/html" in ct or _looks_like_html(text) or _looks_like_celestrak_error(text):
                raise RuntimeError("CelesTrak returned non-TLE content (HTML/error page)")

            return _parse_tle(text, norad_id)

        except requests.HTTPError as he:
            last_exc = he
            status = he.response.status_code if he.response is not None else None
            if status == 404:
                raise RuntimeError(f"CelesTrak: NORAD {norad_id} not found (404)") from he

        except (requests.RequestException, RuntimeError, ValueError) as e:
            last_exc = e

        # small backoff
        try:
            import time
            time.sleep(0.6 * attempt)
        except Exception:
            pass

    raise RuntimeError(f"CelesTrak failed after retries for NORAD {norad_id}: {last_exc}") from last_exc

# -----------------------
# Public API
# -----------------------
@lru_cache(maxsize=512)
def fetch_tle(norad_id: int) -> Tuple[str, str, str]:
    """
    Public fetcher:
      - checks disk cache (TTL)
      - tries Space-Track (per-NORAD safe)
      - falls back to CelesTrak
    Raises RuntimeError if both fail.
    """
    norad_str = str(norad_id)
    now = datetime.now(timezone.utc)

    cache = _load_cache()
    cached = _cache_get(cache, norad_str, now)
    if cached is not None:
        return cached

    last_exc: Optional[Exception] = None

    # 1) Space-Track
    try:
        name, l1, l2 = _fetch_space_track(norad_id)
        _cache_put_success(cache, norad_str, now, name, l1, l2, source="Space-Track")
        logger.info("Fetched TLE for NORAD %s from Space-Track", norad_str)
        return name, l1, l2
    except Exception as st_e:
        last_exc = st_e
        logger.warning("Space-Track failed: %s", st_e)

    # 2) CelesTrak fallback
    try:
        name, l1, l2 = _fetch_celestrak(norad_id)
        _cache_put_success(cache, norad_str, now, name, l1, l2, source="CelesTrak")
        logger.info("Fetched TLE for NORAD %s from CelesTrak", norad_str)
        return name, l1, l2
    except Exception as cel_e:
        last_exc = cel_e
        logger.warning("CelesTrak failed: %s", cel_e)

    raise RuntimeError(f"TLE unavailable for NORAD {norad_id}") from last_exc

# -----------------------
# Quick test
# -----------------------
if __name__ == "__main__":
    try:
        print(fetch_tle(25544))  # ISS
    except Exception as e:
        print("Test failed:", e)
