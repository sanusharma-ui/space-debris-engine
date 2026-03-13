from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.simulation.service import run_from_payload

router = APIRouter()


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class TLEObj(BaseModel):
    name: str = "OBJ"
    tle1: str
    tle2: str


class DebrisState(BaseModel):
    name: str = "debris"
    pos: Vec3
    vel: Vec3
    ballistic_coeff: Optional[float] = None
    cov_pos: Optional[List[List[float]]] = None
    cov_vel: Optional[List[List[float]]] = None


class RunRequest(BaseModel):
    mode: str = Field(default="auto", description="engine1|engine2|auto|pipeline")
    lookahead: float = 600.0
    dt: float = 1.0

    # Manual mode
    sat_pos: Optional[Vec3] = None
    sat_vel: Optional[Vec3] = None
    debris_states: Optional[List[DebrisState]] = None

    # TLE mode
    satellite_tle: Optional[TLEObj] = None
    debris_tles: Optional[List[TLEObj]] = None
    epoch_utc: Optional[str] = None

    max_candidates: int = 10


class RealTLERequest(BaseModel):
    mode: str = Field(default="auto", description="engine1|engine2|auto|pipeline")
    lookahead: float = 900.0
    dt: float = 2.0
    satellite_norad_id: int
    max_candidates: int = 10

    # backend file or uploaded text
    debris_source: str = "backend"   # "backend" | "upload"
    debris_file: str = "debris_ids.txt"
    debris_ids_text: Optional[str] = None


def _vec3_to_list(v: Any):
    if v is None:
        return None
    if isinstance(v, dict):
        return [v.get("x"), v.get("y"), v.get("z")]
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return [v.x, v.y, v.z]
    return v


def _prune_none(obj: Any) -> Any:
    """
    Remove None keys recursively.
    Prevents service from switching mode because satellite_tle exists with null.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[k] = _prune_none(v)
        return out
    if isinstance(obj, list):
        return [_prune_none(x) for x in obj if x is not None]
    return obj


def _load_debris_ids(filepath: str) -> List[int]:
    path = Path(filepath)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Debris file not found: {filepath}")

    ids: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                ids.append(int(s))
            except ValueError:
                continue

    if not ids:
        raise HTTPException(status_code=400, detail=f"No valid NORAD IDs found in {filepath}")

    return ids


def _parse_debris_ids_text(text: str) -> List[int]:
    ids: List[int] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            ids.append(int(s))
        except ValueError:
            continue

    if not ids:
        raise HTTPException(
            status_code=400,
            detail="No valid NORAD IDs found in uploaded/pasted text"
        )
    return ids


@router.get("/presets/manual-leo")
def preset_manual_leo() -> Dict[str, Any]:
    return {
        "mode": "auto",
        "lookahead": 900.0,
        "dt": 2.0,
        "max_candidates": 10,
        "sat_pos": {"x": 6778137.0, "y": 0.0, "z": 0.0},
        "sat_vel": {"x": 0.0, "y": 7660.0, "z": 0.0},
        "debris_states": [
            {
                "name": "DEB-1",
                "pos": {"x": 6778137.0 + 150.0, "y": 800.0, "z": 0.0},
                "vel": {"x": -0.2, "y": 7660.0 - 0.6, "z": 0.0},
            }
        ],
    }


@router.post("/")
def run(req: RunRequest) -> Dict[str, Any]:
    payload = req.model_dump(mode="json")

    # normalize Vec3 -> [x,y,z] for service.py
    if payload.get("sat_pos") is not None:
        payload["sat_pos"] = _vec3_to_list(payload["sat_pos"])
    if payload.get("sat_vel") is not None:
        payload["sat_vel"] = _vec3_to_list(payload["sat_vel"])

    if payload.get("debris_states"):
        for d in payload["debris_states"]:
            if d.get("pos") is not None:
                d["pos"] = _vec3_to_list(d["pos"])
            if d.get("vel") is not None:
                d["vel"] = _vec3_to_list(d["vel"])

    payload = _prune_none(payload)

    if "satellite_tle" in payload:
        st = payload["satellite_tle"]
        if not st.get("tle1") or not st.get("tle2"):
            raise HTTPException(status_code=422, detail="satellite_tle.tle1 and satellite_tle.tle2 are required")

    return run_from_payload(payload)


@router.post("/real-tle")
def run_real_tle(req: RealTLERequest) -> Dict[str, Any]:
    from src.data.tle_fetcher import fetch_tle

    epoch0 = datetime.now(timezone.utc)

    if req.debris_source == "upload":
        if not req.debris_ids_text or not req.debris_ids_text.strip():
            raise HTTPException(
                status_code=422,
                detail="debris_ids_text is required when debris_source='upload'"
            )
        debris_ids = _parse_debris_ids_text(req.debris_ids_text)
    else:
        debris_ids = _load_debris_ids(req.debris_file)

    sat_name, sat_tle1, sat_tle2 = fetch_tle(req.satellite_norad_id)

    debris_tles = []
    skipped = []

    for did in debris_ids:
        try:
            name_d, tle1_d, tle2_d = fetch_tle(did)
            debris_tles.append(
                {
                    "name": name_d or f"NORAD-{did}",
                    "tle1": tle1_d,
                    "tle2": tle2_d,
                }
            )
        except Exception as e:
            skipped.append({"norad_id": did, "error": str(e)})

    payload = {
        "mode": req.mode,
        "lookahead": req.lookahead,
        "dt": req.dt,
        "epoch_utc": epoch0.isoformat(),
        "max_candidates": req.max_candidates,
        "satellite_tle": {
            "name": sat_name or f"NORAD-{req.satellite_norad_id}",
            "tle1": sat_tle1,
            "tle2": sat_tle2,
        },
        "debris_tles": debris_tles,
    }

    result = run_from_payload(payload)
    result["real_tle_meta"] = {
        "satellite_norad_id": req.satellite_norad_id,
        "debris_source": req.debris_source,
        "debris_file": req.debris_file if req.debris_source == "backend" else None,
        "loaded_debris_count": len(debris_tles),
        "skipped_debris": skipped,
    }
    return result