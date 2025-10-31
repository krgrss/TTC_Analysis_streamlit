"""
Network routes — GTFS stops for coverage map

Purpose:
- Provide a typed FastAPI endpoint to retrieve TTC stops with coordinates for the frontend map coverage view.

Security:
- Returns sanitized strings only; no HTML. Reads from local cleaned data via ttc_analysis utilities.
"""
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

try:
    from ttc_analysis.delay_ingestion import load_station_coordinates
except Exception:  # pragma: no cover - fallback when module layout differs
    load_station_coordinates = None  # type: ignore


router = APIRouter(prefix="/network", tags=["network"])


class Stop(BaseModel):
    id: str
    name: str
    lng: float
    lat: float
    routes: int = 0


def _mk_id(name: str, lng: float, lat: float) -> str:
    base = name.strip() if name else f"{lng:.5f},{lat:.5f}"
    return base.lower().replace(" ", "-")


@router.get("/stops", response_model=List[Stop])
def get_stops() -> List[Stop]:
    """Return station/stop coordinates for the TTC network coverage map.

    Fallback to an empty list if the coordinates source is unavailable.
    """
    if load_station_coordinates is None:
        return []
    df = load_station_coordinates()
    out: List[Stop] = []
    if df is None or df.empty:  # type: ignore[attr-defined]
        return out
    for row in df.itertuples(index=False):  # type: ignore[attr-defined]
        name = str(getattr(row, "location", ""))
        lat = float(getattr(row, "lat", 0.0))
        lng = float(getattr(row, "lon", 0.0))
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
            continue
        out.append(
            Stop(
                id=_mk_id(name, lng, lat),
                name=name,
                lng=lng,
                lat=lat,
                routes=0,
            )
        )
    return out

