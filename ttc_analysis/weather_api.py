# ttc_analysis/weather_api.py
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests as r

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Timezone -----------------------------------------------------------------
TZ_NAME = "America/Toronto"
TZ = ZoneInfo(TZ_NAME)

# --- Open-Meteo archive endpoint ---------------------------------------------
BASE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "surface_pressure",
    "weathercode",
]
HOURLY = ",".join(HOURLY_VARS)

# Archive allows <= yesterday
ARCHIVE_MAX_DATE = date.today() - timedelta(days=1)

# Canonical (renamed) schema we return everywhere
CANON_COLS = [
    "ts",
    "temp_c",
    "rel_humidity",
    "precip_mm",
    "rain_mm",
    "snow_cm",
    "wind_kph",
    "pressure_hpa",
    "wmo_code",
    "is_rain",
    "is_snow",
]


# =============================================================================
# Helpers
# =============================================================================


def _to_local_ts(series: pd.Series, tz_name: str = TZ_NAME) -> pd.Series:
    """
    Parse strings to timezone-aware timestamps in `tz_name`, robust to DST gaps.
    - Try parsing as UTC-aware first (strings with 'Z' or offsets).
    - For naive timestamps, localize with DST-safe rules:
        nonexistent='shift_forward' (spring-forward gap)
        ambiguous='NaT'            (fall-back ambiguity)
    """
    ts = pd.to_datetime(series, errors="coerce", utc=True)

    if ts.isna().any():
        raw = pd.to_datetime(series, errors="coerce")
        safe_local = raw.dt.tz_localize(
            tz_name,
            nonexistent="shift_forward",
            ambiguous="NaT",
        )
        # convert localized to UTC to combine with any UTC-parsed values
        safe_utc = safe_local.dt.tz_convert("UTC")
        ts = ts.where(~ts.isna(), safe_utc)

    return ts.dt.tz_convert(tz=tz_name)


def _clip_to_archive(start_like, end_like) -> tuple[date | None, date | None]:
    """
    Clip (start, end) to the archive window (<= ARCHIVE_MAX_DATE).
    Returns (None, None) if empty after clipping.
    """
    s = pd.to_datetime(start_like, errors="coerce")
    e = pd.to_datetime(end_like, errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return None, None
    s = s.date()
    e = min(e.date(), ARCHIVE_MAX_DATE)
    if s > e:
        return None, None
    return s, e


def _month_iter(start: date, end: date):
    """Yield (month_start, month_end) covering the inclusive range [start, end]."""
    d = date(start.year, start.month, 1)
    while d <= end:
        if d.month == 12:
            me = date(d.year, 12, 31)
        else:
            me = date(d.year, d.month + 1, 1) - timedelta(days=1)
        yield max(d, start), min(me, end)
        d = me + timedelta(days=1)


def _coord_tag(lat: float, lon: float) -> str:
    """Make coordinates safe for filenames (e.g., 43p653_m79p383)."""

    def fmt(x):
        s = f"{x:.3f}"
        return s.replace(".", "p").replace("-", "m")

    return f"{fmt(lat)}_{fmt(lon)}"


def _cache_path(lat: float, lon: float, ym: str, ext: str) -> Path:
    return CACHE_DIR / f"openmeteo_{_coord_tag(lat, lon)}_{ym}.{ext}"


def _write_cached(df: pd.DataFrame, path_parquet: Path, path_csv: Path):
    # Always try parquet; fall back to CSV if parquet engine is unavailable.
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        df.to_csv(path_csv, index=False)


def _read_cached(path_parquet: Path, path_csv: Path) -> pd.DataFrame | None:
    # Try parquet first (preserves types); fallback to CSV.
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet)
        except Exception:
            pass
    if path_csv.exists():
        try:
            df = pd.read_csv(path_csv)
            # Re-hydrate ts as tz-aware
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(
                    TZ_NAME
                )
            return df
        except Exception:
            pass
    return None


def _empty_canon_df() -> pd.DataFrame:
    """Standard empty frame in canonical schema."""
    return pd.DataFrame({c: pd.Series(dtype="float") for c in CANON_COLS}).assign(
        ts=pd.Series(dtype=f"datetime64[ns, {TZ_NAME}]")
    )[CANON_COLS]


def _rename_hourly_columns(H: pd.DataFrame) -> pd.DataFrame:
    """Rename Open-Meteo hourly fields into our canonical names and add flags."""
    if H.empty:
        return _empty_canon_df()

    out = H.drop(columns=["time"], errors="ignore").rename(
        columns={
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "rel_humidity",
            "precipitation": "precip_mm",
            "rain": "rain_mm",
            "snowfall": "snow_cm",  # archive unit is cm
            "wind_speed_10m": "wind_kph",  # archive returns km/h
            "surface_pressure": "pressure_hpa",
            "weathercode": "wmo_code",
        }
    )
    # Derived binary flags
    out["is_rain"] = (pd.to_numeric(out.get("rain_mm", 0), errors="coerce").fillna(0) > 0).astype(
        int
    )
    out["is_snow"] = (pd.to_numeric(out.get("snow_cm", 0), errors="coerce").fillna(0) > 0).astype(
        int
    )

    # Ensure column order and presence
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = pd.NA if c != "ts" else pd.NaT
    return out[CANON_COLS]


# =============================================================================
# Network calls
# =============================================================================


def _fetch_month(a: date, b: date, lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch a single month (inclusive range [a, b]) from the archive API.
    Returns DataFrame in canonical schema.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": a.isoformat(),
        "end_date": b.isoformat(),
        "hourly": HOURLY,
        "timezone": TZ_NAME,
    }
    resp = r.get(BASE, params=params, timeout=60)

    # If a month straddles the future, archive may return 400; skip gracefully.
    if resp.status_code == 400:
        return _empty_canon_df()

    resp.raise_for_status()
    js = resp.json()
    if "hourly" not in js:
        return _empty_canon_df()

    H = pd.DataFrame(js["hourly"])
    if H.empty:
        return _empty_canon_df()

    H["ts"] = _to_local_ts(H["time"], TZ_NAME)
    H = _rename_hourly_columns(H)
    return H.sort_values("ts").reset_index(drop=True)


def fetch_open_meteo_hourly_cached(
    start: date,
    end: date,
    lat: float = 43.653,  # Toronto core
    lon: float = -79.383,
) -> pd.DataFrame:
    """
    Fetch hourly weather for [start, end] inclusive (clipped to <= yesterday),
    caching monthly results. Always returns canonical schema:

        ts (tz-aware in America/Toronto),
        temp_c, rel_humidity, precip_mm, rain_mm, snow_cm,
        wind_kph, pressure_hpa, wmo_code, is_rain, is_snow
    """
    s, e = _clip_to_archive(start, end)
    if s is None or e is None:
        return _empty_canon_df()

    frames: list[pd.DataFrame] = []
    for a, b in _month_iter(s, e):  # NOTE: iterate on clipped window
        ym = f"{a:%Y_%m}"
        p_parq = _cache_path(lat, lon, ym, "parquet")
        p_csv = _cache_path(lat, lon, ym, "csv")

        cached = _read_cached(p_parq, p_csv)
        if cached is None:
            df = _fetch_month(a, b, lat, lon)
            _write_cached(df, p_parq, p_csv)
            frames.append(df)
        # slice the month cache to [a, b]
        elif not cached.empty:
            mask = (cached["ts"].dt.date >= a) & (cached["ts"].dt.date <= b)
            frames.append(cached.loc[mask].copy())
        else:
            frames.append(_empty_canon_df())

    if not frames:
        return _empty_canon_df()

    out = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)

    # Enforce column presence/order
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = pd.NA if c != "ts" else pd.NaT
    out = out[CANON_COLS]

    return out
