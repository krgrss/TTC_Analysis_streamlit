"""
delay_ingestion.py

Ingest TTC delay data (subway / streetcar / bus), standardize columns,
and prepare it for the Streamlit dashboard.

Key features:
- Supports 2024 .xlsx and 2025+ .csv for each mode.
- Joins TTC incident codes to human-readable descriptions from
  code-descriptions.csv / code-descriptions.json.
- Preserves a direction/bound column if provided by the source
  so we can plot "Delay by direction / bound" (E/W/N/S/etc).
- Normalizes timestamps, delay minutes, route/line name, etc.
- Loads station coordinates for hotspot mapping.

OUTPUTS
-------
load_all_delay_events() -> DataFrame:
    mode              ("subway"|"streetcar"|"bus")
    route_or_line     (string)
    location          (string station/stop/intersection)
    direction         (string like "E", "W", "N", "S", "NB", etc., may be NA)
    delay_start_ts    (datetime64)
    delay_minutes     (float >= 0)
    code              (short incident code, e.g. "MUIS")
    code_desc         (best-effort human-readable cause)

load_station_coordinates() -> DataFrame:
    location, lat, lon
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# CONFIG: file locations and naming
# ---------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if not DATA_DIR.exists():
    # Fallback if repository uses capitalized folder name
    alt = Path(__file__).resolve().parents[1] / "Data"
    if alt.exists():
        DATA_DIR = alt

SUBWAY_2024_XLSX = "ttc-subway-delay-data-2024.xlsx"
SUBWAY_2025_CSV = "TTC Subway Delay Data since 2025.csv"

STREETCAR_2024_XLSX = "ttc-streetcar-delay-data-2024.xlsx"
STREETCAR_2025_CSV = "TTC Streetcar Delay Data since 2025.csv"

BUS_2024_XLSX = "ttc-bus-delay-data-2024.xlsx"
BUS_2025_CSV = "TTC Bus Delay Data since 2025.csv"

CODEBOOK_CSV = "code-descriptions.csv"
CODEBOOK_JSON = "code-descriptions.json"

STATION_COORDS_CSV = "subway_coor.csv"


# ---------------------------------------------------------------------
# Helpers: header normalization, picking columns, file readers
# ---------------------------------------------------------------------


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to snake_case, lowercase, no punctuation.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace("[^a-z0-9_]+", "_", regex=True)
    return df


def _pick_exact(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column in df.columns that exactly matches any candidate.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_fuzzy(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """
    Return the first column whose name CONTAINS any of the given keywords.
    This helps with things like "incident_type_description".
    """
    for col in df.columns:
        for kw in keywords:
            if kw in col:
                return col
    return None


def _pick_fuzzy_all(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """
    Pick first column whose name contains ALL keywords (case-insensitive),
    useful for things like 'delay minutes' that might be written in many ways.
    """
    for col in df.columns:
        name = col.lower()
        if all(kw in name for kw in keywords):
            return col
    return None


def _read_delay_csv(path: Path) -> pd.DataFrame:
    """
    CSV reader: keep all columns as string (so nothing gets silently cast).
    """
    return pd.read_csv(
        path,
        dtype="string",
        encoding="utf-8-sig",
        low_memory=False,
    )


def _read_delay_excel(path: Path) -> pd.DataFrame:
    """
    Excel reader: keep all columns as string.
    Requires openpyxl in your environment.
    """
    return pd.read_excel(
        path,
        dtype="string",
        engine="openpyxl",
    )


# ---------------------------------------------------------------------
# Datetime parsing helpers (robust, warning-free)
# ---------------------------------------------------------------------


def _try_parse_series(series: pd.Series, formats: list[str]) -> pd.Series:
    """
    Try parsing a datetime-like series with a list of explicit formats.
    Returns the parse with the fewest NaT. If all fail, returns a best-effort
    fallback using pandas to_datetime with errors='coerce'.
    """
    best = None
    best_non_null = -1
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        non_null = int(parsed.notna().sum())
        if non_null > best_non_null:
            best_non_null = non_null
            best = parsed
    if best is None or best_non_null == 0:
        return pd.to_datetime(series, errors="coerce")
    return best


def _parse_datetime_fields(
    raw: pd.DataFrame,
    col_ts: str | None,
    col_date: str | None,
    col_time: str | None,
    *,
    combined_formats_hint: list[str] | None = None,
    date_formats_hint: list[str] | None = None,
    time_formats_hint: list[str] | None = None,
) -> pd.Series:
    """
    Parse timestamp from either a single datetime column or separate date/time columns.
    Attempts a handful of common formats to avoid per-row dateutil fallbacks.
    """
    dt_formats = combined_formats_hint or [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%Y%m%d",
    ]

    if col_ts:
        s = raw[col_ts].astype("string")
        return _try_parse_series(s, dt_formats)

    # Parse separate date and time
    date_formats = date_formats_hint or ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y%m%d"]
    time_formats = time_formats_hint or ["%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"]

    if col_date is None and col_time is None:
        return pd.to_datetime(pd.Series([], dtype="string"), errors="coerce")

    if col_date is not None:
        d = _try_parse_series(raw[col_date].astype("string"), date_formats)
    else:
        d = pd.Series(pd.NaT, index=raw.index)

    if col_time is not None:
        t_parsed = None
        best = None
        best_non_null = -1
        for fmt in time_formats:
            tmp = pd.to_datetime(raw[col_time].astype("string"), format=fmt, errors="coerce")
            non_null = int(tmp.notna().sum())
            if non_null > best_non_null:
                best_non_null = non_null
                best = tmp
        if best is None:
            # general fallback
            best = pd.to_datetime(raw[col_time].astype("string"), errors="coerce")
        # turn into strings HH:MM:SS
        t_parsed = best.dt.strftime("%H:%M:%S")
        t_parsed = t_parsed.fillna("00:00:00")
    else:
        t_parsed = pd.Series("00:00:00", index=raw.index)

    # combine
    d_str = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    d_str = d
    out = pd.to_datetime(
        d_str.dt.strftime("%Y-%m-%d") + " " + t_parsed,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    # if still NaT, try general fallback on original pieces
    mask = out.isna()
    if mask.any():
        combo = (
            raw[col_date].astype("string").fillna("")
            + " "
            + (raw[col_time].astype("string") if col_time else "")
        )
        out.loc[mask] = pd.to_datetime(combo[mask], errors="coerce")
    return out


def _guess_format_hints(
    mode_value: str, origin: Path | None
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    """Return (combined_formats, date_formats, time_formats) hints.

    We prioritize mode-specific patterns observed in TTC files.
    - Bus 2024 often: mm/dd/YYYY HH:MM (24h)
    - Subway 2024/2025: mixed but ISO-like or mm/dd/YYYY
    - Streetcar: similar to subway
    The origin filename can further bias selection.
    """
    combined: list[str] | None = None
    dformats: list[str] | None = None
    tformats: list[str] | None = None

    name = origin.name.lower() if origin is not None else ""

    if mode_value == "bus":
        # Bus data frequently uses US-style dates
        combined = ["%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y"]
        dformats = ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"]
        tformats = ["%H:%M:%S", "%H:%M", "%I:%M %p"]
    if "2024" in name:
        # Many 2024 files are Excel with mm/dd/YYYY in your dataset
        combined = ["%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y"] + (combined or [])
        dformats = ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"]

    return combined, dformats, tformats


def _iter_files(dir_path: Path) -> Iterable[Path]:
    try:
        return list(dir_path.iterdir())
    except Exception:
        return []


def _find_first_file_by_keywords(
    dir_path: Path, keywords: list[str], exts: list[str]
) -> Path | None:
    """
    Best-effort discovery when exact filenames don't match.
    Returns the first path whose lowercase filename contains ALL keywords
    and ends with one of the provided extensions.
    """
    kw = [k.lower() for k in keywords]
    extset = {e.lower() for e in exts}
    for p in _iter_files(dir_path):
        name = p.name.lower()
        if all(k in name for k in kw) and any(name.endswith(e) for e in extset):
            return p
    return None


# ---------------------------------------------------------------------
# Codebook loader
# ---------------------------------------------------------------------


def _load_codebook() -> pd.DataFrame:
    """
    Load TTC incident code → human-readable description mapping from
    code-descriptions.csv or code-descriptions.json in Data/.

    We try csv first, then json.

    We attempt to locate:
      - code_col   like "code", "incident_code", "delay_code"
      - desc_col   like "description", "incident_type", etc.

    Returns DataFrame:
        code      (UPPERCASE string)
        code_desc (string)

    If we fail, returns empty df with the same columns so
    downstream merge() still works.
    """
    csv_path = DATA_DIR / CODEBOOK_CSV
    json_path = DATA_DIR / CODEBOOK_JSON

    cb_raw = None

    if csv_path.exists():
        cb_raw = pd.read_csv(
            csv_path,
            dtype="string",
            encoding="utf-8-sig",
            low_memory=False,
        )
        _record_source("codebook", "path", csv_path)
    elif json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            raw_list = json.load(f)
        cb_raw = pd.DataFrame(raw_list, dtype="string")
        _record_source("codebook", "path", json_path)
    else:
        return pd.DataFrame(
            {
                "code": pd.Series(dtype="string"),
                "code_desc": pd.Series(dtype="string"),
            }
        )

    cb_raw = _normalize_headers(cb_raw)

    code_col = _pick_exact(
        cb_raw,
        [
            "code",
            "incident_code",
            "delay_code",
            "code_number",
            "code_no",
            "problem_code",
        ],
    )
    if code_col is None:
        code_col = _pick_fuzzy(
            cb_raw,
            [
                "code",
                "incident_code",
                "delay_code",
                "problem_code",
                "code_no",
            ],
        )
    desc_col = _pick_exact(
        cb_raw,
        [
            "code_desc",
            "description",
            "desc",
            "details",
            "incident_type",
            "incident_description",
            "incident_desc",
            "type",
        ],
    )
    if desc_col is None:
        desc_col = _pick_fuzzy(
            cb_raw,
            [
                "code_desc",
                "description",
                "desc",
                "details",
                "incident_type",
                "incident_description",
                "incident_desc",
                "type",
            ],
        )

    if code_col is None or desc_col is None:
        return pd.DataFrame(
            {
                "code": pd.Series(dtype="string"),
                "code_desc": pd.Series(dtype="string"),
            }
        )

    cb = pd.DataFrame(
        {
            "code": (cb_raw[code_col].astype("string").str.strip().str.upper()),
            "code_desc": (cb_raw[desc_col].astype("string").str.strip()),
        }
    )

    cb = cb.dropna(subset=["code"]).drop_duplicates(subset=["code"]).reset_index(drop=True)

    return cb


def _normalize_code_desc_series(series: pd.Series) -> pd.Series:
    """
    Normalize human-readable incident cause labels to canonical forms.

    This is intentionally conservative: we unify case and some common
    plurals/synonyms without over-mapping nuanced causes. Extend as you
    observe real-world idiosyncrasies in your files.
    """
    mapping = {
        "UNKNOWN": "UNKNOWN",
        "DIVERSION": "Diversion",
        "OPERATIONS": "Operations",
        "OPERATION": "Operations",
        "SECURITY": "Security",
        "SECURITY INCIDENT": "Security",
        "SECURITY INCIDENTS": "Security",
        "EMERGENCY SERVICES": "Emergency Services",
        "EMERGENCY SERVICE": "Emergency Services",
    }

    s = series.astype("string").str.strip()
    up = s.str.upper()
    norm = up.map(mapping).fillna(s)
    return norm


# ---------------------------------------------------------------------
# Station coordinate lookup for hotspot mapping
# ---------------------------------------------------------------------


def load_station_coordinates() -> pd.DataFrame:
    """
    Load Data/subway_coor.csv and normalize it into:
        location, lat, lon

    We do some sanity filtering for plausible Toronto-ish coords.
    """

    path = DATA_DIR / STATION_COORDS_CSV
    if not path.exists():
        return pd.DataFrame(
            {
                "location": pd.Series(dtype="string"),
                "lat": pd.Series(dtype="float"),
                "lon": pd.Series(dtype="float"),
            }
        )

    df = pd.read_csv(
        path,
        dtype="string",
        encoding="utf-8-sig",
        low_memory=False,
    )
    _record_source("station_coords", "path", path)

    df = _normalize_headers(df)

    def pick(cols: list[str]) -> str | None:
        for c in cols:
            if c in df.columns:
                return c
        return None

    name_col = pick(["station", "station_name", "stop_name", "location", "name"])
    lat_col = pick(["lat", "latitude", "lat_deg"])
    lon_col = pick(["lon", "lng", "longitude", "long", "lon_deg"])

    if name_col is None or lat_col is None or lon_col is None:
        # We'll just return an empty frame; dashboard will show 0 mapped hotspots.
        return pd.DataFrame(
            {
                "location": pd.Series(dtype="string"),
                "lat": pd.Series(dtype="float"),
                "lon": pd.Series(dtype="float"),
            }
        )

    out = pd.DataFrame(
        {
            "location": df[name_col].astype("string").str.strip(),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )

    # Sanity filter for lat/lon ranges near Toronto
    out = out[
        out["lat"].notna()
        & out["lon"].notna()
        & out["lat"].between(40.0, 50.0)
        & out["lon"].between(-90.0, -70.0)
    ].drop_duplicates(subset=["location"])

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Debug info: capture which files were used
# ---------------------------------------------------------------------

_SOURCE_DEBUG: dict[str, dict[str, str | None]] = {
    "codebook": {"path": None},
    "station_coords": {"path": None},
    "subway": {"2024": None, "2025": None},
    "streetcar": {"2024": None, "2025": None},
    "bus": {"2024": None, "2025": None},
}


def _record_source(group: str, key: str, path: Path | None) -> None:
    try:
        if group not in _SOURCE_DEBUG:
            _SOURCE_DEBUG[group] = {}
        _SOURCE_DEBUG[group][key] = None if path is None else str(path)
    except Exception:
        pass


def get_delay_ingestion_debug() -> dict[str, dict[str, str | None]]:
    """Return a dict of source files used for delays/codebook/stations."""
    return _SOURCE_DEBUG


# ---------------------------------------------------------------------
# Core cleaner for one mode's combined data
# ---------------------------------------------------------------------


def _clean_delay_df(
    df: pd.DataFrame,
    mode_value: str,
    codebook: pd.DataFrame,
    origin: Path | None = None,
) -> pd.DataFrame:
    """
    Take raw delay rows for a single mode (can be 2024+2025 merged),
    and normalize into canonical columns:

        mode
        route_or_line
        location
        direction
        delay_start_ts
        delay_minutes
        code
        code_desc
    """
    raw = _normalize_headers(df.copy())

    # Route / line identifier (try exact, then fuzzy)
    col_line = _pick_exact(
        raw,
        [
            "line",
            "line_name",
            "route",
            "route_name",
            "route_number",
            "line_number",
            "line_no",
        ],
    )
    if col_line is None:
        col_line = (
            _pick_fuzzy(raw, ["route"])
            or _pick_fuzzy(raw, ["line"])
            or _pick_fuzzy_all(raw, ["route", "name"])
            or _pick_fuzzy_all(raw, ["route", "number"])
        )  # noqa: E501

    # Physical location
    col_location = _pick_exact(
        raw,
        [
            "station",
            "station_name",
            "station_location",
            "location",
            "stop",
            "intersection",
        ],
    )
    if col_location is None:
        col_location = (
            _pick_fuzzy(raw, ["station"])
            or _pick_fuzzy(raw, ["location"])
            or _pick_fuzzy(raw, ["stop"])
        )  # noqa: E501

    # Delay minutes (exact then fuzzy like 'delay' + 'min')
    col_minutes = _pick_exact(
        raw,
        [
            "delay_min",
            "delay_minutes",
            "min_delay",
            "mins",
            "min_gap",
            "min_gap_delay",
            "gap_min",
            "gap_minutes",
        ],
    )
    if col_minutes is None:
        col_minutes = (
            _pick_fuzzy_all(raw, ["delay", "min"])
            or _pick_fuzzy_all(raw, ["delay", "mins"])
            or _pick_fuzzy_all(raw, ["min", "delay"])
        )  # noqa: E501

    # Timestamp pieces
    col_ts = _pick_exact(
        raw,
        [
            "timestamp",
            "reported_at",
            "datetime",
            "occurred_at",
            "event_time",
        ],
    )
    if col_ts is None:
        # combined 'date time' like 'incident date time'
        col_ts = (
            _pick_fuzzy_all(raw, ["date", "time"])
            or _pick_fuzzy(raw, ["datetime"])
            or _pick_fuzzy(raw, ["reported"])
        )  # noqa: E501

    col_date = _pick_exact(
        raw,
        [
            "date",
            "report_date",
            "delay_date",
            "incident_date",
        ],
    )
    if col_date is None:
        col_date = (
            _pick_fuzzy(raw, ["date"])
            or _pick_fuzzy(raw, ["report"])
            or _pick_fuzzy(raw, ["incident"])
        )  # noqa: E501
    col_time = _pick_exact(
        raw,
        [
            "time",
            "report_time",
            "delay_time",
            "incident_time",
        ],
    )
    if col_time is None:
        col_time = (
            _pick_fuzzy(raw, ["time"])
            or _pick_fuzzy(raw, ["incident"])
            or _pick_fuzzy(raw, ["report"])
        )  # noqa: E501

    # Short TTC incident code (esp. subway/streetcar)
    col_code = _pick_exact(
        raw,
        [
            "code",
            "delay_code",
            "code_number",
            "incident_code",
            "problem_code",
            "code_",
            "code_no",
        ],
    )
    if col_code is None:
        col_code = _pick_fuzzy(raw, ["code"]) or _pick_fuzzy_all(
            raw, ["problem", "code"]
        )  # noqa: E501

    # Direction / bound (E/W/N/S/NB/SB/etc.)
    col_direction = _pick_exact(
        raw,
        [
            "direction",
            "bound",
            "dir",
            "direction_bound",
            "bound_direction",
            "direction_of_travel",
            "travel_direction",
            "train_direction",
            "dir_bound",
        ],
    )
    if col_direction is None:
        col_direction = _pick_fuzzy(raw, ["direction"]) or _pick_fuzzy(raw, ["bound"])  # noqa: E501

    # Free-text cause if provided by the raw feed (bus/streetcar often have this)
    col_cause_text = _pick_exact(
        raw,
        [
            "incident_type",
            "incident_type_description",
            "incident",
            "incident_desc",
            "incident_description",
            "delay_cause",
            "delay_cause_description",
            "cause",
            "cause_description",
            "reason",
            "problem",
            "remarks",
            "comment",
            "description",
            "desc",
            "details",
        ],
    )
    if col_cause_text is None:
        col_cause_text = _pick_fuzzy(
            raw,
            [
                "incident",
                "cause",
                "reason",
                "problem",
                "description",
                "desc",
                "remark",
                "comment",
            ],
        )

    # Build timestamp with mode/file-aware hints
    combined_hint, date_hint, time_hint = _guess_format_hints(mode_value, origin)
    when = _parse_datetime_fields(
        raw,
        col_ts,
        col_date,
        col_time,
        combined_formats_hint=combined_hint,
        date_formats_hint=date_hint,
        time_formats_hint=time_hint,
    )

    # Delay minutes numeric
    if col_minutes:
        mins = pd.to_numeric(raw[col_minutes], errors="coerce")
    else:
        mins = pd.NA

    # Route / line text
    if col_line:
        route_series = raw[col_line].astype("string").str.strip()
    else:
        route_series = pd.Series(pd.NA, index=raw.index, dtype="string")

    # Location text
    if col_location:
        loc_series = raw[col_location].astype("string").str.strip()
    else:
        loc_series = pd.Series(pd.NA, index=raw.index, dtype="string")

    # Direction text
    if col_direction:
        dir_series = raw[col_direction].astype("string").str.strip()
    else:
        dir_series = pd.Series(pd.NA, index=raw.index, dtype="string")

    # TTC code
    if col_code:
        code_series = raw[col_code].astype("string").str.strip().str.upper()
    else:
        code_series = pd.Series(pd.NA, index=raw.index, dtype="string")

    # Free-text cause
    if col_cause_text:
        cause_direct_series = raw[col_cause_text].astype("string").str.strip()
    else:
        cause_direct_series = pd.Series(pd.NA, index=raw.index, dtype="string")

    standardized = pd.DataFrame(
        {
            "mode": mode_value,
            "route_or_line": route_series,
            "location": loc_series,
            "direction": dir_series,  # <- keep it for direction/bound plots
            "delay_start_ts": when,
            "delay_minutes": mins,
            "code": code_series,
            "cause_direct": cause_direct_series,
        }
    )

    # Join in human-readable cause text from codebook
    merged = standardized.merge(codebook, on="code", how="left")

    # Final code_desc: prefer codebook, then fallback to cause_direct
    merged["code_desc"] = (
        merged.get("code_desc", pd.Series(pd.NA, index=merged.index))
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna(merged["cause_direct"].astype("string").str.strip().replace("", pd.NA))
    )
    # Normalize to canonical labels where possible
    merged["code_desc"] = _normalize_code_desc_series(merged["code_desc"])

    # Numeric cleanup
    merged["delay_minutes"] = pd.to_numeric(merged["delay_minutes"], errors="coerce").clip(lower=0)

    # Filter out junk rows that we can't plot meaningfully
    merged = merged[
        merged["delay_start_ts"].notna()
        & merged["route_or_line"].notna()
        & merged["delay_minutes"].notna()
    ].copy()

    merged = merged.drop(columns=["cause_direct"], errors="ignore")

    if "code_desc" not in merged.columns:
        merged["code_desc"] = pd.NA

    return merged


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def load_all_delay_events() -> pd.DataFrame:
    """
    Load subway, streetcar, and bus delay data for 2024 (.xlsx) and 2025 (.csv),
    normalize it with _clean_delay_df(), and combine into one big dataframe.
    """
    codebook = _load_codebook()

    # --- Subway ---
    subway_sources: list[tuple[pd.DataFrame, Path]] = []
    p_s24 = DATA_DIR / SUBWAY_2024_XLSX
    p_s25 = DATA_DIR / SUBWAY_2025_CSV

    if p_s24.exists():
        subway_sources.append((_read_delay_excel(p_s24), p_s24))
        _record_source("subway", "2024", p_s24)
    else:
        alt = _find_first_file_by_keywords(DATA_DIR, ["subway", "2024"], [".xlsx", ".xls", ".xlsm"])
        if alt is not None:
            subway_sources.append((_read_delay_excel(alt), alt))
            _record_source("subway", "2024", alt)
    if p_s25.exists():
        subway_sources.append((_read_delay_csv(p_s25), p_s25))
        _record_source("subway", "2025", p_s25)
    else:
        alt = _find_first_file_by_keywords(DATA_DIR, ["subway", "2025"], [".csv"])
        if alt is not None:
            subway_sources.append((_read_delay_csv(alt), alt))
            _record_source("subway", "2025", alt)

    if subway_sources:
        subway_std = pd.concat(
            [_clean_delay_df(df, "subway", codebook, origin=path) for df, path in subway_sources],
            ignore_index=True,
        )
    else:
        subway_std = pd.DataFrame(
            columns=[
                "mode",
                "route_or_line",
                "location",
                "direction",
                "delay_start_ts",
                "delay_minutes",
                "code",
                "code_desc",
            ]
        )

    # --- Streetcar ---
    streetcar_sources: list[tuple[pd.DataFrame, Path]] = []
    p_sc24 = DATA_DIR / STREETCAR_2024_XLSX
    p_sc25 = DATA_DIR / STREETCAR_2025_CSV

    if p_sc24.exists():
        streetcar_sources.append((_read_delay_excel(p_sc24), p_sc24))
        _record_source("streetcar", "2024", p_sc24)
    else:
        alt = _find_first_file_by_keywords(
            DATA_DIR, ["streetcar", "2024"], [".xlsx", ".xls", ".xlsm"]
        )
        if alt is not None:
            streetcar_sources.append((_read_delay_excel(alt), alt))
            _record_source("streetcar", "2024", alt)
    if p_sc25.exists():
        streetcar_sources.append((_read_delay_csv(p_sc25), p_sc25))
        _record_source("streetcar", "2025", p_sc25)
    else:
        alt = _find_first_file_by_keywords(DATA_DIR, ["streetcar", "2025"], [".csv"])
        if alt is not None:
            streetcar_sources.append((_read_delay_csv(alt), alt))
            _record_source("streetcar", "2025", alt)

    if streetcar_sources:
        streetcar_std = pd.concat(
            [
                _clean_delay_df(df, "streetcar", codebook, origin=path)
                for df, path in streetcar_sources
            ],
            ignore_index=True,
        )
    else:
        streetcar_std = pd.DataFrame(
            columns=[
                "mode",
                "route_or_line",
                "location",
                "direction",
                "delay_start_ts",
                "delay_minutes",
                "code",
                "code_desc",
            ]
        )

    # --- Bus ---
    bus_sources: list[tuple[pd.DataFrame, Path]] = []
    p_b24 = DATA_DIR / BUS_2024_XLSX
    p_b25 = DATA_DIR / BUS_2025_CSV

    if p_b24.exists():
        bus_sources.append((_read_delay_excel(p_b24), p_b24))
        _record_source("bus", "2024", p_b24)
    else:
        alt = _find_first_file_by_keywords(DATA_DIR, ["bus", "2024"], [".xlsx", ".xls", ".xlsm"])
        if alt is not None:
            bus_sources.append((_read_delay_excel(alt), alt))
            _record_source("bus", "2024", alt)
    if p_b25.exists():
        bus_sources.append((_read_delay_csv(p_b25), p_b25))
        _record_source("bus", "2025", p_b25)
    else:
        alt = _find_first_file_by_keywords(DATA_DIR, ["bus", "2025"], [".csv"])
        if alt is not None:
            bus_sources.append((_read_delay_csv(alt), alt))
            _record_source("bus", "2025", alt)

    if bus_sources:
        bus_std = pd.concat(
            [_clean_delay_df(df, "bus", codebook, origin=path) for df, path in bus_sources],
            ignore_index=True,
        )
    else:
        bus_std = pd.DataFrame(
            columns=[
                "mode",
                "route_or_line",
                "location",
                "direction",
                "delay_start_ts",
                "delay_minutes",
                "code",
                "code_desc",
            ]
        )

    # Combine all modes
    delays = pd.concat([subway_std, streetcar_std, bus_std], ignore_index=True)

    # Final cleanup, normalization
    delays["delay_start_ts"] = pd.to_datetime(
        delays["delay_start_ts"],
        errors="coerce",
    )

    if "delay_minutes" in delays.columns:
        delays["delay_minutes"] = pd.to_numeric(delays["delay_minutes"], errors="coerce").clip(
            lower=0
        )

    # strip blanks to NA for text columns
    for c in ["route_or_line", "location", "direction", "code", "code_desc"]:
        if c in delays.columns:
            delays[c] = delays[c].astype("string").str.strip().where(lambda s: s != "", pd.NA)

    # Sort oldest→newest just for stability
    delays = delays.sort_values("delay_start_ts").reset_index(drop=True)

    return delays
