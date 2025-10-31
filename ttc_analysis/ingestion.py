from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if not DATA_DIR.exists():
    # Fallback for environments where the folder is named with a capital D
    alt = Path(__file__).resolve().parents[1] / "Data"
    if alt.exists():
        DATA_DIR = alt

ROUTE_TYPE_MAP = {
    0: "streetcar",
    1: "subway",
    2: "rail",
    3: "bus",
}


def load_stops() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "stops.txt",
        dtype={
            "stop_id": "string",
            "stop_name": "string",
            "stop_lat": "float64",
            "stop_lon": "float64",
        },
        low_memory=False,
        encoding="utf-8-sig",
    )

    df = df[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
    df = df.rename(
        columns={
            "stop_name": "name",
            "stop_lat": "lat",
            "stop_lon": "lon",
        }
    )

    return df


def load_routes() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "routes.txt",
        dtype={
            "route_id": "string",
            "route_short_name": "string",
            "route_long_name": "string",
            "route_type": "Int64",  # nullable int
            "route_color": "string",
            "route_text_color": "string",
        },
        low_memory=False,
        encoding="utf-8-sig",
    )

    df = df[
        [
            "route_id",
            "route_short_name",
            "route_long_name",
            "route_type",
            "route_color",
            "route_text_color",
        ]
    ].copy()

    df = df.rename(
        columns={
            "route_short_name": "short_name",
            "route_long_name": "long_name",
            "route_color": "color",
            "route_text_color": "text_color",
        }
    )

    df["mode"] = df["route_type"].map(ROUTE_TYPE_MAP).fillna("other")

    return df


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string"), format="%Y%m%d", errors="coerce")


def load_calendar() -> pd.DataFrame:
    # We want service_id as string from the start.
    df = pd.read_csv(
        DATA_DIR / "calendar.txt",
        dtype={
            "service_id": "string",
            "monday": "Int64",
            "tuesday": "Int64",
            "wednesday": "Int64",
            "thursday": "Int64",
            "friday": "Int64",
            "saturday": "Int64",
            "sunday": "Int64",
            "start_date": "string",
            "end_date": "string",
        },
        low_memory=False,
        encoding="utf-8-sig",
    ).copy()

    # parse dates
    df["start_date"] = _parse_yyyymmdd(df["start_date"])
    df["end_date"] = _parse_yyyymmdd(df["end_date"])

    weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    weekend_cols = ["saturday", "sunday"]

    df["runs_weekday"] = df[weekday_cols].max(axis=1) > 0
    df["runs_weekend"] = df[weekend_cols].max(axis=1) > 0

    return df[
        [
            "service_id",
            "runs_weekday",
            "runs_weekend",
            "start_date",
            "end_date",
        ]
    ].copy()


def load_calendar_dates() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "calendar_dates.txt",
        dtype={
            "service_id": "string",
            "date": "string",
            "exception_type": "Int64",
        },
        low_memory=False,
        encoding="utf-8-sig",
    ).copy()

    df["date"] = _parse_yyyymmdd(df["date"])

    return df[["service_id", "date", "exception_type"]].copy()


def load_agency() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "agency.txt",
        dtype="string",
        low_memory=False,
        encoding="utf-8-sig",
    ).copy()
    return df


def load_trips() -> pd.DataFrame:
    """
    trip_id, route_id, service_id are *identifiers*, not numbers.
    Force them all to string now.
    """
    df = pd.read_csv(
        DATA_DIR / "trips.txt",
        usecols=["trip_id", "route_id", "service_id"],
        dtype={
            "trip_id": "string",
            "route_id": "string",
            "service_id": "string",
        },
        low_memory=False,
        encoding="utf-8-sig",
    )
    return df


def load_stop_times() -> pd.DataFrame:
    """
    trip_id, stop_id are identifiers => string
    stop_sequence is an integer order along trip, can be nullable Int64
    """
    df = pd.read_csv(
        DATA_DIR / "stop_times.txt",
        usecols=["trip_id", "stop_id", "stop_sequence"],
        dtype={
            "trip_id": "string",
            "stop_id": "string",
            "stop_sequence": "Int64",
        },
        low_memory=False,
        encoding="utf-8-sig",
    )
    return df


def load_all():
    return {
        "stops": load_stops(),
        "routes": load_routes(),
        "calendar": load_calendar(),
        "calendar_dates": load_calendar_dates(),
        "agency": load_agency(),
        "trips": load_trips(),
        "stop_times": load_stop_times(),
    }
