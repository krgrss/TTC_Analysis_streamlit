"""
delay_features.py

Aggregate raw delay events into high-level metrics for dashboarding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _top_code_desc(s: pd.Series):
    """
    s: Series of code_desc values for one group (may contain NaN).
    Return the most frequent non-null value, or NA if none.
    """
    s = s.dropna()
    return s.value_counts().idxmax() if not s.empty else pd.NA


def build_route_delay_stats(delays: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by (mode, route_or_line).

    Returns columns like:
        mode
        route_or_line
        n_incidents          (# delay events)
        mean_delay_minutes   (avg duration)
        p95_delay_minutes    (95th percentile of duration)
        top_code_desc        (most common cause description)
    """

    if delays.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "route_or_line",
                "n_incidents",
                "mean_delay_minutes",
                "p95_delay_minutes",
                "top_code_desc",
            ]
        )

    d = delays.copy()

    # Ensure required columns exist with sane types
    if "code_desc" not in d.columns:
        d["code_desc"] = pd.NA
    d["delay_minutes"] = pd.to_numeric(d["delay_minutes"], errors="coerce").fillna(0)

    agg = (
        d.groupby(["mode", "route_or_line"])
        .agg(
            n_incidents=("delay_minutes", "count"),
            mean_delay_minutes=("delay_minutes", "mean"),
            p95_delay_minutes=(
                "delay_minutes",
                lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95),
            ),
            top_code_desc=("code_desc", _top_code_desc),
        )
        .reset_index()
    )

    agg["mean_delay_minutes"] = agg["mean_delay_minutes"].round(2)
    agg["p95_delay_minutes"] = agg["p95_delay_minutes"].round(2)

    return agg


def build_geo_hotspots(hotspots_df: pd.DataFrame, station_coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join hotspot stats (location, n_incidents, mean_delay_minutes)
    with station/stop coordinates to prepare for mapping.

    Inputs
    ------
    hotspots_df : output of build_location_hotspots()
        columns:
            location
            n_incidents
            mean_delay_minutes
    station_coords_df : output of load_station_coordinates()
        columns:
            location
            lat
            lon

    Returns
    -------
    pd.DataFrame with columns:
        location
        lat
        lon
        n_incidents
        mean_delay_minutes

    Notes
    -----
    We do a string join on 'location'. If names in the delay log don't
    exactly match names in subway_coor.csv (e.g. "St. George Station"
    vs "ST GEORGE"), you may want to normalize here (upper(), remove
    " STATION", etc.). We include a fallback heuristic below.
    """

    if hotspots_df.empty or station_coords_df.empty:
        return pd.DataFrame(
            columns=[
                "location",
                "lat",
                "lon",
                "n_incidents",
                "mean_delay_minutes",
            ]
        )

    # We'll try two passes:
    #   1. direct merge on location
    #   2. merge after normalizing "STATION", punctuation, uppercasing
    def normalize_name(s: pd.Series) -> pd.Series:
        tmp = (
            s.astype("string")
            .str.upper()
            .str.replace(" STATION", "", regex=False)
            .str.replace(" STN", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
        )
        return tmp

    h1 = hotspots_df.copy()
    s1 = station_coords_df.copy()

    h1["loc_norm"] = normalize_name(h1["location"])
    s1["loc_norm"] = normalize_name(s1["location"])

    merged = h1.merge(
        s1[["loc_norm", "lat", "lon"]],
        on="loc_norm",
        how="left",
    )

    # Now produce final clean output
    merged = merged[merged["lat"].notna() & merged["lon"].notna()].copy()

    final = merged[
        [
            "location",
            "lat",
            "lon",
            "n_incidents",
            "mean_delay_minutes",
        ]
    ].drop_duplicates(subset=["location"])

    return final.reset_index(drop=True)


def build_time_of_day_profile(delays: pd.DataFrame) -> pd.DataFrame:
    """
    Overall hour-of-day profile across all modes.

    Returns columns compatible with the dashboard:
      - hour           (0..23)            # used by dashboard
      - hour_of_day    (alias of hour)    # kept for robustness/compat
      - n_incidents
      - total_delay_minutes
      - mean_delay_minutes
    """
    if delays.empty or "delay_start_ts" not in delays.columns:
        return pd.DataFrame(
            columns=[
                "hour",
                "hour_of_day",
                "n_incidents",
                "total_delay_minutes",
                "mean_delay_minutes",
            ]
        )

    d = delays.copy()
    d = d[d["delay_start_ts"].notna()].copy()
    d["hour"] = pd.to_datetime(d["delay_start_ts"], errors="coerce").dt.hour
    d["delay_minutes"] = pd.to_numeric(d["delay_minutes"], errors="coerce")

    prof = (
        d.groupby("hour")
        .agg(
            n_incidents=("delay_minutes", "count"),
            total_delay_minutes=("delay_minutes", "sum"),
        )
        .reset_index()
        .sort_values("hour")
        .reset_index(drop=True)
    )

    prof["mean_delay_minutes"] = np.where(
        prof["n_incidents"] > 0,
        prof["total_delay_minutes"] / prof["n_incidents"],
        0.0,
    ).round(2)

    prof["hour_of_day"] = prof["hour"]  # alias for older callers
    return prof[["hour", "hour_of_day", "n_incidents", "total_delay_minutes", "mean_delay_minutes"]]


def build_top_causes(delays: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Top K causes overall across the whole system.
    """
    if delays.empty or "code_desc" not in delays.columns:
        return pd.DataFrame(columns=["code_desc", "n_incidents"])

    s = delays["code_desc"].dropna()
    if s.empty:
        return pd.DataFrame(columns=["code_desc", "n_incidents"])

    return (
        s.value_counts()
        .reset_index()
        .rename(columns={"index": "code_desc", "code_desc": "n_incidents"})
        .head(k)
    )


def build_overall_kpis(delays: pd.DataFrame) -> dict:
    """
    High-level metrics for the Overview tab.
    Returns a dict of simple KPIs we can show as cards.

    KPIs:
      total_incidents
      avg_delay_minutes
      total_delay_hours
      worst_line_by_incidents
    """
    if delays.empty:
        return {
            "total_incidents": 0,
            "avg_delay_minutes": 0.0,
            "total_delay_hours": 0.0,
            "worst_line_by_incidents": None,
        }

    total_incidents = len(delays)
    avg_delay_minutes = float(np.nanmean(pd.to_numeric(delays["delay_minutes"], errors="coerce")))
    total_delay_hours = float(
        np.nansum(pd.to_numeric(delays["delay_minutes"], errors="coerce")) / 60.0
    )

    # most delay-prone line/route
    worst_line = (
        delays["route_or_line"].dropna().value_counts().idxmax()
        if delays["route_or_line"].dropna().size > 0
        else None
    )

    return {
        "total_incidents": int(total_incidents),
        "avg_delay_minutes": round(avg_delay_minutes, 2),
        "total_delay_hours": round(total_delay_hours, 1),
        "worst_line_by_incidents": worst_line,
    }


def build_route_daily_timeseries(delays: pd.DataFrame) -> pd.DataFrame:
    """
    Time series of delays per day per route_or_line.

    Output columns:
        route_or_line
        date
        n_incidents
        mean_delay_minutes
    """
    if delays.empty:
        return pd.DataFrame(
            columns=[
                "route_or_line",
                "date",
                "n_incidents",
                "mean_delay_minutes",
            ]
        )

    d = delays.copy()
    d = d[d["delay_start_ts"].notna()].copy()
    d["date"] = pd.to_datetime(d["delay_start_ts"]).dt.date.astype("string")

    grouped = (
        d.groupby(["route_or_line", "date"])
        .agg(
            n_incidents=("delay_minutes", "count"),
            mean_delay_minutes=("delay_minutes", "mean"),
        )
        .reset_index()
    )

    grouped["mean_delay_minutes"] = grouped["mean_delay_minutes"].round(2)
    return grouped


def build_location_hotspots(delays: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """
    Top-k locations (stations / stops / intersections) by incident count.
    We'll also include mean delay to surface 'painful' hotspots.

    Output columns:
        location
        n_incidents
        mean_delay_minutes
    """
    if delays.empty:
        return pd.DataFrame(columns=["location", "n_incidents", "mean_delay_minutes"])

    d = delays.copy()
    d = d[d["location"].notna()].copy()

    agg = (
        d.groupby("location")
        .agg(
            n_incidents=("delay_minutes", "count"),
            mean_delay_minutes=("delay_minutes", "mean"),
        )
        .reset_index()
    )

    agg["mean_delay_minutes"] = agg["mean_delay_minutes"].round(2)

    # sort by # incidents then avg delay
    agg = agg.sort_values(
        ["n_incidents", "mean_delay_minutes"],
        ascending=False,
    ).head(k)

    return agg.reset_index(drop=True)


def build_dow_profile(delays: pd.DataFrame) -> pd.DataFrame:
    """
    Day-of-week profile (one row per weekday).

    Returns:
      - dow                ('Mon'..'Sun')   # used by dashboard
      - day_of_week        alias ('Mon'..'Sun')
      - n_incidents
    """
    if delays.empty or "delay_start_ts" not in delays.columns:
        return pd.DataFrame(columns=["dow", "day_of_week", "n_incidents"])

    d = delays.copy()
    d = d[d["delay_start_ts"].notna()].copy()

    dt = pd.to_datetime(d["delay_start_ts"], errors="coerce")
    dow_idx = dt.dt.dayofweek  # Monday=0
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    prof = (
        pd.DataFrame({"dow": dow_idx.map(dow_map)})
        .assign(n_incidents=1)
        .groupby("dow", as_index=False)["n_incidents"]
        .sum()
    )

    # enforce Mon..Sun order
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    prof["dow"] = pd.Categorical(prof["dow"], categories=order, ordered=True)
    prof = prof.sort_values("dow").reset_index(drop=True)

    prof["day_of_week"] = prof["dow"]  # alias for robustness
    return prof[["dow", "day_of_week", "n_incidents"]]


def build_mode_dashboard_data(delays: pd.DataFrame, mode: str) -> dict:
    """
    Build the set of dataframes needed for the per-mode dashboard
    (subway / streetcar / bus).

    Returns a dict with:
      daily_smooth_df        -> for the line chart
      heatmap_df             -> for weekday x hour heatmap
      cause_pie_df           -> for donut of incident types
      top_locations_df       -> for treemap
      top_routes_df          -> for bar chart
      direction_df           -> for direction stacked bar (if we have direction)
    """

    # Filter just this mode first
    d = delays.copy()
    d = d[d["mode"] == mode].copy()

    # --- prep common fields ---
    # standard datetime parts
    d["delay_start_ts"] = pd.to_datetime(d["delay_start_ts"], errors="coerce")
    d = d[d["delay_start_ts"].notna()].copy()

    d["date"] = d["delay_start_ts"].dt.date.astype("string")
    d["hour"] = d["delay_start_ts"].dt.hour
    d["dow_idx"] = d["delay_start_ts"].dt.dayofweek  # Monday=0
    d["day_of_week"] = d["dow_idx"].map(
        {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
    )

    # ensure numeric delay
    d["delay_minutes"] = pd.to_numeric(d["delay_minutes"], errors="coerce").fillna(0)

    # ---------------------------
    # 1. daily_smooth_df
    # ---------------------------
    # total minutes of delay per calendar day
    daily = (
        d.groupby("date")["delay_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"delay_minutes": "total_delay_minutes"})
    )

    # sort by date and add rolling / moving avg ("Smoothed Total Minutes of Delay by Date")
    daily["date_ts"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.sort_values("date_ts")
    # 7-day rolling mean (tweak if you want 14, 30, etc.)
    daily["smoothed_delay_minutes"] = (
        daily["total_delay_minutes"].rolling(window=7, min_periods=1).mean()
    )
    daily_smooth_df = daily

    # ---------------------------
    # 2. heatmap_df
    # ---------------------------
    # weekday (rows) x hour (cols), colored by total minutes of delay
    heatmap = (
        d.groupby(["day_of_week", "hour"])["delay_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"delay_minutes": "total_delay_minutes"})
    )

    # We'll keep day_of_week categorical for nicer ordering in the plot:
    day_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    heatmap["day_of_week"] = pd.Categorical(
        heatmap["day_of_week"], ordered=True, categories=day_order
    )
    heatmap_df = heatmap.sort_values(["day_of_week", "hour"])

    # ---------------------------
    # 3. cause_pie_df
    # ---------------------------
    if "code_desc" in d.columns:
        cause = (
            d.groupby("code_desc")["delay_minutes"]
            .sum()
            .reset_index()
            .rename(columns={"delay_minutes": "total_delay_minutes"})
        )
        # top 8 incidents by total minutes of delay
        cause = cause.sort_values("total_delay_minutes", ascending=False).head(8)
        total_all = cause["total_delay_minutes"].sum()
        cause["pct"] = (cause["total_delay_minutes"] / total_all * 100.0).round(2)
        cause_pie_df = cause
    else:
        cause_pie_df = pd.DataFrame(columns=["code_desc", "total_delay_minutes", "pct"])

    # ---------------------------
    # 4. top_locations_df (treemap)
    # ---------------------------
    loc = (
        d.groupby("location")["delay_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"delay_minutes": "total_delay_minutes"})
    )
    loc = loc.sort_values("total_delay_minutes", ascending=False).head(5)
    top_locations_df = loc

    # ---------------------------
    # 5. top_routes_df (bar)
    # ---------------------------
    rts = (
        d.groupby("route_or_line")["delay_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"delay_minutes": "total_delay_minutes"})
    )
    rts = rts.sort_values("total_delay_minutes", ascending=False).head(5)
    top_routes_df = rts

    # ---------------------------
    # 6. direction_df (stacked bar)
    # ---------------------------
    # Tableau screenshot has "Streetcar Direction: Delay & Gap Totals" with E / W / N / S.
    # This assumes your raw data has some column like "direction", "bound", or "dir".
    # We'll try to guess a column and sum delay_minutes.
    dir_col_guess = None
    for cand in ["direction", "bound", "dir", "direction_bound"]:
        if cand in d.columns:
            dir_col_guess = cand
            break

    if dir_col_guess is not None:
        direction_df = (
            d.groupby(dir_col_guess)["delay_minutes"]
            .sum()
            .reset_index()
            .rename(columns={dir_col_guess: "direction", "delay_minutes": "total_delay_minutes"})
        )
    else:
        direction_df = pd.DataFrame(columns=["direction", "total_delay_minutes"])

    return {
        "daily_smooth_df": daily_smooth_df,
        "heatmap_df": heatmap_df,
        "cause_pie_df": cause_pie_df,
        "top_locations_df": top_locations_df,
        "top_routes_df": top_routes_df,
        "direction_df": direction_df,
    }
