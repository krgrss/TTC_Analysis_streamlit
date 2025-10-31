"""
dashboard.py

Interactive TTC dashboard.

Run:
    streamlit run ttc_analysis/dashboard.py

What you get:
- Tab 1 (🗺 Network & Service):
    * Stop connectivity map (GTFS)
    * Top connected stops
    * Route service profile (weekday / weekend, active window)

- Tab 2 (⚠ Reliability / Delays):
    * KPIs (total incidents, avg delay, etc.)
    * Most delay-prone lines/routes
    * Daily trend for a selected line/route
    * Time-of-day + day-of-week patterns
    * Top causes
    * Delay hotspot table
    * Delay hotspot map (using station coordinates)

3. 🧪 Data QA
   - Missing data / bad values
   - % unknown incident causes
   - How many hotspots couldn't be placed on the map
   - How many stops got filtered off the network map
   - List of unknown codes


All heavy lifting is done in:
    ingestion.py / features.py              (static GTFS network)
    delay_ingestion.py / delay_features.py  (delay logs 2024-2025)
    viz_map.py / viz_delay_map.py           (folium maps)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

# ----- Weather dashborad ------
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
from weather_api import fetch_open_meteo_hourly_cached

from ttc_analysis.delay_features import (
    build_dow_profile,
    build_geo_hotspots,
    build_location_hotspots,
    build_overall_kpis,
    build_route_daily_timeseries,
    build_route_delay_stats,
    build_time_of_day_profile,
    build_top_causes,
)

# --- reliability / delay imports ---
from ttc_analysis.delay_ingestion import (
    load_all_delay_events,
    load_station_coordinates,
)
from ttc_analysis.features import build_route_features, build_stop_features

# --- static network (GTFS) imports ---
from ttc_analysis.ingestion import load_all
from ttc_analysis.viz_delay_map import make_delay_hotspot_map
from ttc_analysis.viz_map import (
    build_connectivity_map_df,
    make_stop_connectivity_map,
)

_HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None

# =====================================================
# HELPERS TO COMPUTE QA METRICS
# =====================================================


def _compute_network_qa(stop_features_df: pd.DataFrame, map_df: pd.DataFrame) -> dict:
    """
    Produce basic QA metrics for the GTFS network view.
    """
    total_stops_after_features = len(stop_features_df)
    total_stops_plotted = len(map_df)

    # stops that didn't make it to the map: usually invalid/missing lat/lon etc.
    dropped_stops = max(
        0,
        total_stops_after_features - total_stops_plotted,
    )

    return {
        "total_stops_after_features": total_stops_after_features,
        "total_stops_on_map": total_stops_plotted,
        "stops_dropped_from_map": dropped_stops,
    }


def _compute_reliability_qa(
    delays: pd.DataFrame,
    hotspots_df: pd.DataFrame,
    geo_hotspots_df: pd.DataFrame,
) -> dict:
    """
    Produce QA metrics for the delay / reliability pipeline.
    """

    d = delays.copy()

    total_rows = len(d)

    # delay_minutes validity
    d["delay_minutes_num"] = pd.to_numeric(d.get("delay_minutes"), errors="coerce")
    missing_delay_minutes = d["delay_minutes_num"].isna().sum()

    # code_desc coverage
    if "code_desc" in d.columns:
        code_desc_clean = d["code_desc"].astype("string").str.strip().replace("", pd.NA)
        missing_code_desc = code_desc_clean.isna().sum()
    else:
        missing_code_desc = total_rows  # nothing present
    pct_missing_code_desc = (missing_code_desc / total_rows * 100.0) if total_rows > 0 else 0.0

    # hotspot geocoding success
    num_hotspots = len(hotspots_df)
    num_geo_hotspots = len(geo_hotspots_df)
    unmatched_hotspots = max(0, num_hotspots - num_geo_hotspots)

    # unknown codes table:
    # rows where code_desc is missing -> group by 'code', summarize freq & minutes
    if "code_desc" in d.columns and "code" in d.columns:
        unknown_mask = d["code_desc"].astype("string").str.strip().replace("", pd.NA).isna()
        unk = d[unknown_mask].copy()
        if not unk.empty:
            unk["delay_minutes_num"] = pd.to_numeric(unk["delay_minutes"], errors="coerce").fillna(
                0
            )
            unknown_codes_df = (
                unk.groupby("code")
                .agg(
                    n_rows=("code", "count"),
                    total_delay_minutes=("delay_minutes_num", "sum"),
                )
                .reset_index()
                .sort_values(
                    ["n_rows", "total_delay_minutes"],
                    ascending=False,
                )
            )
        else:
            unknown_codes_df = pd.DataFrame(columns=["code", "n_rows", "total_delay_minutes"])
    else:
        unknown_codes_df = pd.DataFrame(columns=["code", "n_rows", "total_delay_minutes"])

    return {
        "total_delay_rows": total_rows,
        "missing_delay_minutes": int(missing_delay_minutes),
        "pct_missing_code_desc": round(pct_missing_code_desc, 2),
        "num_hotspots": num_hotspots,
        "num_geo_hotspots": num_geo_hotspots,
        "unmatched_hotspots": unmatched_hotspots,
        "unknown_codes_df": unknown_codes_df,
    }


# =========================
# DATA LOADERS (CACHED)
# =========================


@st.cache_data(show_spinner=False)
def load_static_network():
    """
    Load GTFS network data and compute:
    - stop_features_df  (stop_id, name, lat, lon, n_routes)
    - route_features_df (route-level weekday/weekend flags, etc.)
    - map_df            (cleaned stops for mapping)
    """
    data = load_all()
    stops_df = data["stops"]
    routes_df = data["routes"]
    calendar_df = data["calendar"]
    trips_df = data["trips"]
    stop_times_df = data["stop_times"]

    stop_features_df = build_stop_features(
        stops_df=stops_df,
        stop_times_df=stop_times_df,
        trips_df=trips_df,
    )

    route_features_df = build_route_features(
        routes_df=routes_df,
        trips_df=trips_df,
        calendar_df=calendar_df,
    )

    map_df = build_connectivity_map_df(stop_features_df)

    total_stops_after_features = len(stop_features_df)
    total_stops_on_map = len(map_df)
    stops_dropped_from_map = max(
        0,
        total_stops_after_features - total_stops_on_map,
    )

    qa_network = {
        "total_stops_after_features": total_stops_after_features,
        "total_stops_on_map": total_stops_on_map,
        "stops_dropped_from_map": stops_dropped_from_map,
    }

    return {
        "stop_features_df": stop_features_df,
        "route_features_df": route_features_df,
        "map_df": map_df,
        "qa_network": qa_network,
    }


@st.cache_data(show_spinner=False)
def load_reliability():
    """
    Load aggregated 2024/2025 delay data for subway/streetcar/bus.

    Returns dict with:
        full_delays_df        raw combined standardized delay rows
        kpis                  overall KPI cards
        route_stats           per-route/line summary
        route_ts              per-route daily incident trend
        tod_profile           incidents by hour_of_day
        dow_profile           incidents by day_of_week x hour_of_day
        top_causes            top incident causes system-wide
        hotspots              top locations (stations/intersections) by total delay
        geo_hotspots          hotspots with lat/lon for mapping
        qa_reliability        QA metrics, including % missing cause and per-mode breakdown
    """

    # 1. Load all cleaned events (subway/streetcar/bus, 2024+2025)
    delays = load_all_delay_events()

    # 2. Compute the normal analytics we surface in the dashboard
    kpis = build_overall_kpis(delays)
    route_stats = build_route_delay_stats(delays)
    route_ts = build_route_daily_timeseries(delays)
    tod_profile = build_time_of_day_profile(delays)
    dow_profile = build_dow_profile(delays)
    top_causes = build_top_causes(delays, k=10)
    hotspots = build_location_hotspots(delays, k=15)

    station_coords = load_station_coordinates()
    geo_hotspots = build_geo_hotspots(hotspots, station_coords)

    # 3. Build QA metrics inline so we can inspect data health
    d = delays.copy()

    total_rows = len(d)

    # delay_minutes coverage
    d["delay_minutes_num"] = pd.to_numeric(d.get("delay_minutes"), errors="coerce")
    missing_delay_minutes = int(d["delay_minutes_num"].isna().sum())

    # code_desc (human-readable incident cause) coverage
    if "code_desc" in d.columns:
        code_clean = d["code_desc"].astype("string").str.strip().replace("", pd.NA)
        missing_code_desc = int(code_clean.isna().sum())
    else:
        # if column doesn't exist at all, then everything is "missing"
        missing_code_desc = total_rows

    pct_missing_code_desc = (missing_code_desc / total_rows * 100.0) if total_rows > 0 else 0.0

    # hotspot coordinate coverage
    num_hotspots = len(hotspots)
    num_geo_hotspots = len(geo_hotspots)
    unmatched_hotspots = max(0, num_hotspots - num_geo_hotspots)

    # unknown_codes_df:
    # rows where we do NOT have a mapped description, grouped by the raw "code"
    if "code_desc" in d.columns and "code" in d.columns:
        unknown_mask = d["code_desc"].astype("string").str.strip().replace("", pd.NA).isna()

        unk = d[unknown_mask].copy()

        if not unk.empty:
            unk["delay_minutes_num"] = pd.to_numeric(unk["delay_minutes"], errors="coerce").fillna(
                0
            )

            unknown_codes_df = (
                unk.groupby("code")
                .agg(
                    n_rows=("code", "count"),
                    total_delay_minutes=("delay_minutes_num", "sum"),
                )
                .reset_index()
                .sort_values(
                    ["n_rows", "total_delay_minutes"],
                    ascending=False,
                )
            )
        else:
            unknown_codes_df = pd.DataFrame(columns=["code", "n_rows", "total_delay_minutes"])
    else:
        unknown_codes_df = pd.DataFrame(columns=["code", "n_rows", "total_delay_minutes"])

    # per-mode breakdown of missing cause_desc
    # this tells us: is subway the problem? streetcar? bus?
    if "code_desc" in d.columns and "mode" in d.columns:
        tmp = d.copy()
        tmp["_missing_cause"] = (
            tmp["code_desc"].astype("string").str.strip().replace("", pd.NA).isna()
        )

        mode_cause_stats = (
            tmp.groupby("mode")
            .agg(
                rows=("mode", "count"),
                rows_missing=("_missing_cause", "sum"),
            )
            .reset_index()
        )

        mode_cause_stats["pct_missing"] = (
            mode_cause_stats["rows_missing"] / mode_cause_stats["rows"] * 100.0
        ).round(2)

    else:
        mode_cause_stats = pd.DataFrame(columns=["mode", "rows", "rows_missing", "pct_missing"])

    qa_reliability = {
        "total_delay_rows": total_rows,
        "missing_delay_minutes": missing_delay_minutes,
        "pct_missing_code_desc": round(pct_missing_code_desc, 2),
        "num_hotspots": num_hotspots,
        "num_geo_hotspots": num_geo_hotspots,
        "unmatched_hotspots": unmatched_hotspots,
        "unknown_codes_df": unknown_codes_df,
        "missing_cause_by_mode": mode_cause_stats,
    }

    # 4. Return the full bundle that the dashboard tabs expect
    return {
        "full_delays_df": delays,
        "kpis": kpis,
        "route_stats": route_stats,
        "route_ts": route_ts,
        "tod_profile": tod_profile,
        "dow_profile": dow_profile,
        "top_causes": top_causes,
        "hotspots": hotspots,
        "geo_hotspots": geo_hotspots,
        "qa_reliability": qa_reliability,
    }


# =========================
# VIEW HELPERS
# =========================


def tab_network_view(net: dict):
    """
    Render the static network / coverage / connectivity view.
    - Folium map of stops sized+colored by n_routes
    - Top connected stops table + bar chart
    - Route service profile table
    """
    st.header("Network Coverage & Connectivity")

    col_map, col_side = st.columns([2, 1], gap="large")

    # ---- left: connectivity map
    with col_map:
        st.subheader("Stop Connectivity Map")
        st.markdown(
            "Each point is a TTC stop. Size and color reflect how many distinct "
            "routes serve that stop. Red/big = more connections."
        )

        try:
            folium_map = make_stop_connectivity_map(
                net["map_df"],
                base_zoom=11,
                tiles="OpenStreetMap",
                min_radius=3.0,
                max_radius=12.0,
            )
            st_folium(
                folium_map,
                width=800,
                height=600,
                returned_objects=[],
            )
        except Exception as e:
            st.error("Map render failed.")
            st.exception(e)

    # ---- right: top connected stops
    with col_side:
        st.subheader("Top 15 most connected stops (by # of routes)")

        top_df = (
            net["stop_features_df"]
            .sort_values("n_routes", ascending=False)
            .head(15)
            .reset_index(drop=True)
            .copy()
        )

        st.dataframe(
            top_df[["stop_id", "name", "n_routes", "lat", "lon"]],
            hide_index=True,
            use_container_width=True,
        )

        fig = px.bar(
            top_df,
            x="name",
            y="n_routes",
            title="Stops with the highest number of distinct routes",
            labels={"name": "Stop", "n_routes": "# routes"},
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=50, b=120),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Route Service Profile")
    st.caption("Which routes run weekdays vs weekends, and their service window.")

    route_df = net["route_features_df"].copy()

    # prettify service window dates for display
    if "start_date_min" in route_df.columns:
        route_df["start_date_min"] = pd.to_datetime(
            route_df["start_date_min"], errors="coerce"
        ).dt.date.astype("string")
    if "end_date_max" in route_df.columns:
        route_df["end_date_max"] = pd.to_datetime(
            route_df["end_date_max"], errors="coerce"
        ).dt.date.astype("string")

    st.dataframe(
        route_df[
            [
                "route_id",
                "short_name",
                "long_name",
                "mode",
                "runs_weekday",
                "runs_weekend",
                "start_date_min",
                "end_date_max",
            ]
        ].reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )


def tab_reliability_view(rel: dict):
    """
    Render the reliability / delays analytics view.
    - KPI cards
    - Worst routes
    - Trend over time
    - Time-of-day + DOW patterns
    - Top causes
    - Hotspot table + hotspot map
    """
    st.header("Service Reliability / Delays")

    # -------- KPIs --------
    st.subheader("System Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Incidents", rel["kpis"]["total_incidents"])
    k2.metric("Avg Delay (min)", rel["kpis"]["avg_delay_minutes"])
    k3.metric("Total Delay Hours", rel["kpis"]["total_delay_hours"])
    k4.metric("Most Impacted Line/Route", rel["kpis"]["worst_line_by_incidents"])

    # Optional diagnostics for quick sanity checks
    with st.expander("Debug diagnostics (years by mode, sources)", expanded=False):
        try:
            import pandas as pd

            d = rel.get("full_delays_df")
            if d is not None and not d.empty:
                tmp = d.copy()
                tmp["year"] = pd.to_datetime(tmp["delay_start_ts"], errors="coerce").dt.year
                year_counts = (
                    tmp.groupby(["mode", "year"])
                    .size()
                    .reset_index(name="rows")
                    .sort_values(["mode", "year"])
                    .reset_index(drop=True)
                )
                st.markdown("**Rows by mode × year**")
                st.dataframe(year_counts, hide_index=True, use_container_width=True)
            from ttc_analysis.delay_ingestion import get_delay_ingestion_debug

            src = get_delay_ingestion_debug()
            rows = []
            for group, kv in src.items():
                if isinstance(kv, dict):
                    for k, v in kv.items():
                        rows.append({"category": group, "key": k, "path": v})
            if rows:
                st.markdown("**Delay sources detected**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        except Exception:
            st.info("Diagnostics unavailable.")

    # -------- Deep dive for each transportation mode ---------
    st.subheader("Mode Deep Dive")
    mode_tab_subway, mode_tab_streetcar, mode_tab_bus = st.tabs(
        ["Subway Data", "Streetcar Data", "Bus Data"]
    )

    def render_mode_panel(mode_name: str):
        from ttc_analysis.delay_features import build_mode_dashboard_data

        mode_data = build_mode_dashboard_data(rel["full_delays_df"], mode_name)

        # 1. Smoothed daily delay minutes (rolling avg)
        st.markdown("**Smoothed Total Minutes of Delay by Date**")
        daily_df = mode_data["daily_smooth_df"]
        if daily_df.empty:
            st.info("No data for this mode.")
        else:
            fig_line = px.line(
                daily_df,
                x="date_ts",
                y="smoothed_delay_minutes",
                labels={
                    "date_ts": "Date",
                    "smoothed_delay_minutes": "Moving Avg of Min Delay",
                },
            )
            st.plotly_chart(fig_line, use_container_width=True)

        top_row = st.columns([2, 1])
        with top_row[0]:
            # Heatmap weekday x hour
            st.markdown("**Delay Time Heatmap**")
            heat_df = mode_data["heatmap_df"]
            if heat_df.empty:
                st.info("No heatmap data.")
            else:
                fig_heat = px.density_heatmap(
                    heat_df,
                    x="hour",
                    y="day_of_week",
                    z="total_delay_minutes",
                    histfunc="sum",
                    nbinsx=24,
                    nbinsy=7,
                    color_continuous_scale="Oranges",
                    labels={
                        "hour": "Hour of Day",
                        "day_of_week": "Day",
                        "total_delay_minutes": "Total Minutes of Delay",
                    },
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        with top_row[1]:
            # Direction totals (E/W/N/S etc)
            st.markdown("**Direction: Total Delay Minutes**")
            dir_df = mode_data["direction_df"]
            if dir_df.empty:
                st.info("No direction/bound data for this mode.")
            else:
                fig_dir = px.bar(
                    dir_df,
                    x="direction",
                    y="total_delay_minutes",
                    labels={
                        "direction": "Direction / Bound",
                        "total_delay_minutes": "Total Minutes of Delay",
                    },
                )
                st.plotly_chart(fig_dir, use_container_width=True)

            # Cause donut
            st.markdown("**Top Incident Types (Share of Total Minutes)**")
            cause_df = mode_data["cause_pie_df"].copy()
            if cause_df.empty:
                st.info("No incident cause breakdown.")
            else:
                vals = cause_df["total_delay_minutes"].fillna(0)
                if float(vals.sum()) <= 0.0 and "n_incidents" in cause_df.columns:
                    # Fallback to incident counts when minutes sum to zero
                    fig_pie = px.pie(
                        cause_df,
                        names="code_desc",
                        values="n_incidents",
                        hole=0.4,
                    )
                else:
                    fig_pie = px.pie(
                        cause_df,
                        names="code_desc",
                        values="total_delay_minutes",
                        hole=0.4,
                    )
                st.plotly_chart(fig_pie, use_container_width=True)

        bottom_row = st.columns(2)
        with bottom_row[0]:
            # treemap: locations by total delay minutes
            st.markdown("**Top Locations by Total Minutes of Delay**")
            loc_df = mode_data["top_locations_df"]
            if loc_df.empty:
                st.info("No location data.")
            else:
                fig_tree = px.treemap(
                    loc_df,
                    path=["location"],
                    values="total_delay_minutes",
                    color="total_delay_minutes",
                    color_continuous_scale="Oranges",
                )
                st.plotly_chart(fig_tree, use_container_width=True)

        with bottom_row[1]:
            # bar: top routes/lines
            st.markdown("**Top Routes / Lines by Total Minutes of Delay**")
            r_df = mode_data["top_routes_df"]
            if r_df.empty:
                st.info("No route data.")
            else:
                fig_routes = px.bar(
                    r_df,
                    x="route_or_line",
                    y="total_delay_minutes",
                    labels={
                        "route_or_line": "Route / Line",
                        "total_delay_minutes": "Total Delay (min)",
                    },
                )
                st.plotly_chart(fig_routes, use_container_width=True)

    with mode_tab_subway:
        render_mode_panel("subway")

    with mode_tab_streetcar:
        render_mode_panel("streetcar")

    with mode_tab_bus:
        render_mode_panel("bus")

    # -------- Most delay-prone lines/routes --------
    st.subheader("Most Delay-Prone Lines / Routes")
    route_stats_sorted = (
        rel["route_stats"]
        .sort_values(["n_incidents", "mean_delay_minutes"], ascending=False)
        .head(15)
        .reset_index(drop=True)
        .copy()
    )

    st.dataframe(
        route_stats_sorted[
            [
                "mode",
                "route_or_line",
                "n_incidents",
                "mean_delay_minutes",
                "p95_delay_minutes",
                "top_code_desc",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    # -------- Trend Over Time --------
    st.subheader("Delay Trend Over Time (Daily)")
    all_routes = (
        rel["route_ts"]["route_or_line"].dropna().unique().tolist()
        if not rel["route_ts"].empty
        else []
    )
    chosen_route = st.selectbox("Select a line/route to plot:", all_routes)

    if chosen_route:
        ts = rel["route_ts"][rel["route_ts"]["route_or_line"] == chosen_route].copy()
        ts = ts.sort_values("date")
        fig_ts = px.line(
            ts,
            x="date",
            y="n_incidents",
            title=f"Daily incidents for {chosen_route}",
            markers=True,
        )
        fig_ts.update_layout(margin=dict(l=20, r=20, t=50, b=50))
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Pick a line/route to see its incident timeline.")

    # -------- Time patterns --------
    st.subheader("When Do Delays Happen?")
    col_tod, col_dow = st.columns(2)

    with col_tod:
        st.markdown("**Incidents by Hour of Day**")
        if rel["tod_profile"].empty:
            st.warning("No time-of-day profile available.")
        else:
            fig_tod = px.bar(
                rel["tod_profile"],
                x="hour",
                y="n_incidents",
                labels={"hour": "Hour", "n_incidents": "# Incidents"},
            )
            fig_tod.update_layout(height=300)
            st.plotly_chart(fig_tod, use_container_width=True)

    with col_dow:
        st.markdown("**Incidents by Day of Week**")
        if rel["dow_profile"].empty:
            st.warning("No day-of-week profile available.")
        else:
            fig_dow = px.bar(
                rel["dow_profile"],
                x="dow",
                y="n_incidents",
                labels={"dow": "Day", "n_incidents": "# Incidents"},
            )
            fig_dow.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_dow, use_container_width=True)

    # -------- Causes + Hotspots --------
    st.subheader("Where / Why")

    col_hot, col_cause = st.columns(2)

    with col_hot:
        st.markdown("**Top Delay Hotspots (Stations / Intersections)**")
        st.dataframe(
            rel["hotspots"],
            hide_index=True,
            use_container_width=True,
        )

    with col_cause:
        st.markdown("**Top Reported Causes**")
        st.dataframe(
            rel["top_causes"],
            hide_index=True,
            use_container_width=True,
        )

    # -------- Hotspot Map --------
    st.subheader("Delay Hotspot Map")
    st.markdown(
        "Circle size = incident volume. Color = avg delay length (yellow <5 min, "
        "orange 5–10, red >10)."
    )

    if rel["geo_hotspots"].empty:
        st.warning(
            "No hotspot coordinates matched. "
            "If station naming differs between delays and subway_coor.csv, "
            "tweak build_geo_hotspots() normalization."
        )
    else:
        try:
            hotspot_map = make_delay_hotspot_map(
                rel["geo_hotspots"],
                base_zoom=11,
                tiles="OpenStreetMap",
            )
            st_folium(
                hotspot_map,
                width=800,
                height=600,
                returned_objects=[],
            )
        except Exception as e:
            st.error("Hotspot map render failed.")
            st.exception(e)


def tab_data_qa_view(net: dict, rel: dict):
    """
    Render Data QA / Health tab.

    - Network QA: how clean / mappable the GTFS network data is.
    - Reliability QA: how complete the delay logs are (minutes, causes, geocoding).
    - Per-mode cause coverage: which mode is missing cause labels.
    - Unknown codes: which incident codes didn't map to a human-readable cause.
    """

    st.header("Data QA / Health Checks")
    st.markdown(
        "This tab surfaces data quality issues so we know how much to trust the visuals. "
        "It shows missing values, unmatched stations, and incident cause coverage."
    )

    # -------------------------------------------------
    # Network QA
    # -------------------------------------------------
    st.subheader("GTFS Network QA")

    qa_n = net.get("qa_network", {})
    total_stops_after_features = qa_n.get("total_stops_after_features", 0)
    total_stops_on_map = qa_n.get("total_stops_on_map", 0)
    stops_dropped_from_map = qa_n.get("stops_dropped_from_map", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Stops after feature engineering",
        total_stops_after_features,
    )
    c2.metric(
        "Stops rendered on map",
        total_stops_on_map,
    )
    c3.metric(
        "Stops dropped from map",
        stops_dropped_from_map,
        help="Likely missing / invalid lat/lon or filtered out-of-bounds.",
    )

    st.divider()

    # -------------------------------------------------
    # Reliability QA
    # -------------------------------------------------
    st.subheader("Delay / Reliability QA")

    qa_r = rel.get("qa_reliability", {})

    total_delay_rows = qa_r.get("total_delay_rows", 0)
    missing_delay_minutes = qa_r.get("missing_delay_minutes", 0)
    pct_missing_code_desc = qa_r.get("pct_missing_code_desc", 0.0)
    num_hotspots = qa_r.get("num_hotspots", 0)
    num_geo_hotspots = qa_r.get("num_geo_hotspots", 0)
    unmatched_hotspots = qa_r.get("unmatched_hotspots", 0)

    row1col1, row1col2, row1col3 = st.columns(3)
    row1col1.metric(
        "Total delay rows loaded",
        total_delay_rows,
    )
    row1col2.metric(
        "Rows missing delay_minutes",
        missing_delay_minutes,
        help="Rows where delay_minutes could not be parsed.",
    )
    row1col3.metric(
        "% rows missing cause (code_desc)",
        f"{pct_missing_code_desc}%",
        help="Higher = more incidents without a readable cause label.",
    )

    row2col1, row2col2, row2col3 = st.columns(3)
    row2col1.metric(
        "Hotspots detected",
        num_hotspots,
    )
    row2col2.metric(
        "Hotspots geocoded",
        num_geo_hotspots,
    )
    row2col3.metric(
        "Hotspots w/ no coordinates",
        unmatched_hotspots,
        help="These appear in Top Delay Hotspots but not on the hotspot map.",
    )

    # -------------------------------------------------
    # Per-mode cause coverage
    # -------------------------------------------------
    st.markdown("### Missing Cause Coverage by Mode")
    st.caption(
        "This shows how well each mode (subway / streetcar / bus) is labeled with an incident cause. "
        "If one mode has high % missing, that mode's raw file likely uses a different column name "
        "for 'incident type' that we haven't mapped yet."
    )

    import pandas as pd  # local import for this view helper

    per_mode_df = qa_r.get(
        "missing_cause_by_mode",
        pd.DataFrame(columns=["mode", "rows", "rows_missing", "pct_missing"]),
    )

    if per_mode_df.empty:
        st.info("No per-mode breakdown available.")
    else:
        # nice formatting: rename columns for display
        pretty = per_mode_df.rename(
            columns={
                "mode": "mode",
                "rows": "rows_total",
                "rows_missing": "rows_missing_cause",
                "pct_missing": "% missing cause",
            }
        )
        st.dataframe(
            pretty,
            hide_index=True,
            use_container_width=True,
        )

    # -------------------------------------------------
    # Unknown / unmapped incident codes
    # -------------------------------------------------
    st.markdown("### Unknown / Unmapped Incident Codes")
    st.caption(
        "These codes appeared in the raw delay logs but did not map to a human-readable cause. "
        "If we update the codebook (Code Descriptions.csv) OR we grab the text cause column "
        "directly from those source files, this list should shrink and the donut chart in the "
        "Reliability tab becomes more accurate."
    )

    unk_df = qa_r.get(
        "unknown_codes_df",
        pd.DataFrame(columns=["code", "n_rows", "total_delay_minutes"]),
    )

    if unk_df.empty:
        st.success("All incident codes are mapped to a description (or provided as free text).")
    else:
        st.dataframe(
            unk_df.reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

    # -------------------------------------------------
    # Debug: Delay sources detected
    # -------------------------------------------------
    st.markdown("### Delay Sources Detected (debug)")
    try:
        from ttc_analysis.delay_ingestion import get_delay_ingestion_debug

        src = get_delay_ingestion_debug()
        src_rows = []
        for group, kv in src.items():
            if isinstance(kv, dict):
                for k, v in kv.items():
                    src_rows.append({"category": group, "key": k, "path": v})
        if src_rows:
            import pandas as pd

            st.dataframe(pd.DataFrame(src_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No debug source info available yet.")
    except Exception:
        st.info("Debug source info not available.")


def tab_correlation_view(full_delays_df: pd.DataFrame):
    st.header("Correlation & Drivers (Weather / Calendar)")

    # Controls
    mode = st.selectbox("Mode", ["all", "subway", "streetcar", "bus"])
    target = st.selectbox("Target", ["n_incidents", "total_delay_minutes", "mean_delay_minutes"])
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date(2024, 1, 1))
    with col2:
        end = st.date_input("End date", value=date(2025, 12, 31))

    # 1) Hourly delays for the selected mode/date
    d = full_delays_df.copy()
    if mode != "all":
        d = d[d["mode"] == mode]

    d["delay_start_ts"] = pd.to_datetime(d["delay_start_ts"], errors="coerce")
    d = d.dropna(subset=["delay_start_ts"])
    # Align to local TZ, then floor to hour
    d["delay_start_ts"] = d["delay_start_ts"].dt.tz_localize(
        "America/Toronto", nonexistent="shift_forward", ambiguous="NaT"
    )
    d["ts_hour"] = d["delay_start_ts"].dt.floor("H")

    d = d[(d["ts_hour"].dt.date >= start) & (d["ts_hour"].dt.date <= end)].copy()
    d["delay_minutes"] = pd.to_numeric(d["delay_minutes"], errors="coerce").fillna(0)

    hourly = (
        d.groupby("ts_hour")
        .agg(n_incidents=("delay_minutes", "count"), total_delay_minutes=("delay_minutes", "sum"))
        .reset_index()
    )
    if hourly.empty:
        st.info("No delay data for the chosen filters.")
        return

    hourly["mean_delay_minutes"] = np.where(
        hourly["n_incidents"] > 0, hourly["total_delay_minutes"] / hourly["n_incidents"], 0.0
    )

    # 2) Weather via Open-Meteo (cached by month)
    wx = fetch_open_meteo_hourly_cached(start=start, end=end)
    if wx.empty:
        st.warning("Weather fetch returned no data.")
        return

    wx = wx.rename(columns={"ts": "ts_hour"})
    merged = hourly.merge(wx, on="ts_hour", how="left")

    # 3) Correlations
    feature_cols = [
        c
        for c in [
            "temp_c",
            "rel_humidity",
            "precip_mm",
            "rain_mm",
            "snow_cm",
            "wind_kph",
            "pressure_hpa",
            "is_rain",
            "is_snow",
        ]
        if c in merged.columns
    ]

    rows = []
    for f in feature_cols:
        m = merged[[f, target]].dropna()
        if len(m) >= 10:
            from scipy.stats import pearsonr, spearmanr

            pr, _ = pearsonr(m[f], m[target])
            sr, _ = spearmanr(m[f], m[target])
            rows.append({"feature": f, "pearson_r": pr, "spearman_r": sr, "n": len(m)})
        else:
            rows.append({"feature": f, "pearson_r": np.nan, "spearman_r": np.nan, "n": len(m)})
    corr_df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    st.subheader("Correlation table")
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

    # 4) Quick scatter + trendline
    pick = st.selectbox("Scatter vs feature", feature_cols)
    trendline_opt = "ols" if _HAS_STATSMODELS else None

    fig = px.scatter(
        merged,
        x=pick,
        y=target,
        trendline=trendline_opt,
        opacity=0.35,
        labels={pick: pick, target: target},
    )

    # If statsmodels is missing, add a manual OLS line so visuals stay consistent
    if not _HAS_STATSMODELS:
        xy = merged[[pick, target]].dropna()
        if len(xy) >= 2:
            x = xy[pick].astype(float).to_numpy()
            y = xy[target].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                m, b = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(x[mask].min(), x[mask].max(), 100)
                ys = m * xs + b
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name="OLS",
                        hoverinfo="skip",
                    )
                )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# MAIN STREAMLIT ENTRY
# =========================


def main():
    st.set_page_config(
        page_title="TTC Network + Reliability Dashboard",
        page_icon="🚌",
        layout="wide",
    )

    st.title("TTC Network + Reliability Dashboard")
    st.caption(
        "Static TTC network (GTFS) + observed delay incidents (2024-2025). "
        "Network tab = coverage/connectivity. "
        "Reliability tab = pain points by line, time, and location."
        "Data QA tab = data health / missing values / mapping coverage."
    )

    # Load both pipelines up front so tabs can render instantly
    net = load_static_network()
    rel = load_reliability()

    # 🔽 THIS is the part that creates the two tabs 🔽
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺 Network & Service", "⚠ Reliability / Delays", "🧪 Data QA", "📈 Correlation & Drivers"]
    )

    with tab1:
        tab_network_view(net)

    with tab2:
        tab_reliability_view(rel)

    with tab3:
        tab_data_qa_view(net, rel)
    with tab4:
        tab_correlation_view(rel["full_delays_df"])

    # footer / provenance
    st.markdown("---")
    st.caption(
        "Internal prototype. Data sources: GTFS static service data "
        "+ TTC delay incident logs (2024-2025)."
    )


if __name__ == "__main__":
    main()
