"""
viz_map.py

Generate an interactive Folium map visualizing TTC stops and their
"connectivity" (how many distinct routes serve each stop).

This module is intentionally self-contained and side-effect free except
for `save_map_html`, which writes an .html file.

Main entrypoints you care about:
- build_connectivity_map_df(stop_features_df)  -> sanitized dataframe
- make_stop_connectivity_map(df, **opts)       -> Folium map object
- save_map_html(map_obj, out_path)             -> Path

The pipeline in run_analysis.py should be:
    df_clean = build_connectivity_map_df(stop_features_df)
    fmap     = make_stop_connectivity_map(df_clean)
    save_map_html(fmap, "map_stop_connectivity.html")

Design goals:
- Robust against weird rows (NaN lat/lon, negative n_routes, etc.)
- Visual encodes importance using both color + marker radius
- Export is a standalone HTML you can open in any browser
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd


def build_connectivity_map_df(stop_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the stop_features_df for mapping.

    Expected input columns in stop_features_df:
        stop_id      : string-like identifier for the stop
        name         : human-readable stop name
        lat          : latitude (float, WGS84)
        lon          : longitude (float, WGS84)
        n_routes     : int (# distinct routes serving this stop)

    Returns
    -------
    pd.DataFrame
        A copy with:
        - only the required columns
        - rows filtered to valid coordinate + numeric n_routes
        - n_routes coerced to integer >= 0
        - no duplicate stop_id rows (first occurrence kept)

    Notes
    -----
    We do *not* group by stop_id here; upstream feature engineering should
    already have one row per stop. We just guard against garbage because
    GTFS feeds can be messy.
    """
    required_cols = ["stop_id", "name", "lat", "lon", "n_routes"]
    missing = [c for c in required_cols if c not in stop_features_df.columns]
    if missing:
        raise ValueError(f"stop_features_df is missing required columns: {missing}")

    df = stop_features_df[required_cols].copy()

    # Drop obvious coordinate junk: lat/lon null or wildly out of Toronto-ish bounds
    # Toronto-ish bounding box: lat ~ 43 to 44, lon ~ -80 to -78.5
    # We keep this loose to avoid accidentally throwing out legitimate outliers.
    df = df[
        df["lat"].notna()
        & df["lon"].notna()
        & (df["lat"].between(40.0, 50.0))
        & (df["lon"].between(-90.0, -70.0))
    ].copy()

    # n_routes -> integer >= 0
    df["n_routes"] = pd.to_numeric(df["n_routes"], errors="coerce").fillna(0)
    df.loc[df["n_routes"] < 0, "n_routes"] = 0
    df["n_routes"] = df["n_routes"].astype(int)

    # Ensure stop_id is string, name is string (avoid numpy types in popup HTML)
    df["stop_id"] = df["stop_id"].astype("string")
    df["name"] = df["name"].astype("string")

    # Deduplicate just in case (keep first occurrence)
    df = df.drop_duplicates(subset=["stop_id"]).reset_index(drop=True)

    return df


def _compute_radius_scale(
    n_routes_series: pd.Series, min_radius: float = 3.0, max_radius: float = 12.0
) -> pd.Series:
    """
    Map n_routes -> circle marker radius.

    We normalize linearly between min(n_routes) and max(n_routes) so the
    visualization is adaptive to different cities / feeds, then clamp
    into [min_radius, max_radius].

    If all stops have the same n_routes (flat network), everyone just
    gets min_radius.
    """
    n = n_routes_series.to_numpy(dtype=float)

    if n.size == 0:
        return pd.Series(dtype=float)

    n_min = np.nanmin(n)
    n_max = np.nanmax(n)

    if not np.isfinite(n_min):
        n_min = 0.0
    if not np.isfinite(n_max):
        n_max = n_min

    # Avoid divide-by-zero if all stops identical
    denom = (n_max - n_min) if (n_max - n_min) != 0 else 1.0

    scaled_0_1 = (n - n_min) / denom  # in [0,1]
    scaled_radius = min_radius + scaled_0_1 * (max_radius - min_radius)

    # Safety clamp
    scaled_radius = np.clip(scaled_radius, min_radius, max_radius)

    return pd.Series(scaled_radius, index=n_routes_series.index)


def _color_from_n_routes(n: int) -> str:
    """
    Convert connectivity to a qualitative color category.
    These are standard web color names that Folium/Leaflet understands.

    Simple rule:
        0 routes  -> "gray"     (shouldn't happen if data is good)
        1-2       -> "yellow"
        3-4       -> "orange"
        >=5       -> "red"

    You can tune these buckets later with domain logic
    (e.g. streetcar hubs vs subway hubs).
    """
    if n >= 5:
        return "red"
    elif n >= 3:
        return "orange"
    elif n >= 1:
        return "yellow"
    return "gray"


def _add_static_legend(fmap: folium.Map) -> None:
    """
    Inject a basic HTML legend into bottom-left of the Folium map.

    Folium doesn't ship a built-in categorical legend helper, so we
    add a positioned <div>. This is lightweight and works offline.
    """
    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.9);
        padding: 8px 12px;
        border: 1px solid #888;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.4;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
      <div style="font-weight:600; margin-bottom:4px;">
        Stop Connectivity
      </div>
      <div><span style="display:inline-block;width:10px;height:10px;
            background:red;border-radius:50%;margin-right:6px;"></span>
           5+ routes</div>
      <div><span style="display:inline-block;width:10px;height:10px;
            background:orange;border-radius:50%;margin-right:6px;"></span>
           3-4 routes</div>
      <div><span style="display:inline-block;width:10px;height:10px;
            background:yellow;border-radius:50%;margin-right:6px;
            border:1px solid #aaa;"></span>
           1-2 routes</div>
      <div><span style="display:inline-block;width:10px;height:10px;
            background:gray;border-radius:50%;margin-right:6px;"></span>
           0 routes / unknown</div>
    </div>
    """
    # Folium's root object exposes an .html container at runtime; typing is loose
    fmap.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]


def _toronto_center(df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute a reasonable map center.
    We'll take the median lat/lon of all stops.
    If df is empty (shouldn't happen in normal TTC feeds), fall back
    to downtown Toronto coords.
    """
    if len(df) == 0:
        return (43.6532, -79.3832)  # fallback: Toronto City Hall-ish

    lat_med = float(df["lat"].median())
    lon_med = float(df["lon"].median())

    # sanity clamp in case of weird outliers
    if not np.isfinite(lat_med):
        lat_med = 43.6532
    if not np.isfinite(lon_med):
        lon_med = -79.3832

    return (lat_med, lon_med)


def make_stop_connectivity_map(
    df_clean: pd.DataFrame,
    base_zoom: int = 11,
    tiles: str = "OpenStreetMap",
    min_radius: float = 3.0,
    max_radius: float = 12.0,
) -> folium.Map:
    """
    Build a Folium map with one circle marker per TTC stop.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Output of build_connectivity_map_df.
        Must have columns:
            stop_id (string),
            name (string),
            lat (float),
            lon (float),
            n_routes (int)
    base_zoom : int
        Initial zoom level for the map.
    tiles : str
        Basemap tiles for Folium. "OpenStreetMap" is default & free.
        Others (e.g. "CartoDB positron") are also common.
    min_radius, max_radius : float
        Size bounds (pixels) for the circle markers. The final radius
        is computed per-stop using a linear scale over n_routes.

    Returns
    -------
    folium.Map
        An in-memory Folium map object. You still need to call
        save_map_html() to write it to disk.

    Behavior
    --------
    - Marker color encodes connectivity class.
    - Marker radius scales with n_routes relative to dataset min/max.
    - Each marker has a popup with stop info, for quick inspection.
    - A small HTML legend is injected into bottom-left of the map.
    """
    required_cols = ["stop_id", "name", "lat", "lon", "n_routes"]
    for col in required_cols:
        if col not in df_clean.columns:
            raise ValueError(
                f"df_clean is missing required column '{col}'. "
                "Did you forget to call build_connectivity_map_df()?"
            )

    if len(df_clean) == 0:
        # Empty dataset → still build a map so downstream code doesn't explode.
        center_lat, center_lon = (43.6532, -79.3832)
    else:
        center_lat, center_lon = _toronto_center(df_clean)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=base_zoom,
        tiles=tiles,
    )

    # Compute dynamic radius scale for all stops
    radii = _compute_radius_scale(
        df_clean["n_routes"],
        min_radius=min_radius,
        max_radius=max_radius,
    )

    # Iterate once with `itertuples` for speed / clarity
    for i, row in enumerate(df_clean.itertuples(index=False), start=0):
        lat = float(row.lat)
        lon = float(row.lon)
        n_routes_val = int(row.n_routes)
        stop_id_val = str(row.stop_id)
        name_val = str(row.name)

        # popup HTML for each stop
        popup_html = (
            f"<b>{name_val}</b><br>"
            f"Stop ID: {stop_id_val}<br>"
            f"Distinct routes serving stop: {n_routes_val}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=float(radii.iloc[i]),
            color=_color_from_n_routes(n_routes_val),
            fill=True,
            fill_color=_color_from_n_routes(n_routes_val),
            fill_opacity=0.85,
            weight=1,
            popup=popup_html,
        ).add_to(fmap)

    _add_static_legend(fmap)
    return fmap


def save_map_html(fmap: folium.Map, out_path: str = "map_stop_connectivity.html") -> Path:
    """
    Write a Folium map object to disk as a standalone HTML file.

    Parameters
    ----------
    fmap : folium.Map
        The object returned by make_stop_connectivity_map().
    out_path : str
        Desired file path for the output HTML. Relative paths are
        resolved against the working directory of the caller.

    Returns
    -------
    pathlib.Path
        Absolute path to the written file. You can print it so the user
        can click/open it.

    Notes
    -----
    The output HTML is fully self-contained (Leaflet + markers are
    embedded). You can send this file to someone else and they can open
    it locally in a browser without running Python.
    """
    out_file = Path(out_path).resolve()
    fmap.save(str(out_file))
    return out_file
