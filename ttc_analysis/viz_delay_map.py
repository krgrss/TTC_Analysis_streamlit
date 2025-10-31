"""
viz_delay_map.py

Map high-delay hotspots (stations/intersections) using Folium.
Marker size/color reflects incident volume and/or severity.
"""

from __future__ import annotations

import folium
import numpy as np
import pandas as pd


def _size_from_incidents(n_incidents: pd.Series, min_radius=4.0, max_radius=16.0) -> pd.Series:
    if n_incidents.empty:
        return pd.Series([], dtype=float)
    vals = n_incidents.to_numpy(dtype=float)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    denom = (vmax - vmin) if vmax != vmin else 1.0
    scaled = (vals - vmin) / denom
    rad = min_radius + scaled * (max_radius - min_radius)
    return pd.Series(np.clip(rad, min_radius, max_radius), index=n_incidents.index)


def _color_from_delay(mins: float) -> str:
    """
    Encode avg delay severity with color.
    0-5 min   -> 'yellow'
    5-10 min  -> 'orange'
    >10 min   -> 'red'
    """
    if mins is None or np.isnan(mins):
        return "gray"
    if mins > 10:
        return "red"
    elif mins > 5:
        return "orange"
    else:
        return "yellow"


def make_delay_hotspot_map(
    geo_hotspots_df: pd.DataFrame, base_zoom: int = 11, tiles: str = "OpenStreetMap"
) -> folium.Map:
    """
    Build a Folium map where each marker is a high-delay location.

    Required columns in geo_hotspots_df:
        location
        lat
        lon
        n_incidents
        mean_delay_minutes
    """

    required = [
        "location",
        "lat",
        "lon",
        "n_incidents",
        "mean_delay_minutes",
    ]
    for col in required:
        if col not in geo_hotspots_df.columns:
            raise ValueError(f"geo_hotspots_df missing required column {col}")

    df = geo_hotspots_df.copy()
    df = df[
        df["lat"].notna()
        & df["lon"].notna()
        & df["lat"].between(40.0, 50.0)
        & df["lon"].between(-90.0, -70.0)
    ].reset_index(drop=True)

    if len(df) == 0:
        # Fallback center: downtown Toronto
        center_lat, center_lon = 43.6532, -79.3832
    else:
        center_lat = float(df["lat"].median())
        center_lon = float(df["lon"].median())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=base_zoom,
        tiles=tiles,
    )

    radii = _size_from_incidents(df["n_incidents"])

    for i, row in df.iterrows():
        popup_html = (
            f"<b>{row['location']}</b><br>"
            f"Incidents: {row['n_incidents']}<br>"
            f"Avg delay: {row['mean_delay_minutes']} min"
        )

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=float(radii.iloc[i]),
            color=_color_from_delay(float(row["mean_delay_minutes"])),
            fill=True,
            fill_color=_color_from_delay(float(row["mean_delay_minutes"])),
            fill_opacity=0.85,
            weight=1,
            popup=popup_html,
        ).add_to(fmap)

    # Add legend
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
        Delay Hotspots
      </div>
      <div>Marker size = incident volume</div>
      <div>Marker color = avg delay length</div>
      <div style="margin-top:6px;">
        <span style="display:inline-block;width:10px;height:10px;
            background:yellow;border-radius:50%;margin-right:6px;
            border:1px solid #aaa;"></span>
        &lt;=5 min avg
      </div>
      <div>
        <span style="display:inline-block;width:10px;height:10px;
            background:orange;border-radius:50%;margin-right:6px;"></span>
        5-10 min avg
      </div>
      <div>
        <span style="display:inline-block;width:10px;height:10px;
            background:red;border-radius:50%;margin-right:6px;"></span>
        &gt;10 min avg
      </div>
    </div>
    """
    # Typing: Folium root has a dynamic .html attribute
    fmap.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]

    return fmap
