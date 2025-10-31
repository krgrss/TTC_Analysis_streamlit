"""
run_analysis.py

End-to-end batch pipeline:
1. Load & preprocess GTFS data (ingestion).
2. Derive analytics features (feature engineering).
3. Generate summary tables (printed to console for now).
4. Generate an interactive connectivity map (HTML output).

Recommended:
    python -m ttc_analysis.run_analysis

Artifacts:
- Console output (top connected stops, route summary preview)
- map_stop_connectivity.html in the project root (or cwd)
"""

from ttc_analysis.delay_features import (
    build_route_delay_stats,
    build_time_of_day_profile,
    build_top_causes,
)
from ttc_analysis.delay_ingestion import load_all_delay_events
from ttc_analysis.features import build_route_features, build_stop_features
from ttc_analysis.ingestion import load_all
from ttc_analysis.viz_map import (
    build_connectivity_map_df,
    make_stop_connectivity_map,
    save_map_html,
)


def main() -> None:
    # 1. Load + preprocess all relevant GTFS tables
    data = load_all()
    stops_df = data["stops"]
    routes_df = data["routes"]
    calendar_df = data["calendar"]
    trips_df = data["trips"]
    stop_times_df = data["stop_times"]

    # 2. Feature engineering
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

    # 3. Basic summaries (terminal-friendly analytics for sanity check)
    top_connected = (
        stop_features_df.sort_values("n_routes", ascending=False).head(10).reset_index(drop=True)
    )

    print("=== Top 10 most connected stops (by n_routes) ===")
    print(top_connected)

    print("\n=== Route service summary (first 10 rows) ===")
    print(route_features_df.head(10).reset_index(drop=True))

    # 4. Map generation
    #    4a. sanitize/validate for mapping
    map_df = build_connectivity_map_df(stop_features_df)

    #    4b. build folium map in memory
    folium_map = make_stop_connectivity_map(
        map_df,
        base_zoom=11,
        tiles="OpenStreetMap",  # can switch to "CartoDB positron" if you like
        min_radius=3.0,
        max_radius=12.0,
    )

    #    4c. export HTML
    out_file = save_map_html(
        folium_map,
        out_path="map_stop_connectivity.html",
    )

    print(f"\n[OK] Map written to: {out_file}")
    print("Open that file in a browser to explore TTC stop connectivity.")

    delays = load_all_delay_events()

    route_delay_stats = build_route_delay_stats(delays)
    tod_profile = build_time_of_day_profile(delays)
    top_causes = build_top_causes(delays, k=10)

    print("\n=== Delay Hotspots by Route/Line ===")
    print(
        route_delay_stats.sort_values(["n_incidents", "mean_delay_minutes"], ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    print("\n=== Time-of-Day Profile (system-wide) ===")
    print(tod_profile)

    print("\n=== Top Delay Causes ===")
    print(top_causes)


if __name__ == "__main__":
    main()
