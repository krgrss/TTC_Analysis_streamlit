import pandas as pd


def build_stop_features(
    stops_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    trips_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Output:
      stop_id
      name
      lat
      lon
      n_routes  (# distinct routes serving this stop)
    """

    st_times = stop_times_df.copy()
    trips = trips_df.copy()

    # normalize join keys (defensive)
    st_times["trip_id"] = st_times["trip_id"].astype("string")
    st_times["stop_id"] = st_times["stop_id"].astype("string")
    trips["trip_id"] = trips["trip_id"].astype("string")
    trips["route_id"] = trips["route_id"].astype("string")

    # join to attach route_id to each (trip_id, stop_id)
    st_join = st_times.merge(
        trips[["trip_id", "route_id"]],
        on="trip_id",
        how="left",
    )

    # count distinct routes per stop
    routes_per_stop = (
        st_join.groupby("stop_id")["route_id"]
        .nunique(dropna=True)
        .reset_index()
        .rename(columns={"route_id": "n_routes"})
    )

    # attach stop metadata
    out = stops_df.copy()
    out["stop_id"] = out["stop_id"].astype("string")

    out = out.merge(routes_per_stop, on="stop_id", how="left")
    out["n_routes"] = out["n_routes"].fillna(0).astype(int)

    return out[["stop_id", "name", "lat", "lon", "n_routes"]].copy()


def build_route_features(
    routes_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Output:
      route_id
      short_name
      long_name
      mode
      runs_weekday   (bool)
      runs_weekend   (bool)
      start_date_min (datetime)
      end_date_max   (datetime)
    """

    trips_local = trips_df.copy()
    cal_local = calendar_df.copy()
    r_local = routes_df.copy()

    # normalize join keys
    trips_local["route_id"] = trips_local["route_id"].astype("string")
    trips_local["service_id"] = trips_local["service_id"].astype("string")
    cal_local["service_id"] = cal_local["service_id"].astype("string")
    r_local["route_id"] = r_local["route_id"].astype("string")

    # unique (route_id, service_id) combos
    r_svc = trips_local[["route_id", "service_id"]].drop_duplicates()

    # attach calendar info to each (route_id, service_id)
    rsvc_cal = r_svc.merge(
        cal_local[
            [
                "service_id",
                "runs_weekday",
                "runs_weekend",
                "start_date",
                "end_date",
            ]
        ],
        on="service_id",
        how="left",
    )

    # aggregate per route_id
    agg = (
        rsvc_cal.groupby("route_id")
        .agg(
            runs_weekday=("runs_weekday", "max"),
            runs_weekend=("runs_weekend", "max"),
            start_date_min=("start_date", "min"),
            end_date_max=("end_date", "max"),
        )
        .reset_index()
    )

    # join route metadata
    out = r_local.merge(agg, on="route_id", how="left")

    # fill missing booleans with False
    out["runs_weekday"] = out["runs_weekday"].fillna(False)
    out["runs_weekend"] = out["runs_weekend"].fillna(False)

    return out[
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
    ].copy()
