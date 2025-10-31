"""
correlation.py

External-factor correlation for TTC delay analytics.

This module keeps I/O minimal and robust:
- Weather loader: expects a CSV in data/ (or Data/) with date and a
  mean temperature column; header names are detected via fuzzy rules.
- Special days loader: optional CSV with date and label.

Public API:
- load_weather_daily() -> DataFrame[date, temp_mean_c]
- load_special_days() -> DataFrame[date, label]
- build_delay_external_correlations(delays, weather_df, special_days_df)

All functions are pure transforms except file I/O in the loaders.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# -----------------------------
# Internal helpers
# -----------------------------


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower().str.replace("[^a-z0-9_]+", "_", regex=True)
    return out


def _data_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    d = root / "data"
    if d.exists():
        return d
    d2 = root / "Data"
    return d2 if d2.exists() else d


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_fuzzy(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        name = col.lower()
        if all(kw in name for kw in keywords):
            return col
    return None


# -----------------------------
# Loaders
# -----------------------------


def load_weather_daily() -> pd.DataFrame:
    """
    Load a daily weather CSV from data/ (or Data/).

    Expected columns (fuzzy-detected):
    - date: date, datetime, day
    - temp_mean_c: tavg, temp_mean, avg_temperature, temperature_mean, temp_c

    Returns a DataFrame with columns:
        date (datetime64[ns])
        temp_mean_c (float)

    If no suitable file exists, returns an empty frame with the same columns.
    """
    data_dir = _data_dir()
    # try common filenames first
    candidates = [
        data_dir / "weather_daily.csv",
        data_dir / "daily_weather.csv",
    ]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        # try to discover a reasonable file
        for p in data_dir.glob("*.csv"):
            n = p.name.lower()
            if "weather" in n and ("daily" in n or "day" in n):
                path = p
                break

    if path is None or not path.exists():
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "temp_mean_c": pd.Series(dtype="float"),
            }
        )

    df = pd.read_csv(path, dtype="string", encoding="utf-8-sig", low_memory=False)
    df = _normalize_headers(df)

    date_col = (
        _pick(df, ["date", "day", "datetime"])
        or _pick_fuzzy(df, ["date"])
        or _pick_fuzzy(df, ["day"])
    )  # noqa: E501
    t_col = (
        _pick(
            df,
            [
                "temp_mean_c",
                "tavg",
                "avg_temperature",
                "avg_temperature_c",
                "temperature_mean",
                "temp_c",
            ],
        )
        or _pick_fuzzy(df, ["temp", "mean"])
        or _pick_fuzzy(df, ["tavg"])
        or _pick_fuzzy(df, ["avg", "temp"])  # noqa: E501
    )

    if date_col is None or t_col is None:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "temp_mean_c": pd.Series(dtype="float"),
            }
        )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "temp_mean_c": pd.to_numeric(df[t_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"]).copy()
    out["date"] = out["date"].dt.normalize()
    return out.reset_index(drop=True)


def load_special_days() -> pd.DataFrame:
    """
    Optional special days file: data/special_days.csv.

    Expected columns (fuzzy-detected):
      - date
      - label or name or type

    Returns DataFrame with columns [date, label]. If not found, returns empty frame.
    """
    data_dir = _data_dir()
    path = data_dir / "special_days.csv"
    if not path.exists():
        # allow discovery by pattern
        for p in data_dir.glob("*special*days*.csv"):
            path = p
            break
        else:
            return pd.DataFrame(
                {
                    "date": pd.Series(dtype="datetime64[ns]"),
                    "label": pd.Series(dtype="string"),
                }
            )

    df = pd.read_csv(path, dtype="string", encoding="utf-8-sig", low_memory=False)
    df = _normalize_headers(df)
    date_col = _pick(df, ["date"]) or _pick_fuzzy(df, ["date"]) or _pick(df, ["day"])  # noqa: E501
    label_col = (
        _pick(df, ["label", "name", "type"])
        or _pick_fuzzy(df, ["label"])
        or _pick_fuzzy(df, ["name"])
    )  # noqa: E501

    if date_col is None:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "label": pd.Series(dtype="string"),
            }
        )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "label": (
                df[label_col] if label_col else pd.Series(pd.NA, index=df.index, dtype="string")
            )
            .astype("string")
            .str.strip(),  # noqa: E501
        }
    )
    out = out.dropna(subset=["date"]).copy()
    out["date"] = out["date"].dt.normalize()
    return out.reset_index(drop=True)


# -----------------------------
# Correlation builder
# -----------------------------


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Fall"


def build_delay_external_correlations(
    delays: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    special_days_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute correlations between daily total TTC delay minutes and external factors.

    Inputs:
      delays: standardized delay rows (see delay_ingestion.load_all_delay_events)
      weather_df: DataFrame with columns [date, temp_mean_c] (optional)
      special_days_df: DataFrame with columns [date, label] (optional)

    Returns dict of DataFrames:
      - system_weather_corr: pearson r and n_days across all modes
      - weather_corr_by_mode: r per mode
      - seasonal_averages_by_mode: mean total delay per season per mode
      - special_day_effect_by_mode: mean total delay on special vs non-special days
      - daily_system: daily totals (system) with merges (for plotting)
      - daily_by_mode: daily totals by mode with merges
    """
    if delays.empty:
        empty = pd.DataFrame()
        return {
            "system_weather_corr": empty,
            "weather_corr_by_mode": empty,
            "seasonal_averages_by_mode": empty,
            "special_day_effect_by_mode": empty,
            "daily_system": empty,
            "daily_by_mode": empty,
        }

    d = delays.copy()
    d["date"] = pd.to_datetime(d["delay_start_ts"], errors="coerce").dt.normalize()
    d = d[d["date"].notna()].copy()

    daily_sys = (
        d.groupby("date")
        .agg(
            n_incidents=("delay_minutes", "count"),
            total_delay_minutes=("delay_minutes", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily_mode = (
        d.groupby(["mode", "date"])  # type: ignore[index]
        .agg(
            n_incidents=("delay_minutes", "count"),
            total_delay_minutes=("delay_minutes", "sum"),
        )
        .reset_index()
        .sort_values(["mode", "date"])  # type: ignore[index]
    )

    # Merge weather, if available
    w = weather_df if weather_df is not None else pd.DataFrame()
    if not w.empty and set(["date", "temp_mean_c"]).issubset(w.columns):
        daily_sys = daily_sys.merge(w, on="date", how="left")
        daily_mode = daily_mode.merge(w, on="date", how="left")

    # Add season for seasonal breakdowns
    daily_mode["season"] = daily_mode["date"].dt.month.map(_season_from_month)

    # Merge special days
    sd = special_days_df if special_days_df is not None else pd.DataFrame()
    if not sd.empty and "date" in sd.columns:
        # reduce to unique date->is_special, keep a representative label
        sd2 = sd.copy()
        sd2["label"] = sd2["label"].astype("string").str.strip()
        sd2 = (
            sd2.groupby("date")
            .agg(label=("label", lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA))
            .reset_index()
        )
        daily_mode = daily_mode.merge(sd2, on="date", how="left")
        daily_mode["is_special_day"] = daily_mode["label"].notna()
    else:
        daily_mode["is_special_day"] = False

    # Correlations
    if "temp_mean_c" in daily_sys.columns:
        sys_r = daily_sys[["total_delay_minutes", "temp_mean_c"]].dropna()
        r_val = (
            float(sys_r["total_delay_minutes"].corr(sys_r["temp_mean_c"]))
            if not sys_r.empty
            else float("nan")
        )
        system_corr = pd.DataFrame(
            {
                "pearson_r": [r_val],
                "n_days": [len(sys_r)],
            }
        )
    else:
        system_corr = pd.DataFrame(
            {"pearson_r": pd.Series(dtype="float"), "n_days": pd.Series(dtype="int")}
        )

    if "temp_mean_c" in daily_mode.columns:
        rows = []
        for mode, sub in daily_mode.groupby("mode"):
            sub2 = sub[["total_delay_minutes", "temp_mean_c"]].dropna()
            r = (
                float(sub2["total_delay_minutes"].corr(sub2["temp_mean_c"]))
                if not sub2.empty
                else float("nan")
            )
            rows.append({"mode": mode, "pearson_r": r, "n_days": int(len(sub2))})
        mode_corr = pd.DataFrame(rows)
    else:
        mode_corr = pd.DataFrame(columns=["mode", "pearson_r", "n_days"])

    # Seasonal averages
    seasonal = (
        daily_mode.groupby(["mode", "season"])  # type: ignore[index]
        .agg(
            mean_total_delay_minutes=("total_delay_minutes", "mean"),
            mean_n_incidents=("n_incidents", "mean"),
        )
        .reset_index()
        .sort_values(["mode", "season"])  # type: ignore[index]
    )

    # Special-day effect
    if "is_special_day" in daily_mode.columns:
        rows = []
        for mode, sub in daily_mode.groupby("mode"):
            if sub.empty:
                continue
            base = sub[~sub["is_special_day"]]
            base_mean = (
                float(base["total_delay_minutes"].mean()) if not base.empty else float("nan")
            )
            spec = sub[sub["is_special_day"]]
            if not spec.empty:
                mean_special = float(spec["total_delay_minutes"].mean())
                diff = mean_special - base_mean if pd.notna(base_mean) else float("nan")
                pct = (
                    (diff / base_mean * 100.0)
                    if pd.notna(base_mean) and base_mean != 0
                    else float("nan")
                )
                rows.append(
                    {
                        "mode": mode,
                        "mean_total_delay_special": mean_special,
                        "baseline_mean_total_delay": base_mean,
                        "diff": diff,
                        "pct_diff": pct,
                    }
                )
        special_effect = pd.DataFrame(rows)
    else:
        special_effect = pd.DataFrame(
            columns=[
                "mode",
                "mean_total_delay_special",
                "baseline_mean_total_delay",
                "diff",
                "pct_diff",
            ]
        )

    return {
        "system_weather_corr": system_corr,
        "weather_corr_by_mode": mode_corr,
        "seasonal_averages_by_mode": seasonal,
        "special_day_effect_by_mode": special_effect,
        "daily_system": daily_sys,
        "daily_by_mode": daily_mode,
    }
