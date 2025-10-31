"""
QA routes — Data Quality & Health

Purpose:
- Provide endpoints for data quality dashboards: freshness, completeness, uniqueness, schema drift, and geo issues.

Security:
- Returns only sanitized text; no HTML. All numeric fields are validated.
"""
from __future__ import annotations

from typing import List, Optional, Literal
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

try:
    import pandas as pd  # type: ignore
    from ttc_analysis.delay_ingestion import load_station_coordinates
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    load_station_coordinates = None  # type: ignore


router = APIRouter(prefix="/qa", tags=["qa"])


class QaSummary(BaseModel):
    dataset: str
    updated_at: str
    freshness_min: int = Field(ge=0)
    row_count: int = Field(ge=0)
    missing_pct: float = Field(ge=0.0)
    duplicate_keys: int = Field(ge=0)
    schema_version: str
    drift_detected: bool


class Anomaly(BaseModel):
    id: str
    dataset: str
    severity: Literal['low', 'medium', 'high']
    category: Literal['freshness', 'completeness', 'uniqueness', 'schema', 'bounds']
    message: str
    first_seen: str
    last_seen: str
    rows_affected: Optional[int] = None


class ColumnStat(BaseModel):
    column: str
    type: str
    null_pct: float = Field(ge=0.0, le=100.0)
    distinct_count: int = Field(ge=0)
    min: Optional[str | float] = None
    max: Optional[str | float] = None
    examples: Optional[List[str]] = None


class GeoIssue(BaseModel):
    id: str
    dataset: Literal['stops', 'routes']
    issue: Literal['invalid_coord', 'duplicate_coord', 'out_of_bbox']
    lng: float
    lat: float
    count: Optional[int] = None
    details: Optional[str] = None


@router.get("/summary", response_model=List[QaSummary])
def qa_summary() -> List[QaSummary]:
    """Return summary rows per dataset. Currently derives minimal info from available sources.

    - `stops`: derived from station coordinates if available.
    - Other datasets: placeholder entries omitted until backend sources are wired.
    """
    out: List[QaSummary] = []
    now = datetime.now(timezone.utc)

    if load_station_coordinates is not None and pd is not None:
        try:
            df = load_station_coordinates()
            rc = int(len(df)) if df is not None else 0
        except Exception:
            rc = 0
        out.append(
            QaSummary(
                dataset="stops",
                updated_at=now.isoformat(),
                freshness_min=0,
                row_count=rc,
                missing_pct=0.0,
                duplicate_keys=0,
                schema_version="n/a",
                drift_detected=False,
            )
        )

    return out


@router.get("/anomalies", response_model=List[Anomaly])
def qa_anomalies(
    dataset: str = Query("stops"),
    severity: Literal['all', 'high', 'medium', 'low'] = Query("all"),
    from_: Optional[str] = Query(None, alias="from"),
) -> List[Anomaly]:
    """Return anomalies for the selected dataset and time window.

    Placeholder returns an empty list until quality checks are wired.
    """
    return []


@router.get("/columns/{dataset}", response_model=List[ColumnStat])
def qa_columns(dataset: Literal['stops', 'trips', 'incidents']) -> List[ColumnStat]:
    """Return per-column statistics for the dataset.

    - For `stops`, derive from station coordinates if available.
    - Otherwise, return an empty list until dataset stats are wired.
    """
    if dataset == 'stops' and load_station_coordinates is not None and pd is not None:
        try:
            df = load_station_coordinates()
            if df is None or df.empty:  # type: ignore[attr-defined]
                return []
            out: List[ColumnStat] = []
            for col in df.columns:  # type: ignore[attr-defined]
                series = df[col]
                null_pct = float(series.isna().mean() * 100.0)
                distinct = int(series.nunique(dropna=True))
                ex: List[str] = []
                for v in series.dropna().astype(str).unique()[:3]:  # type: ignore[attr-defined]
                    ex.append(str(v))
                ctype = str(series.dtype)
                min_v = series.min() if hasattr(series, 'min') else None
                max_v = series.max() if hasattr(series, 'max') else None
                out.append(
                    ColumnStat(
                        column=str(col),
                        type=ctype,
                        null_pct=round(null_pct, 2),
                        distinct_count=distinct,
                        min=(float(min_v) if isinstance(min_v, (int, float)) else str(min_v) if min_v is not None else None),
                        max=(float(max_v) if isinstance(max_v, (int, float)) else str(max_v) if max_v is not None else None),
                        examples=ex or None,
                    )
                )
            return out
        except Exception:
            return []
    return []


@router.get("/geo/issues", response_model=List[GeoIssue])
def qa_geo_issues() -> List[GeoIssue]:
    """Return geo issues; placeholder returns empty until checks are wired."""
    return []

