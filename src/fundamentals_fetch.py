"""Fetch quarterly earnings and fundamental metrics from Finnhub."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from finnhub import Client as FinnhubClient

from src.sentiment_cache import CACHE_DIR

# Calendar days after fiscal quarter-end before features use the report (conservative).
DEFAULT_REPORT_LAG_DAYS = 45

_QUARTERLY_METRICS = (
    "eps",
    "netMargin",
    "grossMargin",
    "operatingMargin",
    "roeTTM",
    "totalDebtToEquity",
)


def _finnhub_client(api_key: str | None = None) -> FinnhubClient | None:
    key = api_key if api_key is not None else os.environ.get("FINNHUB_API_KEY", "dummy")
    if not key or key in ("dummy", "your_key"):
        return None
    try:
        return FinnhubClient(api_key=key)
    except Exception:
        return None


def _earnings_cache_path(ticker: str) -> Path:
    d = CACHE_DIR / ticker.strip().upper()
    d.mkdir(parents=True, exist_ok=True)
    return d / "quarterly_earnings.parquet"


def _metrics_cache_path(ticker: str) -> Path:
    d = CACHE_DIR / ticker.strip().upper()
    d.mkdir(parents=True, exist_ok=True)
    return d / "quarterly_metrics_long.parquet"


def fetch_finnhub_quarterly_earnings(
    ticker: str,
    *,
    limit: int = 40,
    api_key: str | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Quarterly EPS surprises from Finnhub ``company_earnings``.

    Columns: ``period_end``, ``eps_actual``, ``eps_estimate``, ``surprise``,
    ``surprise_pct``, ``quarter``, ``year``.
    """
    sym = ticker.strip().upper()
    path = _earnings_cache_path(sym)
    if use_cache and path.is_file():
        cached = pl.read_parquet(path)
        if cached.height > 0:
            return cached

    client = _finnhub_client(api_key)
    schema = {
        "period_end": pl.Date,
        "eps_actual": pl.Float64,
        "eps_estimate": pl.Float64,
        "surprise": pl.Float64,
        "surprise_pct": pl.Float64,
        "quarter": pl.Int32,
        "year": pl.Int32,
    }
    if client is None:
        return pl.DataFrame(schema=schema)

    try:
        raw: list = client.company_earnings(sym, limit=limit) or []
    except Exception:
        raw = []

    rows: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        period = item.get("period")
        if not period:
            continue
        try:
            period_end = date.fromisoformat(str(period)[:10])
        except ValueError:
            continue
        rows.append(
            {
                "period_end": period_end,
                "eps_actual": _to_float(item.get("actual")),
                "eps_estimate": _to_float(item.get("estimate")),
                "surprise": _to_float(item.get("surprise")),
                "surprise_pct": _to_float(item.get("surprisePercent")),
                "quarter": int(item["quarter"]) if item.get("quarter") is not None else None,
                "year": int(item["year"]) if item.get("year") is not None else None,
            }
        )

    if not rows:
        return pl.DataFrame(schema=schema)

    out = pl.DataFrame(rows).unique(subset=["period_end"]).sort("period_end")
    if use_cache:
        out.write_parquet(path)
    return out


def fetch_finnhub_quarterly_metrics(
    ticker: str,
    *,
    api_key: str | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Long-format quarterly ratios from Finnhub ``company_basic_financials`` (``series.quarterly``).

    Columns: ``period_end``, ``metric``, ``value``.
    """
    sym = ticker.strip().upper()
    path = _metrics_cache_path(sym)
    if use_cache and path.is_file():
        return pl.read_parquet(path)

    client = _finnhub_client(api_key)
    schema = {"period_end": pl.Date, "metric": pl.Utf8, "value": pl.Float64}
    if client is None:
        return pl.DataFrame(schema=schema)

    try:
        raw = client.company_basic_financials(sym, "all")
    except Exception:
        return pl.DataFrame(schema=schema)

    quarterly = {}
    if isinstance(raw, dict):
        series = raw.get("series") or {}
        if isinstance(series, dict):
            quarterly = series.get("quarterly") or {}

    rows: list[dict] = []
    if isinstance(quarterly, dict):
        for metric_name in _QUARTERLY_METRICS:
            points = quarterly.get(metric_name)
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, dict):
                    continue
                period = point.get("period")
                value = point.get("v")
                if period is None or value is None:
                    continue
                try:
                    period_end = date.fromisoformat(str(period)[:10])
                except ValueError:
                    continue
                rows.append(
                    {
                        "period_end": period_end,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )

    if not rows:
        return pl.DataFrame(schema=schema)

    out = pl.DataFrame(rows).unique(subset=["period_end", "metric"]).sort(["metric", "period_end"])
    if use_cache:
        out.write_parquet(path)
    return out


def _to_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def with_available_date(frame: pl.DataFrame, *, lag_days: int = DEFAULT_REPORT_LAG_DAYS) -> pl.DataFrame:
    """Add ``available_date`` = ``period_end`` + ``lag_days`` (when the market may know the report)."""
    return frame.with_columns(
        (pl.col("period_end") + pl.duration(days=lag_days)).dt.date().alias("available_date")
    )
