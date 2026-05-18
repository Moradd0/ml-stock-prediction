"""Join quarterly earnings / fundamental metrics onto daily feature rows (leakage-safe)."""

from __future__ import annotations

import polars as pl

from src.fundamentals_fetch import (
    DEFAULT_REPORT_LAG_DAYS,
    fetch_finnhub_quarterly_earnings,
    fetch_finnhub_quarterly_metrics,
    with_available_date,
)

_EARNINGS_FEATURE_COLS = [
    "qtr_days_since_earnings",
    "qtr_earnings_surprise_pct",
    "qtr_earnings_beat",
    "qtr_eps_surprise_ratio",
]

_METRIC_FEATURE_MAP = {
    "eps": "qtr_eps",
    "netMargin": "qtr_net_margin",
    "grossMargin": "qtr_gross_margin",
    "operatingMargin": "qtr_operating_margin",
    "roeTTM": "qtr_roe_ttm",
    "totalDebtToEquity": "qtr_debt_to_equity",
}


def _join_asof_metric(
    sessions: pl.DataFrame,
    panel: pl.DataFrame,
    value_col: str,
    out_col: str,
) -> pl.DataFrame:
    if panel.height == 0:
        return sessions.with_columns(pl.lit(0.0).alias(out_col))
    right = panel.select("available_date", pl.col(value_col).alias(out_col)).sort("available_date")
    joined = (
        sessions.sort("Date")
        .join_asof(right, left_on="Date", right_on="available_date", strategy="backward")
        .drop("available_date", strict=False)
    )
    if out_col not in joined.columns:
        joined = joined.with_columns(pl.lit(0.0).alias(out_col))
    return joined.with_columns(pl.col(out_col).fill_null(0.0))


def _earnings_panel(earnings: pl.DataFrame, *, lag_days: int) -> pl.DataFrame:
    if earnings.height == 0:
        return pl.DataFrame(
            schema={
                "available_date": pl.Date,
                "qtr_earnings_surprise_pct": pl.Float64,
                "qtr_earnings_beat": pl.Float64,
                "qtr_eps_surprise_ratio": pl.Float64,
            }
        )
    base = with_available_date(earnings, lag_days=lag_days)
    ratio = (
        pl.when(pl.col("eps_estimate").abs() > 1e-12)
        .then(pl.col("eps_actual") / pl.col("eps_estimate") - 1.0)
        .otherwise(0.0)
    )
    return base.with_columns(
        [
            pl.col("surprise_pct").fill_null(0.0).alias("qtr_earnings_surprise_pct"),
            (pl.col("surprise").fill_null(0.0) > 0).cast(pl.Float64).alias("qtr_earnings_beat"),
            ratio.alias("qtr_eps_surprise_ratio"),
        ]
    ).select(
        "available_date",
        "qtr_earnings_surprise_pct",
        "qtr_earnings_beat",
        "qtr_eps_surprise_ratio",
    )


def merge_quarterly_fundamentals(
    df: pl.DataFrame,
    ticker: str | None,
    *,
    report_lag_days: int = DEFAULT_REPORT_LAG_DAYS,
    use_cache: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Attach quarterly-report features known only after ``period_end + report_lag_days``.

    Returns the enriched frame and the list of new column names.
    """
    if not ticker:
        zeros = {c: pl.lit(0.0) for c in list(_EARNINGS_FEATURE_COLS) + list(_METRIC_FEATURE_MAP.values())}
        return df.with_columns(**zeros), list(_EARNINGS_FEATURE_COLS) + list(_METRIC_FEATURE_MAP.values())

    sessions = df.select("Date").unique().sort("Date")
    sym = ticker.strip().upper()

    earnings = fetch_finnhub_quarterly_earnings(sym, use_cache=use_cache)
    earn_panel = _earnings_panel(earnings, lag_days=report_lag_days)

    if earn_panel.height > 0:
        joined = sessions.sort("Date").join_asof(
            earn_panel.sort("available_date"),
            left_on="Date",
            right_on="available_date",
            strategy="backward",
        )
        joined = joined.with_columns(
            (pl.col("Date") - pl.col("available_date")).dt.total_days().alias("qtr_days_since_earnings")
        ).with_columns(pl.col("qtr_days_since_earnings").fill_null(999.0))
        for c in ("qtr_earnings_surprise_pct", "qtr_earnings_beat", "qtr_eps_surprise_ratio"):
            joined = joined.with_columns(pl.col(c).fill_null(0.0))
        joined = joined.drop("available_date")
    else:
        joined = sessions.with_columns(
            [
                pl.lit(999.0).alias("qtr_days_since_earnings"),
                pl.lit(0.0).alias("qtr_earnings_surprise_pct"),
                pl.lit(0.0).alias("qtr_earnings_beat"),
                pl.lit(0.0).alias("qtr_eps_surprise_ratio"),
            ]
        )

    metrics_long = fetch_finnhub_quarterly_metrics(sym, use_cache=use_cache)
    for metric_key, out_col in _METRIC_FEATURE_MAP.items():
        sub = metrics_long.filter(pl.col("metric") == metric_key)
        if sub.height == 0:
            joined = joined.with_columns(pl.lit(0.0).alias(out_col))
            continue
        panel = with_available_date(sub.select("period_end", "value"), lag_days=report_lag_days)
        joined = _join_asof_metric(joined, panel, "value", out_col)

    new_cols = _EARNINGS_FEATURE_COLS + list(_METRIC_FEATURE_MAP.values())
    drop_extra = [c for c in joined.columns if c != "Date" and c not in new_cols]
    joined = joined.drop(drop_extra) if drop_extra else joined
    return df.join(joined, on="Date", how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in new_cols]
    ), new_cols
