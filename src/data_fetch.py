"""Download historical market data with yfinance and align as Polars DataFrames."""

from __future__ import annotations

import os
import re
from datetime import date, timedelta

import polars as pl
import yfinance as yf
from finnhub import Client as FinnhubClient

from src.peer_maps import peers_for_ticker

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
MACRO_TICKER = "^TNX"
_ACTION_COLUMNS = ("Dividends", "Stock Splits")


def _sanitize_symbol(sym: str) -> str:
    """ASCII-safe token for column names (Yahoo tickers may contain ^ or -)."""
    s = sym.strip().upper()
    return re.sub(r"[^A-Z0-9]", "", s)


def _pandas_ohlcv_to_polars(pdf) -> pl.DataFrame:
    """Build a Polars frame from a pandas OHLCV slice without requiring pyarrow."""
    columns = {}
    for name in pdf.columns:
        if name == "Date":
            columns[name] = pdf[name].astype("datetime64[ns]").to_numpy(copy=False)
        else:
            columns[name] = pdf[name].to_numpy(copy=False)
    return pl.DataFrame(columns).with_columns(pl.col("Date").cast(pl.Date))


def _history_to_polars(
    ticker: str,
    period: str = "10y",
    *,
    include_actions: bool = False,
) -> pl.DataFrame:
    """Fetch daily OHLCV + Adj Close; optionally Dividends / Stock Splits (``actions=True``)."""
    raw = yf.Ticker(ticker).history(
        period=period,
        auto_adjust=False,
        interval="1d",
        actions=include_actions,
    )
    if raw.empty:
        raise ValueError(f"No data returned for ticker {ticker!r} (period={period!r}).")

    pdf = raw.reset_index()
    date_col = "Datetime" if "Datetime" in pdf.columns else "Date"
    pdf = pdf.rename(columns={date_col: "Date"})
    pdf["Date"] = pdf["Date"].dt.normalize().dt.tz_localize(None)

    keep = ["Date"] + [c for c in OHLCV_COLUMNS if c in pdf.columns]
    if include_actions:
        for c in _ACTION_COLUMNS:
            if c in pdf.columns:
                keep.append(c)
    pdf = pdf[keep]

    return _pandas_ohlcv_to_polars(pdf)


def _finnhub_news_schema() -> dict[str, pl.DataType]:
    return {
        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        "headline": pl.Utf8,
    }


def _parse_finnhub_news_items(raw_news: list) -> tuple[list[int], list[str]]:
    times: list[int] = []
    heads: list[str] = []
    for item in raw_news:
        if not isinstance(item, dict):
            continue
        ts = item.get("datetime")
        hl = item.get("headline")
        if ts is None or hl is None:
            continue
        try:
            times.append(int(ts))
        except (TypeError, ValueError):
            continue
        heads.append(str(hl))
    return times, heads


def fetch_finnhub_news_headlines(
    ticker: str,
    *,
    days: int = 365,
    api_key: str | None = None,
    chunk_days: int = 30,
) -> pl.DataFrame:
    """
    Fetch company news from Finnhub for roughly the last ``days`` calendar days.

    Finnhub's ``company_news`` endpoint only returns a few days of headlines when ``from`` and
    ``to`` span many months; this function walks the window in ``chunk_days`` slices (default
    30) so free-tier keys still retrieve roughly one month per request.

    Returns a Polars frame with columns ``datetime`` (UTC) and ``headline``. On API errors or
    when no articles are returned, returns an empty frame with the same schema.

    Parameters
    ----------
    ticker
        Equity symbol, e.g. ``'AAPL'``.
    days
        Window length ending today (inclusive of ``to`` date).
    api_key
        Finnhub API key. Defaults to ``FINNHUB_API_KEY`` env var, or the placeholder ``"dummy"``
        for local development (Finnhub will reject invalid keys; the frame will be empty).
    chunk_days
        Maximum calendar span per API call (Finnhub behaves best at ~30 days).
    """
    key = api_key if api_key is not None else os.environ.get("FINNHUB_API_KEY", "dummy")
    if not key or key in ("dummy", "your_key"):
        return pl.DataFrame(schema=_finnhub_news_schema())

    sym = ticker.strip().upper()
    end_d = date.today()
    start_d = end_d - timedelta(days=max(days, 1))
    span = max(chunk_days, 1)

    all_times: list[int] = []
    all_heads: list[str] = []
    try:
        client = FinnhubClient(api_key=key)
    except Exception:
        return pl.DataFrame(schema=_finnhub_news_schema())

    cursor = start_d
    while cursor <= end_d:
        chunk_end = min(cursor + timedelta(days=span - 1), end_d)
        try:
            raw_news: list = client.company_news(
                sym, _from=cursor.isoformat(), to=chunk_end.isoformat()
            )
        except Exception:
            raw_news = []
        times, heads = _parse_finnhub_news_items(raw_news or [])
        all_times.extend(times)
        all_heads.extend(heads)
        cursor = chunk_end + timedelta(days=1)

    if not all_heads:
        return pl.DataFrame(schema=_finnhub_news_schema())

    dt_series = pl.from_epoch(
        pl.Series("datetime", all_times, dtype=pl.Int64), time_unit="s"
    ).dt.replace_time_zone("UTC")
    out = pl.DataFrame(
        {"datetime": dt_series, "headline": pl.Series("headline", all_heads, dtype=pl.Utf8)}
    )
    return out.unique(subset=["datetime", "headline"]).sort("datetime")


def _prefix_columns(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Rename all columns except Date with the given prefix."""
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c != "Date"}
    return df.rename(rename_map)


def _join_peer_universe(
    merged: pl.DataFrame,
    *,
    tickers: list[str],
    period: str,
    column_prefix: str,
) -> pl.DataFrame:
    """
    Left-join each peer's ``Adj Close`` onto the target calendar (one column per peer).

    Column pattern: ``{column_prefix}_{SANITIZED}_Adj Close`` (spaces preserved like ``target_*``).
    Missing peer dates are forward-filled after join (limited cross-session staleness).
    """
    out = merged
    for sym in tickers:
        key = _sanitize_symbol(sym)
        col_name = f"{column_prefix}_{key}_Adj Close"
        try:
            hist = _history_to_polars(sym, period, include_actions=False)
        except ValueError:
            continue
        if "Adj Close" not in hist.columns:
            continue
        peer = hist.select(
            [
                "Date",
                pl.col("Adj Close").alias(col_name),
            ]
        )
        out = out.join(peer, on="Date", how="left")
        out = out.with_columns(pl.col(col_name).fill_null(strategy="forward"))
    return out


def fetch_aligned_market_data(
    target_ticker: str,
    benchmark_ticker: str,
    period: str = "10y",
    *,
    competitors: list[str] | None = None,
    partners: list[str] | None = None,
    use_peer_maps: bool = True,
) -> pl.DataFrame:
    """
    Fetch daily data for a stock, a benchmark, the 10Y Treasury, optional peers, and dividends.

    When ``use_peer_maps`` is True (default), competitor/partner tickers are taken from
    ``src.peer_maps`` unless ``competitors`` / ``partners`` are passed explicitly.

    Peer columns use names like ``pcomp_MSFT_Adj Close`` and ``psupp_TSM_Adj Close`` (left-joined
    on the target's trading calendar, then forward-filled for sparse sessions).

    The target row includes ``target_Dividends`` and ``target_Stock Splits`` when Yahoo returns them.
    """
    t_up = target_ticker.strip().upper()
    b_up = benchmark_ticker.strip().upper()

    if competitors is None or partners is None:
        cmap, pmap = peers_for_ticker(t_up)
        if competitors is None:
            competitors = cmap if use_peer_maps else []
        if partners is None:
            partners = pmap if use_peer_maps else []

    target = _prefix_columns(
        _history_to_polars(t_up, period, include_actions=True),
        "target",
    )
    benchmark = _prefix_columns(
        _history_to_polars(b_up, period, include_actions=False), "benchmark"
    )
    macro = _prefix_columns(
        _history_to_polars(MACRO_TICKER, period, include_actions=False), "macro"
    )

    merged = (
        target.join(benchmark, on="Date", how="left")
        .join(macro, on="Date", how="left")
        .sort("Date")
    )

    fill_cols = [c for c in merged.columns if c.startswith(("benchmark_", "macro_"))]
    if fill_cols:
        merged = merged.with_columns(
            [pl.col(c).fill_null(strategy="forward") for c in fill_cols]
        )

    if "target_Dividends" not in merged.columns:
        merged = merged.with_columns(pl.lit(0.0).alias("target_Dividends"))
    else:
        merged = merged.with_columns(pl.col("target_Dividends").fill_null(0.0))

    if "target_Stock Splits" not in merged.columns:
        merged = merged.with_columns(pl.lit(0.0).alias("target_Stock Splits"))
    else:
        merged = merged.with_columns(pl.col("target_Stock Splits").fill_null(0.0))

    merged = _join_peer_universe(
        merged, tickers=competitors, period=period, column_prefix="pcomp"
    )
    merged = _join_peer_universe(
        merged, tickers=partners, period=period, column_prefix="psupp"
    )

    return merged
