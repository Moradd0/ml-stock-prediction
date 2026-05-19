"""Engineer stationary features and a next-day direction label from aligned OHLCV data."""

from __future__ import annotations

import functools
import re
from typing import Iterable

import pandas as pd
import polars as pl

import pandas_ta  # noqa: F401 — registers ``DataFrame.ta`` accessor on pandas

from src.data_fetch import fetch_finnhub_news_headlines
from src.fundamentals_features import merge_quarterly_fundamentals
from src.sentiment_cache import (
    read_daily_cache,
    read_headlines_cache,
    read_scored_cache,
    write_daily_cache,
    write_headlines_cache,
    write_scored_cache,
)


_REQUIRED = (
    "Date",
    "target_Open",
    "target_High",
    "target_Low",
    "target_Close",
    "target_Adj Close",
    "target_Volume",
    "benchmark_Adj Close",
    "macro_Close",
)

_PEER_ADJ_RE = re.compile(r"^p(?:comp|supp)_([^_]+)_Adj Close$")


def _pick_col(columns: list[str], pattern: str) -> str:
    rx = re.compile(pattern)
    for c in columns:
        if rx.match(c):
            return c
    raise ValueError(f"No column matching {pattern!r} among {columns}")


def _ta_frame(df: pl.DataFrame) -> pd.DataFrame:
    """Target OHLCV as a pandas frame with standard names, row order = ``Date`` sort."""
    pdf = df.sort("Date")
    return pd.DataFrame(
        {
            "Date": pdf["Date"].to_numpy(),
            "Open": pdf["target_Open"].to_numpy(),
            "High": pdf["target_High"].to_numpy(),
            "Low": pdf["target_Low"].to_numpy(),
            "Close": pdf["target_Close"].to_numpy(),
            "Volume": pdf["target_Volume"].to_numpy(),
        }
    )


def _append_ta(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger bandwidth, and ATR (no future rows used)."""
    out = pdf.copy()
    out.ta.rsi(close="Close", length=14, append=True)
    out.ta.macd(close="Close", fast=12, slow=26, signal=9, append=True)
    out.ta.bbands(close="Close", length=20, std=2.0, append=True)
    out.ta.atr(high="High", low="Low", close="Close", length=14, append=True)
    return out


def _peer_adj_columns(df: pl.DataFrame, table_prefix: str) -> list[str]:
    """Columns like ``pcomp_MSFT_Adj Close`` (``table_prefix`` = ``pcomp_``)."""
    return sorted(
        c for c in df.columns if c.startswith(table_prefix) and c.endswith("Adj Close")
    )


def _add_dividend_and_split_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Causal dividend / split proxies (information available through each row's close).

    No forward shifts from realized *future* prices; rolling sums are ``.shift(1)`` so row ``t``
    uses cash amounts booked on dates ``<= t-1`` for multi-day windows ending yesterday.
    """
    div = pl.col("target_Dividends").fill_null(0.0)
    sp = pl.col("target_Stock Splits").fill_null(0.0)
    exdiv = (div > 0).cast(pl.Int8)
    exdiv_date = pl.when(div > 0).then(pl.col("Date")).otherwise(None)
    last_ex = exdiv_date.forward_fill()
    days_since = (pl.col("Date") - last_ex).dt.total_days().cast(pl.Float64)
    close_lag1 = pl.col("target_Close").shift(1)
    denom = pl.max_horizontal(close_lag1, pl.lit(1e-12))

    step = df.with_columns(
        [
            exdiv.alias("exdiv_flag"),
            exdiv.shift(1).fill_null(0).cast(pl.Int8).alias("exdiv_lag1"),
            div.rolling_sum(window_size=30, min_periods=1)
            .shift(1)
            .alias("div_sum_30_lag1"),
            div.rolling_sum(window_size=252, min_periods=1)
            .shift(1)
            .alias("div_sum_252_lag1"),
            pl.when(div > 0)
            .then(div)
            .otherwise(None)
            .forward_fill()
            .shift(1)
            .fill_null(0.0)
            .alias("last_div_amount_lag1"),
            (exdiv.shift(1) == 1).cast(pl.Int8).fill_null(0).alias("post_exdiv_1d"),
            days_since.clip(0.0, 365.0).fill_null(365.0).alias("days_since_exdiv"),
            (sp.abs() > 1e-9)
            .cast(pl.Int8)
            .shift(1)
            .fill_null(0)
            .alias("split_event_lag1"),
        ]
    )
    return step.with_columns(
        (pl.col("div_sum_252_lag1") / denom)
        .fill_null(0.0)
        .clip(0.0, 1.0)
        .alias("trailing_div_yield_lag1"),
    )


def _add_peer_table_features(
    df: pl.DataFrame, table_prefix: str, short_prefix: str
) -> tuple[pl.DataFrame, list[str]]:
    """
    Equal-weight peer returns, dispersion, and per-peer lagged returns (no lookahead).

    Returns the frame plus the list of **new** feature column names added.
    """
    peer_cols = _peer_adj_columns(df, table_prefix)
    new_names: list[str] = [f"{short_prefix}_peer_n"]

    if not peer_cols:
        z = pl.lit(0.0)
        out = df.with_columns(
            [
                pl.lit(0).cast(pl.Int32).alias(new_names[0]),
                z.alias(f"{short_prefix}_eq_ret1_lag1"),
                z.alias(f"{short_prefix}_eq_ret5_lag1"),
                z.alias(f"{short_prefix}_eq_ret10_lag1"),
                z.alias(f"{short_prefix}_disp_ret1_lag1"),
            ]
        )
        return out, new_names + [
            f"{short_prefix}_eq_ret1_lag1",
            f"{short_prefix}_eq_ret5_lag1",
            f"{short_prefix}_eq_ret10_lag1",
            f"{short_prefix}_disp_ret1_lag1",
        ]

    ret_exprs: list[pl.Expr] = []
    ret_aliases: list[str] = []
    for c in peer_cols:
        m = _PEER_ADJ_RE.match(c)
        sym = m.group(1) if m else c.replace(table_prefix, "").replace("_Adj Close", "")
        alias = f"__ret_{short_prefix}_{sym}"
        ret_aliases.append(alias)
        ret_exprs.append((pl.col(c) / pl.col(c).shift(1) - 1.0).alias(alias))

    mean_e = pl.mean_horizontal([pl.col(a) for a in ret_aliases]).alias(
        f"__mean_{short_prefix}"
    )
    std_e = (
        pl.concat_list([pl.col(a) for a in ret_aliases])
        .list.std(ddof=0)
        .alias(f"__std_{short_prefix}")
    )

    out = (
        df.with_columns(ret_exprs)
        .with_columns([mean_e, std_e])
        .with_columns(
            [
                pl.lit(len(peer_cols)).cast(pl.Int32).alias(f"{short_prefix}_peer_n"),
                pl.col(f"__mean_{short_prefix}")
                .shift(1)
                .fill_null(0.0)
                .alias(f"{short_prefix}_eq_ret1_lag1"),
                pl.col(f"__mean_{short_prefix}")
                .rolling_sum(window_size=5, min_periods=1)
                .shift(1)
                .fill_null(0.0)
                .alias(f"{short_prefix}_eq_ret5_lag1"),
                pl.col(f"__mean_{short_prefix}")
                .rolling_sum(window_size=10, min_periods=1)
                .shift(1)
                .fill_null(0.0)
                .alias(f"{short_prefix}_eq_ret10_lag1"),
                pl.col(f"__std_{short_prefix}")
                .shift(1)
                .fill_null(0.0)
                .alias(f"{short_prefix}_disp_ret1_lag1"),
            ]
        )
    )

    ll_names: list[str] = []
    for c, ra in zip(peer_cols, ret_aliases, strict=True):
        m = _PEER_ADJ_RE.match(c)
        sym = m.group(1) if m else ra
        ll = f"ll_{short_prefix}_{sym}_ret1_lag1"
        ll_names.append(ll)
        out = out.with_columns(pl.col(ra).shift(1).fill_null(0.0).alias(ll))

    drop_tmp = ret_aliases + [f"__mean_{short_prefix}", f"__std_{short_prefix}"]
    out = out.drop(drop_tmp)

    new_names.extend(
        [
            f"{short_prefix}_eq_ret1_lag1",
            f"{short_prefix}_eq_ret5_lag1",
            f"{short_prefix}_eq_ret10_lag1",
            f"{short_prefix}_disp_ret1_lag1",
        ]
        + ll_names
    )
    return out, new_names


@functools.lru_cache(maxsize=1)
def load_finbert_pipeline():
    """
    Load the ProsusAI/finbert sentiment classifier from Hugging Face (cached process-wide).

    Uses CPU by default. First call downloads model weights.
    """
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
        device=-1,
    )


def _finbert_label_to_score(label: str) -> float:
    """Map FinBERT labels to scores: positive = 1, neutral = 0, negative = -1."""
    key = str(label).strip().lower()
    if key == "positive" or key.endswith("positive"):
        return 1.0
    if key == "negative" or key.endswith("negative"):
        return -1.0
    if key == "neutral" or key.endswith("neutral"):
        return 0.0
    return 0.0


def _sentiment_per_trading_session(
    scored: pl.DataFrame,
    trading_dates: pl.Series,
    *,
    lookback_calendar_days: int = 14,
) -> pl.DataFrame:
    """
    For each trading session ``Date``, average FinBERT scores of headlines whose calendar
    ``cal_date`` falls in ``(Date - lookback, Date]`` (weekend/holiday news included).

    This spreads sentiment across many trading rows instead of matching calendar days exactly.
    """
    articles = scored.with_columns(pl.col("datetime").dt.date().alias("cal_date"))
    sessions = (
        pl.DataFrame({"Date": trading_dates})
        .unique()
        .sort("Date")
        .with_columns(pl.lit(1).alias("_key"))
    )
    keyed = articles.with_columns(pl.lit(1).alias("_key"))
    paired = sessions.join(keyed, on="_key", how="inner").drop("_key")
    age_days = (pl.col("Date") - pl.col("cal_date")).dt.total_days()
    paired = paired.filter(
        (pl.col("cal_date") <= pl.col("Date")) & (age_days <= lookback_calendar_days)
    )

    if paired.height == 0:
        return sessions.select("Date").with_columns(
            pl.lit(0.0).alias("Daily_Sentiment")
        )

    daily = paired.group_by("Date").agg(
        pl.col("sent_score").mean().alias("Daily_Sentiment")
    )
    return (
        sessions.select("Date")
        .join(daily, on="Date", how="left")
        .with_columns(pl.col("Daily_Sentiment").fill_null(0.0))
    )


def _score_headlines_with_finbert(headlines: list[str]) -> list[float]:
    pipe = load_finbert_pipeline()
    scores: list[float] = []
    batch_size = 16
    for i in range(0, len(headlines), batch_size):
        chunk = headlines[i : i + batch_size]
        preds = pipe(chunk, truncation=True, max_length=512)
        for p in preds:
            scores.append(_finbert_label_to_score(p.get("label", "")))
    return scores


def _compute_daily_sentiment_scores(
    ticker: str,
    trading_dates: pl.Series,
    *,
    news_days: int = 365,
    lookback_calendar_days: int = 14,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Finnhub headlines (``news_days`` calendar lookback) scored with FinBERT, then aggregated
    per trading session via a rolling calendar window (see ``_sentiment_per_trading_session``).

    When ``use_cache`` is True, headlines / scores / daily aggregates are stored under
    ``data/cache/{TICKER}/`` (refreshed daily for headlines; daily table keyed by session range).
    """
    sym = ticker.strip().upper()
    sessions = pl.DataFrame({"Date": trading_dates}).unique().sort("Date")
    if sessions.height == 0:
        return pl.DataFrame(schema={"Date": pl.Date, "Daily_Sentiment": pl.Float64})

    session_start = sessions["Date"][0]
    session_end = sessions["Date"][-1]
    if use_cache:
        cached_daily = read_daily_cache(
            sym,
            news_days=news_days,
            lookback_calendar_days=lookback_calendar_days,
            session_start=session_start,
            session_end=session_end,
        )
        if cached_daily is not None:
            return sessions.join(cached_daily, on="Date", how="left").with_columns(
                pl.col("Daily_Sentiment").fill_null(0.0)
            )

    news = read_headlines_cache(sym, news_days) if use_cache else None
    if news is None:
        news = fetch_finnhub_news_headlines(sym, days=news_days)
        if use_cache and news.height > 0:
            write_headlines_cache(sym, news_days, news)

    if news.height == 0:
        empty = sessions.with_columns(pl.lit(0.0).alias("Daily_Sentiment"))
        return empty

    scored_cached = read_scored_cache(sym, news_days) if use_cache else None
    if scored_cached is not None and scored_cached.height > 0:
        new_articles = news.join(scored_cached.select("datetime", "headline"), on=["datetime", "headline"], how="anti")
        if new_articles.height == 0:
            scored = scored_cached
        else:
            new_scores = _score_headlines_with_finbert(new_articles["headline"].to_list())
            scored_new = new_articles.with_columns(pl.Series("sent_score", new_scores, dtype=pl.Float64))
            scored = pl.concat([scored_cached, scored_new], how="vertical_relaxed").unique(
                subset=["datetime", "headline"], keep="last"
            )
            if use_cache:
                write_scored_cache(sym, news_days, scored)
    else:
        scores = _score_headlines_with_finbert(news["headline"].to_list())
        scored = news.with_columns(pl.Series("sent_score", scores, dtype=pl.Float64))
        if use_cache:
            write_scored_cache(sym, news_days, scored)

    daily = _sentiment_per_trading_session(
        scored,
        trading_dates,
        lookback_calendar_days=lookback_calendar_days,
    )
    if use_cache:
        write_daily_cache(
            sym,
            daily,
            news_days=news_days,
            lookback_calendar_days=lookback_calendar_days,
            session_start=session_start,
            session_end=session_end,
        )
    return daily


def _merge_daily_sentiment(
    enriched: pl.DataFrame,
    target_ticker: str | None,
    *,
    news_days: int = 365,
    lookback_calendar_days: int = 14,
    use_sentiment_cache: bool = True,
) -> pl.DataFrame:
    if not target_ticker:
        return enriched.with_columns(pl.lit(0.0).alias("Daily_Sentiment"))
    trading_dates = enriched.get_column("Date")
    daily = _compute_daily_sentiment_scores(
        target_ticker,
        trading_dates,
        news_days=news_days,
        lookback_calendar_days=lookback_calendar_days,
        use_cache=use_sentiment_cache,
    )
    if daily.height == 0:
        return enriched.with_columns(pl.lit(0.0).alias("Daily_Sentiment"))
    return enriched.join(daily, on="Date", how="left").with_columns(
        pl.col("Daily_Sentiment").fill_null(0.0)
    )


# Primary regression target: next session adjusted close (known only after that session).
TARGET_PRICE_COLUMN = "Target_Next_Adj_Close"
# Derived classification label for reporting / direction accuracy.
TARGET_DIRECTION_COLUMN = "Target_Direction"


def _ordered_unique(iterable: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in iterable:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def engineer_features(
    df: pl.DataFrame,
    *,
    keep_incomplete_target: bool = False,
    target_ticker: str | None = None,
    use_sentiment_cache: bool = True,
    use_fundamentals: bool = True,
) -> pl.DataFrame:
    """
    Build leakage-safe features and price target ``Target_Next_Adj_Close`` from
    ``fetch_aligned_market_data`` output.

    Includes **Finnhub headlines** (last 30 days) scored with **ProsusAI/finbert** as
    ``Daily_Sentiment`` (daily mean of mapped scores: positive = 1, neutral = 0, negative = -1),
    merged on ``Date`` next to technical indicators such as RSI and MACD.

    Also includes **dividend / split** proxies on the target and **competitor / partner** return
    panels (see ``src.peer_maps``) when those columns are present on ``df``.

    ``Target_Next_Adj_Close`` is the *next* session's ``target_Adj Close`` (regression label).
    ``Target_Direction`` is derived (1 if next close > today). Peer inputs use only prices
    through each session's close; peer
    aggregates apply ``.shift(1)`` so row ``t`` does not embed same-day peer closes in the
    contemporaneous mean (mirrors the spirit of the reference coursework pipeline).

    Parameters
    ----------
    df
        Polars frame from ``data_fetch.fetch_aligned_market_data`` (target, benchmark, macro,
        optional ``pcomp_*`` / ``psupp_*`` Adj Close joins, ``target_Dividends``, splits).
    keep_incomplete_target
        If True, keep the last row even when ``Target_Next_Adj_Close`` is null (inference).
    target_ticker
        Target symbol for Finnhub news + FinBERT ``Daily_Sentiment`` (``news_days`` fetch +
        per-session rolling calendar mean). If ``None``, ``Daily_Sentiment`` is ``0.0``.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input frame missing required columns: {missing}")

    sorted_df = df.sort("Date")
    if "target_Dividends" not in sorted_df.columns:
        sorted_df = sorted_df.with_columns(pl.lit(0.0).alias("target_Dividends"))
    if "target_Stock Splits" not in sorted_df.columns:
        sorted_df = sorted_df.with_columns(pl.lit(0.0).alias("target_Stock Splits"))

    adj = pl.col("target_Adj Close")
    bench_adj = pl.col("benchmark_Adj Close")
    macro_px = pl.col("macro_Close")

    with_returns = sorted_df.with_columns(
        [
            (adj / adj.shift(1) - 1.0).alias("daily_return"),
            (bench_adj / bench_adj.shift(1) - 1.0).alias("benchmark_daily_return"),
            (macro_px / macro_px.shift(1) - 1.0).alias("treasury_yield_daily_change"),
            adj.shift(-1).alias(TARGET_PRICE_COLUMN),
        ]
    ).with_columns(
        [
            (pl.col(TARGET_PRICE_COLUMN) > adj)
            .cast(pl.Int8)
            .alias(TARGET_DIRECTION_COLUMN),
        ]
    ).with_columns(
        [
            (pl.col("daily_return") - pl.col("benchmark_daily_return")).alias(
                "relative_return_vs_benchmark"
            ),
            pl.col("daily_return").shift(3).alias("daily_return_lag_3"),
            pl.col("daily_return").shift(5).alias("daily_return_lag_5"),
            pl.col("daily_return")
            .rolling_std(window_size=20, min_samples=20)
            .alias("return_volatility_20d"),
        ]
    )

    ohlc = _ta_frame(with_returns)
    ohlc_ta = _append_ta(ohlc)
    cols = list(ohlc_ta.columns)

    rsi_c = _pick_col(cols, r"^RSI_14$")
    macd_c = _pick_col(cols, r"^MACD_12_26_9$")
    macdh_c = _pick_col(cols, r"^MACDh_12_26_9$")
    macds_c = _pick_col(cols, r"^MACDs_12_26_9$")
    bbb_c = _pick_col(cols, r"^BBB_")
    atr_c = _pick_col(cols, r"^ATRr?_14$")

    atr_vals = ohlc_ta[atr_c].to_numpy()
    close_vals = ohlc["Close"].to_numpy()
    atr_norm = atr_vals / close_vals

    n = len(with_returns)
    if len(ohlc_ta) != n:
        raise ValueError("Internal error: TA frame length does not match input.")

    enriched = with_returns.with_columns(
        [
            pl.Series("rsi_14", ohlc_ta[rsi_c].to_numpy()),
            pl.Series("macd", ohlc_ta[macd_c].to_numpy()),
            pl.Series("macd_histogram", ohlc_ta[macdh_c].to_numpy()),
            pl.Series("macd_signal", ohlc_ta[macds_c].to_numpy()),
            pl.Series("bollinger_bandwidth", ohlc_ta[bbb_c].to_numpy()),
            pl.Series("atr_14_normalized", atr_norm),
        ]
    )

    enriched = _merge_daily_sentiment(
        enriched, target_ticker, use_sentiment_cache=use_sentiment_cache
    )

    if use_fundamentals and target_ticker:
        enriched, qtr_cols = merge_quarterly_fundamentals(enriched, target_ticker)
    else:
        qtr_cols = []

    enriched = _add_dividend_and_split_features(enriched)
    div_feature_cols = [
        "exdiv_flag",
        "exdiv_lag1",
        "div_sum_30_lag1",
        "div_sum_252_lag1",
        "last_div_amount_lag1",
        "post_exdiv_1d",
        "days_since_exdiv",
        "trailing_div_yield_lag1",
        "split_event_lag1",
    ]

    enriched, comp_meta = _add_peer_table_features(enriched, "pcomp_", "comp")
    enriched, supp_meta = _add_peer_table_features(enriched, "psupp_", "supp")

    base_feature_cols = [
        "daily_return",
        "daily_return_lag_3",
        "daily_return_lag_5",
        "rsi_14",
        "macd",
        "macd_histogram",
        "macd_signal",
        "Daily_Sentiment",
        "bollinger_bandwidth",
        "atr_14_normalized",
        "return_volatility_20d",
        "relative_return_vs_benchmark",
        "treasury_yield_daily_change",
    ]

    target_cols = [TARGET_PRICE_COLUMN, TARGET_DIRECTION_COLUMN]
    feature_cols = _ordered_unique(
        base_feature_cols + qtr_cols + div_feature_cols + comp_meta + supp_meta + target_cols
    )

    float_cols = [c for c in feature_cols if c not in target_cols]
    out = enriched.select(["Date"] + feature_cols)
    out = out.with_columns([pl.col(c).fill_nan(None) for c in float_cols])
    if keep_incomplete_target:
        return out.drop_nulls(subset=float_cols)
    return out.drop_nulls(subset=feature_cols)
