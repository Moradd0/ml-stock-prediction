"""Disk cache for Finnhub headlines and FinBERT daily sentiment (speeds up repeat runs)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"


def _ticker_dir(ticker: str) -> Path:
    d = CACHE_DIR / ticker.strip().upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


def headlines_cache_path(ticker: str, news_days: int) -> Path:
    return _ticker_dir(ticker) / f"finnhub_headlines_{news_days}d.parquet"


def scored_cache_path(ticker: str, news_days: int) -> Path:
    return _ticker_dir(ticker) / f"finnhub_scored_{news_days}d.parquet"


def daily_cache_path(
    ticker: str,
    *,
    news_days: int,
    lookback_calendar_days: int,
    session_start: date,
    session_end: date,
) -> Path:
    return _ticker_dir(ticker) / (
        f"daily_sentiment_{news_days}d_{lookback_calendar_days}lb_"
        f"{session_start.isoformat()}_{session_end.isoformat()}.parquet"
    )


def _meta_path(data_path: Path) -> Path:
    return data_path.with_suffix(".meta.json")


def read_headlines_cache(ticker: str, news_days: int, *, built_today: bool = True) -> pl.DataFrame | None:
    path = headlines_cache_path(ticker, news_days)
    if not path.is_file():
        return None
    meta = _read_meta(path)
    if built_today and meta.get("built_on") != date.today().isoformat():
        return None
    if meta.get("news_days") != news_days:
        return None
    return pl.read_parquet(path)


def write_headlines_cache(ticker: str, news_days: int, frame: pl.DataFrame) -> None:
    path = headlines_cache_path(ticker, news_days)
    frame.write_parquet(path)
    _write_meta(path, {"built_on": date.today().isoformat(), "news_days": news_days, "rows": frame.height})


def read_scored_cache(ticker: str, news_days: int) -> pl.DataFrame | None:
    path = scored_cache_path(ticker, news_days)
    if not path.is_file():
        return None
    return pl.read_parquet(path)


def write_scored_cache(ticker: str, news_days: int, frame: pl.DataFrame) -> None:
    path = scored_cache_path(ticker, news_days)
    frame.write_parquet(path)
    _write_meta(
        path,
        {
            "built_on": date.today().isoformat(),
            "news_days": news_days,
            "rows": frame.height,
        },
    )


def read_daily_cache(
    ticker: str,
    *,
    news_days: int,
    lookback_calendar_days: int,
    session_start: date,
    session_end: date,
    built_today: bool = True,
) -> pl.DataFrame | None:
    path = daily_cache_path(
        ticker,
        news_days=news_days,
        lookback_calendar_days=lookback_calendar_days,
        session_start=session_start,
        session_end=session_end,
    )
    if not path.is_file():
        return None
    meta = _read_meta(path)
    if built_today and meta.get("built_on") != date.today().isoformat():
        return None
    expected = {
        "news_days": news_days,
        "lookback_calendar_days": lookback_calendar_days,
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
    }
    for k, v in expected.items():
        if meta.get(k) != v:
            return None
    return pl.read_parquet(path)


def write_daily_cache(
    ticker: str,
    frame: pl.DataFrame,
    *,
    news_days: int,
    lookback_calendar_days: int,
    session_start: date,
    session_end: date,
) -> None:
    path = daily_cache_path(
        ticker,
        news_days=news_days,
        lookback_calendar_days=lookback_calendar_days,
        session_start=session_start,
        session_end=session_end,
    )
    frame.select("Date", "Daily_Sentiment").write_parquet(path)
    _write_meta(
        path,
        {
            "built_on": date.today().isoformat(),
            "news_days": news_days,
            "lookback_calendar_days": lookback_calendar_days,
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "rows": frame.height,
        },
    )


def _read_meta(data_path: Path) -> dict:
    import json

    mp = _meta_path(data_path)
    if not mp.is_file():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_meta(data_path: Path, meta: dict) -> None:
    import json

    _meta_path(data_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
