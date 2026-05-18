#!/usr/bin/env python3
"""Quick check that FINNHUB_API_KEY is set and company news is returned."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.project_env import load_project_env  # noqa: E402
from src.data_fetch import fetch_finnhub_news_headlines  # noqa: E402

load_project_env()


def run_check(ticker: str = "MSFT", *, days: int = 90) -> None:
    from src.project_env import finnhub_key_configured, require_finnhub_key

    ok = finnhub_key_configured()
    print(f"FINNHUB_API_KEY set: {ok}")
    if not ok:
        require_finnhub_key()

    sym = ticker.strip().upper()
    news = fetch_finnhub_news_headlines(sym, days=days)
    print(f"Headlines ({days}d, chunked): {news.height}")
    if news.height == 0:
        print("No articles — check key at https://finnhub.io/dashboard")
        raise SystemExit(1)

    cal = news.with_columns(pl.col("datetime").dt.date().alias("cal_date"))
    print(f"Calendar span: {cal['cal_date'].min()} .. {cal['cal_date'].max()}")
    print(cal.select("datetime", "headline").tail(3))


def main() -> None:
    p = argparse.ArgumentParser(description="Verify Finnhub API key and news fetch.")
    p.add_argument("ticker", nargs="?", default="MSFT")
    p.add_argument("--days", type=int, default=90)
    args = p.parse_args()
    run_check(args.ticker, days=args.days)


if __name__ == "__main__":
    main()
