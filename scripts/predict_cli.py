#!/usr/bin/env python3
"""Run the same logic as POST /predict from the terminal (no Streamlit; uvicorn optional)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.project_env import load_project_env  # noqa: E402

load_project_env()

import requests  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from app_api import PredictRequest, run_predict  # noqa: E402


def _compact_payload(data: dict) -> dict:
    out = {k: v for k, v in data.items() if k not in ("backtest",)}
    if "backtest_summary" in data:
        out["backtest_summary"] = data["backtest_summary"]
    bt = data.get("backtest") or {}
    if not bt:
        return out
    omit = (
        "dates",
        "equity_strategy",
        "equity_open_to_close_benchmark",
        "equity_buy_and_hold",
        "daily_returns_strategy",
        "daily_returns_open_to_close_benchmark",
    )
    out["backtest"] = {k: v for k, v in bt.items() if k not in omit}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict direction (same JSON as /predict). "
            "Default: in-process run_predict. Use --http to call a running API."
        ),
    )
    parser.add_argument("ticker", help="Target ticker, e.g. MSFT")
    parser.add_argument("--benchmark", default="QQQ", help="Benchmark ticker (bundle may override)")
    parser.add_argument(
        "--http",
        action="store_true",
        help="POST to running FastAPI (API_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit long backtest arrays (dates, equity, daily returns) from printed JSON",
    )
    args = parser.parse_args()
    req = PredictRequest(ticker=args.ticker.strip().upper(), benchmark=args.benchmark.strip().upper())

    if args.http:
        api_url = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
        try:
            resp = requests.post(
                f"{api_url}/predict",
                json={"ticker": req.ticker, "benchmark": req.benchmark},
                timeout=180,
            )
        except requests.RequestException as exc:
            print(f"Request failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if resp.status_code != 200:
            print(resp.status_code, resp.text, file=sys.stderr)
            raise SystemExit(1)
        data = resp.json()
    else:
        try:
            data = run_predict(req)
        except HTTPException as exc:
            print(f"{exc.status_code}: {exc.detail}", file=sys.stderr)
            raise SystemExit(1) from exc

    if args.compact:
        data = _compact_payload(data)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
