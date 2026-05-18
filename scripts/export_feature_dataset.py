#!/usr/bin/env python3
"""Write the engineered feature table (Date + features + Target_Direction) to CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.project_env import load_project_env  # noqa: E402
from src.data_fetch import fetch_aligned_market_data  # noqa: E402

load_project_env()
from src.features import engineer_features  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Export leakage-safe engineered features to CSV.")
    p.add_argument("ticker", help="Target ticker, e.g. AAPL")
    p.add_argument("--benchmark", default="QQQ")
    p.add_argument("--period", default="5y")
    p.add_argument("-o", "--output", required=True, help="Output CSV path")
    p.add_argument(
        "--no-peer-maps",
        action="store_true",
        help="Fetch only target + benchmark + macro (no competitor/partner columns).",
    )
    args = p.parse_args()
    raw = fetch_aligned_market_data(
        args.ticker.upper(),
        args.benchmark.upper(),
        period=args.period,
        use_peer_maps=not args.no_peer_maps,
    )
    feat = engineer_features(raw, keep_incomplete_target=False, target_ticker=args.ticker.upper())
    feat.write_csv(args.output)
    print(f"Wrote {feat.height} rows x {len(feat.columns)} columns -> {args.output}")


if __name__ == "__main__":
    main()
