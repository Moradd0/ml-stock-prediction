#!/usr/bin/env python3
"""
Single entry point for the stock prediction project.

Loads optional ``.env`` from the project root, sets safe defaults for FinBERT,
and exposes subcommands so you do not need to remember export/PYTHONPATH flags.

Examples
--------
    python scripts/run.py check MSFT
    python scripts/run.py train MSFT --period 5y
    python scripts/run.py predict MSFT --compact
    python scripts/run.py export MSFT -o features_MSFT.csv
    python scripts/run.py pipeline MSFT          # check → train (if needed) → predict
    python scripts/run.py api                    # start FastAPI (blocking)
    python scripts/run.py ui                     # start Streamlit (blocking)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bootstrap() -> None:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.project_env import load_project_env

    load_project_env()


def _bundle_path(ticker: str) -> Path:
    return ROOT / "models" / f"bundle_{ticker.upper()}.joblib"


def cmd_setup(args: argparse.Namespace) -> None:
    from src.project_env import finnhub_key_configured

    dest = ROOT / ".env"
    if dest.is_file() and not args.force:
        print(f"{dest} already exists — edit FINNHUB_API_KEY there (no export needed).")
    else:
        shutil.copy(ROOT / ".env.example", dest)
        print(f"Created {dest}")
        print("  Open .env and set:  FINNHUB_API_KEY=paste_your_key_here")
    if finnhub_key_configured():
        print("FINNHUB_API_KEY is configured.")
    else:
        print("After saving .env, run:  python scripts/run.py check MSFT")


def cmd_check(args: argparse.Namespace) -> None:
    from scripts.check_finnhub import run_check

    run_check(args.ticker, days=args.days)


def cmd_train(args: argparse.Namespace) -> None:
    from src.project_env import require_finnhub_key
    from src.train_bundle import train_and_save_bundle

    require_finnhub_key()

    path = train_and_save_bundle(
        args.ticker,
        benchmark=args.benchmark,
        period=args.period,
        out_dir=args.out_dir,
    )
    print(f"Saved {path}")


def cmd_predict(args: argparse.Namespace) -> None:
    from fastapi import HTTPException

    from app_api import PredictRequest, run_predict
    from src.project_env import require_finnhub_key

    require_finnhub_key()
    from scripts.predict_cli import _compact_payload

    req = PredictRequest(ticker=args.ticker.upper(), benchmark=args.benchmark.upper())
    try:
        data = run_predict(req)
    except HTTPException as exc:
        print(f"{exc.status_code}: {exc.detail}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.compact:
        data = _compact_payload(data)
    print(json.dumps(data, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    from src.data_fetch import fetch_aligned_market_data
    from src.project_env import require_finnhub_key

    require_finnhub_key()
    from src.features import engineer_features

    raw = fetch_aligned_market_data(
        args.ticker.upper(),
        args.benchmark.upper(),
        period=args.period,
        use_peer_maps=not args.no_peer_maps,
    )
    feat = engineer_features(raw, keep_incomplete_target=False, target_ticker=args.ticker.upper())
    out = Path(args.output)
    feat.write_csv(out)
    nz = (feat["Daily_Sentiment"] != 0).sum()
    print(f"Wrote {feat.height} rows x {len(feat.columns)} columns -> {out}")
    print(f"Non-zero Daily_Sentiment rows: {nz}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """check Finnhub → train if bundle missing → predict."""
    if not args.skip_check:
        print("=== Finnhub check ===")
        cmd_check(argparse.Namespace(ticker=args.ticker, days=args.days))

    bundle = _bundle_path(args.ticker)
    if bundle.is_file() and not args.retrain:
        print(f"=== Using existing model: {bundle} ===")
    else:
        print("=== Training ===")
        cmd_train(
            argparse.Namespace(
                ticker=args.ticker,
                benchmark=args.benchmark,
                period=args.period,
                out_dir=args.out_dir,
            )
        )

    print("=== Predict ===")
    cmd_predict(
        argparse.Namespace(
            ticker=args.ticker,
            benchmark=args.benchmark,
            compact=args.compact,
        )
    )


def cmd_api(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app_api:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
            *(["--reload"] if args.reload else []),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def cmd_ui(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    if args.api_url:
        env["API_URL"] = args.api_url
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app_ui.py", "--server.headless", "true"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the ML stock prediction project (one command, .env-friendly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tip: copy .env.example to .env and set FINNHUB_API_KEY once.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="Create .env from .env.example (one-time API key)")
    s.add_argument("--force", action="store_true", help="Overwrite existing .env")
    s.set_defaults(func=cmd_setup)

    c = sub.add_parser("check", help="Verify FINNHUB_API_KEY and news fetch")
    c.add_argument("ticker", nargs="?", default="MSFT")
    c.add_argument("--days", type=int, default=90)
    c.set_defaults(func=cmd_check)

    t = sub.add_parser("train", help="Train and save models/bundle_TICKER.joblib")
    t.add_argument("ticker", nargs="?", default="MSFT")
    t.add_argument("--benchmark", default="QQQ")
    t.add_argument("--period", default="10y", help="Yahoo history window (stored in bundle)")
    t.add_argument("--out-dir", default="models")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Run inference + test backtest (slow: FinBERT)")
    pr.add_argument("ticker", nargs="?", default="MSFT")
    pr.add_argument("--benchmark", default="QQQ")
    pr.add_argument("--compact", action="store_true", help="Omit long equity arrays from JSON")
    pr.set_defaults(func=cmd_predict)

    e = sub.add_parser("export", help="Write engineered features CSV")
    e.add_argument("ticker", nargs="?", default="MSFT")
    e.add_argument("--benchmark", default="QQQ")
    e.add_argument("--period", default="10y")
    e.add_argument("-o", "--output", default="features_MSFT.csv")
    e.add_argument("--no-peer-maps", action="store_true")
    e.set_defaults(func=cmd_export)

    pl = sub.add_parser("pipeline", help="check → train (if needed) → predict")
    pl.add_argument("ticker", nargs="?", default="MSFT")
    pl.add_argument("--benchmark", default="QQQ")
    pl.add_argument("--period", default="10y")
    pl.add_argument("--out-dir", default="models")
    pl.add_argument("--days", type=int, default=90, help="Finnhub check window")
    pl.add_argument("--retrain", action="store_true", help="Force retrain even if bundle exists")
    pl.add_argument("--skip-check", action="store_true")
    pl.add_argument("--compact", action="store_true", default=True)
    pl.add_argument("--no-compact", action="store_false", dest="compact")
    pl.set_defaults(func=cmd_pipeline)

    a = sub.add_parser("api", help="Start FastAPI (blocking)")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--reload", action="store_true", default=True)
    a.add_argument("--no-reload", action="store_false", dest="reload")
    a.set_defaults(func=cmd_api)

    u = sub.add_parser("ui", help="Start Streamlit UI (blocking; run api in another terminal)")
    u.add_argument("--api-url", default=None, help="Default http://127.0.0.1:8000")
    u.set_defaults(func=cmd_ui)

    cmp = sub.add_parser("compare", help="Train/eval 2y,5y,10y,15y and print summary table")
    cmp.add_argument("ticker", nargs="?", default="MSFT")
    cmp.add_argument("--benchmark", default="QQQ")
    cmp.add_argument("--periods", default="2y,5y,10y,15y")
    cmp.add_argument("--save-bundles", action="store_true")
    cmp.set_defaults(func=cmd_compare)

    return p


def cmd_compare(args: argparse.Namespace) -> None:
    from scripts.compare_periods import main as compare_main

    argv = [args.ticker, "--benchmark", args.benchmark, "--periods", args.periods]
    if args.save_bundles:
        argv.append("--save-bundles")
    sys.argv = ["compare_periods", *argv]
    compare_main()


def main() -> None:
    bootstrap()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
