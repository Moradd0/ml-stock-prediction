"""Train XGBoost regressor + scaler on historical data and save artifacts for the API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.project_env import load_project_env

load_project_env()

from src.data_fetch import fetch_aligned_market_data
from src.evaluate import save_model_bundle
from src.features import TARGET_PRICE_COLUMN, TARGET_RETURN_COLUMN, engineer_features
from src.models.regression_model import fit_huber_regressor_tuned
from src.preprocess import (
    chronological_train_val_test_split,
    scale_train_val_test,
    split_features_and_target,
)


def train_and_save_bundle(
    ticker: str,
    *,
    benchmark: str = "QQQ",
    period: str = "10y",
    out_dir: str | Path = "models",
    bundle_filename: str | None = None,
    tune: bool = True,
) -> Path:
    """
    Train on ``Target_Next_Return`` with Huber loss (+ light val tuning), save bundle.

    Direction for backtest: predicted return > ``direction_threshold`` (tuned on validation).
    """
    raw = fetch_aligned_market_data(ticker.upper(), benchmark.upper(), period=period)
    feat = engineer_features(raw, keep_incomplete_target=False, target_ticker=ticker.upper())
    X, y = split_features_and_target(feat, target_column=TARGET_RETURN_COLUMN)
    feature_columns = list(X.columns)
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_train_val_test_split(X, y)
    Xtr, Xva, Xte, scaler = scale_train_val_test(X_train, X_val, X_test)

    reg, tune_meta = fit_huber_regressor_tuned(
        Xtr,
        y_train,
        Xva,
        y_val,
        feature_names=feature_columns,
        tune=tune,
    )
    direction_threshold = float(tune_meta.get("direction_threshold", 0.0))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = bundle_filename or f"bundle_{ticker.upper()}.joblib"
    path = out / fname
    save_model_bundle(
        path,
        {
            "task": "regression",
            "target_column": TARGET_RETURN_COLUMN,
            "target_mode": "return",
            "model": reg,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "ticker": ticker.upper(),
            "benchmark": benchmark.upper(),
            "period": period,
            "news_days": 365,
            "lookback_calendar_days": 14,
            "report_lag_days": 45,
            "loss": "huber",
            "direction_threshold": direction_threshold,
            "tune_meta": tune_meta,
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Huber XGBoost on next-day return; save bundle for /predict."
    )
    parser.add_argument("ticker", help="Target ticker, e.g. MSFT")
    parser.add_argument("--benchmark", default="QQQ")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter search; use default grid middle preset only",
    )
    args = parser.parse_args()
    path = train_and_save_bundle(
        args.ticker,
        benchmark=args.benchmark,
        period=args.period,
        out_dir=args.out_dir,
        tune=not args.no_tune,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
