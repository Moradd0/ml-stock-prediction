"""Train XGBoost regressor + scaler on historical data and save artifacts for the API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m src.train_bundle` from repo root without PYTHONPATH=
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.project_env import load_project_env

load_project_env()

from src.data_fetch import fetch_aligned_market_data
from src.evaluate import save_model_bundle
from src.features import TARGET_PRICE_COLUMN, engineer_features
from src.models.regression_model import init_xgb_regressor, train_xgb_regressor
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
) -> Path:
    """
    Chronological split + scaler fit on train only + XGBoost regressor (MAE loss), then save.

    Predicts ``Target_Next_Adj_Close`` (next session adjusted close). Writes
    ``{out_dir}/bundle_{TICKER}.joblib`` with ``task=regression``.
    """
    raw = fetch_aligned_market_data(ticker.upper(), benchmark.upper(), period=period)
    feat = engineer_features(raw, keep_incomplete_target=False, target_ticker=ticker.upper())
    X, y = split_features_and_target(feat, target_column=TARGET_PRICE_COLUMN)
    feature_columns = list(X.columns)
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_train_val_test_split(X, y)
    Xtr, Xva, Xte, scaler = scale_train_val_test(X_train, X_val, X_test)

    reg = init_xgb_regressor()
    train_xgb_regressor(
        reg,
        Xtr,
        y_train,
        Xva,
        y_val,
        feature_names=feature_columns,
        early_stopping_rounds=20,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = bundle_filename or f"bundle_{ticker.upper()}.joblib"
    path = out / fname
    save_model_bundle(
        path,
        {
            "task": "regression",
            "target_column": TARGET_PRICE_COLUMN,
            "model": reg,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "ticker": ticker.upper(),
            "benchmark": benchmark.upper(),
            "period": period,
            "news_days": 365,
            "lookback_calendar_days": 14,
            "report_lag_days": 45,
            "loss": "mae",
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and save XGBoost regressor (next-day adj. close) for /predict API."
    )
    parser.add_argument("ticker", help="Target ticker, e.g. MSFT")
    parser.add_argument("--benchmark", default="QQQ")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()
    path = train_and_save_bundle(
        args.ticker,
        benchmark=args.benchmark,
        period=args.period,
        out_dir=args.out_dir,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
