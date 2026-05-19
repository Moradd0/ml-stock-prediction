#!/usr/bin/env python3
"""Train and evaluate the same pipeline on multiple Yahoo ``period`` values (2y, 5y, 10y, 15y)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.project_env import load_project_env

load_project_env()

from src.data_fetch import fetch_aligned_market_data  # noqa: E402
from src.evaluate import (  # noqa: E402
    append_next_session_prices,
    direction_accuracy_from_prices,
    regression_metrics_summary,
    run_buy_and_hold_backtest,
    run_open_to_close_backtest,
    save_model_bundle,
)
from src.features import TARGET_PRICE_COLUMN, engineer_features  # noqa: E402
from src.models.regression_model import init_xgb_regressor, train_xgb_regressor  # noqa: E402
from src.preprocess import (  # noqa: E402
    chronological_train_val_test_split,
    scale_train_val_test,
    split_features_and_target,
    to_float_numpy,
)


def _split_test_mask(n_rows: int) -> tuple[int, int]:
    i_train = int(n_rows * 0.7)
    i_val_end = i_train + int(n_rows * 0.15)
    return i_train, i_val_end


def evaluate_period(
    ticker: str,
    benchmark: str,
    period: str,
    *,
    save_bundle: bool = False,
    out_dir: str | Path = "models",
) -> dict:
    raw = fetch_aligned_market_data(ticker.upper(), benchmark.upper(), period=period)
    feat = engineer_features(raw, keep_incomplete_target=False, target_ticker=ticker.upper())
    X, y = split_features_and_target(feat, target_column=TARGET_PRICE_COLUMN)
    feature_columns = list(X.columns)
    n = feat.height
    if n < 50:
        return {"period": period, "error": "too few rows after features"}

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

    i_train, i_val_end = _split_test_mask(n)
    fe = append_next_session_prices(feat, raw)
    test_df = fe[i_val_end:].drop_nulls(
        subset=feature_columns
        + ["today_adj_close", "next_adj_close", "next_open", "next_close", TARGET_PRICE_COLUMN]
    )
    if test_df.height == 0:
        return {"period": period, "error": "empty test window"}

    X_eval = scaler.transform(to_float_numpy(test_df.select(feature_columns)))
    y_pred_price = reg.predict(X_eval).astype(float)
    y_true_price = test_df[TARGET_PRICE_COLUMN].to_numpy().astype(float)
    today_adj = test_df["today_adj_close"].to_numpy().astype(float)
    reg_m = regression_metrics_summary(y_true_price, y_pred_price)
    dir_m = direction_accuracy_from_prices(y_true_price, y_pred_price, today_adj)
    y_pred_dir = (y_pred_price > today_adj).astype(int)
    bt_occ = run_open_to_close_backtest(
        y_pred_dir,
        test_df["next_open"].to_numpy(),
        test_df["next_close"].to_numpy(),
    )
    bt_bh = run_buy_and_hold_backtest(test_df["Date"], raw)

    dates = test_df["Date"]
    if save_bundle:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"bundle_{ticker.upper()}_{period}.joblib"
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
        print(f"Saved {path}")

    return {
        "period": period,
        "feature_rows": n,
        "test_rows": test_df.height,
        "test_start": str(dates[0]),
        "test_end": str(dates[-1]),
        "test_mae": reg_m["mae"],
        "test_rmse": reg_m["rmse"],
        "direction_accuracy": dir_m["accuracy"],
        "f1_up": dir_m["f1_score"],
        "return_strategy": bt_occ["cumulative_return_strategy"],
        "return_open_to_close_benchmark": bt_occ["cumulative_return_open_to_close_benchmark"],
        "return_buy_and_hold": bt_bh["cumulative_return_buy_and_hold"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare train/eval windows by Yahoo period.")
    p.add_argument("ticker", nargs="?", default="MSFT")
    p.add_argument("--benchmark", default="QQQ")
    p.add_argument(
        "--periods",
        default="2y,5y,10y,15y",
        help="Comma-separated Yahoo periods",
    )
    p.add_argument(
        "--save-bundles",
        action="store_true",
        help="Also write models/bundle_{TICKER}_{period}.joblib for each run",
    )
    args = p.parse_args()
    periods = [x.strip() for x in args.periods.split(",") if x.strip()]
    rows: list[dict] = []

    for period in periods:
        print(f"\n=== {args.ticker.upper()} period={period} ===", flush=True)
        try:
            row = evaluate_period(
                args.ticker,
                args.benchmark,
                period,
                save_bundle=args.save_bundles,
            )
        except Exception as exc:
            row = {"period": period, "error": str(exc)}
        rows.append(row)
        print(json.dumps(row, indent=2))

    print("\n--- summary ---")
    for r in rows:
        if "error" in r:
            print(f"{r['period']}: ERROR {r['error']}")
        else:
            print(
                f"{r['period']}: mae={r['test_mae']:.3f} dir_acc={r['direction_accuracy']:.3f} f1={r['f1_up']:.3f} "
                f"test={r['test_start']}..{r['test_end']} "
                f"strat={r['return_strategy']*100:.1f}% B&H={r['return_buy_and_hold']*100:.1f}%"
            )


if __name__ == "__main__":
    main()
