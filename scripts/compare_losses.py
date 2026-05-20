#!/usr/bin/env python3
"""Train/evaluate XGBoost regressors with MAE, squared (RMSE), and Huber objectives."""

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
    classification_metrics_summary,
    direction_from_return_predictions,
    return_metrics_summary,
)
from src.features import (  # noqa: E402
    TARGET_DIRECTION_COLUMN,
    TARGET_PRICE_COLUMN,
    TARGET_RETURN_COLUMN,
    engineer_features,
)
from src.models.regression_model import (  # noqa: E402
    LOSS_CONFIGS,
    default_huber_slope,
    init_xgb_regressor_for_loss,
    train_xgb_regressor,
)
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


def evaluate_loss(
    ticker: str,
    benchmark: str,
    period: str,
    loss_key: str,
    *,
    huber_slope: float | None = None,
) -> dict:
    raw = fetch_aligned_market_data(ticker.upper(), benchmark.upper(), period=period)
    feat = engineer_features(raw, keep_incomplete_target=False, target_ticker=ticker.upper())
    X, y = split_features_and_target(feat, target_column=TARGET_RETURN_COLUMN)
    feature_columns = list(X.columns)
    n = feat.height
    if n < 50:
        return {"loss": loss_key, "error": "too few rows after features"}

    X_train, X_val, X_test, y_train, y_val, y_test = chronological_train_val_test_split(X, y)
    Xtr, Xva, Xte, scaler = scale_train_val_test(X_train, X_val, X_test)

    slope = huber_slope
    if loss_key == "huber" and slope is None:
        slope = default_huber_slope(y_train)

    reg = init_xgb_regressor_for_loss(
        loss_key,
        y_train_for_huber=y_train,
        huber_slope=slope,
    )
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
        + ["today_adj_close", "next_adj_close", TARGET_PRICE_COLUMN]
    )
    if test_df.height == 0:
        return {"loss": loss_key, "error": "empty test window"}

    X_eval = scaler.transform(to_float_numpy(test_df.select(feature_columns)))
    y_pred = reg.predict(X_eval).astype(float)
    y_true = test_df[TARGET_RETURN_COLUMN].to_numpy().astype(float)
    y_true_dir = test_df[TARGET_DIRECTION_COLUMN].to_numpy().astype(int)

    reg_m = return_metrics_summary(y_true, y_pred)
    dir_m = classification_metrics_summary(
        y_true_dir, direction_from_return_predictions(y_pred)
    )

    cfg = LOSS_CONFIGS[loss_key]
    row = {
        "loss": loss_key,
        "loss_label": cfg["label"],
        "objective": cfg["objective"],
        "eval_metric": cfg["eval_metric"],
        "test_rows": test_df.height,
        "test_mae_return_pct": reg_m["mae_return_pct"],
        "test_mape_pct": reg_m["mape_pct"],
        "test_r2": reg_m["r2"],
        "direction_accuracy": dir_m["accuracy"],
    }
    if loss_key == "huber":
        row["huber_slope"] = float(slope) if slope is not None else None
    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare XGBoost regression losses (MAE vs squared vs Huber) on the same split."
    )
    p.add_argument("ticker", nargs="?", default="MSFT")
    p.add_argument("--benchmark", default="QQQ")
    p.add_argument("--period", default="5y")
    p.add_argument(
        "--losses",
        default="mae,squared,huber",
        help="Comma-separated: mae, squared, huber",
    )
    p.add_argument(
        "--huber-slope",
        type=float,
        default=None,
        help="Huber transition in price units (default: ~5%% of median train target)",
    )
    p.add_argument("--out", default=None, help="Optional JSON path for results")
    args = p.parse_args()

    keys = [x.strip() for x in args.losses.split(",") if x.strip()]
    unknown = [k for k in keys if k not in LOSS_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown losses: {unknown}; choose from {list(LOSS_CONFIGS)}")

    rows: list[dict] = []
    sym = args.ticker.upper()
    print(f"\n{sym}  period={args.period}  benchmark={args.benchmark.upper()}\n")
    print(
        f"{'loss':<10} {'objective':<28} {'mae_ret%':>10} {'dir_acc':>8} {'huber_slope':>12}"
    )
    print("-" * 82)

    for key in keys:
        try:
            row = evaluate_loss(
                args.ticker,
                args.benchmark,
                args.period,
                key,
                huber_slope=args.huber_slope,
            )
        except Exception as exc:
            row = {"loss": key, "error": str(exc)}
        rows.append(row)
        if "error" in row:
            print(f"{key:<10} ERROR: {row['error']}")
        else:
            hs = row.get("huber_slope")
            hs_s = f"{hs:.2f}" if hs is not None else ""
            print(
                f"{key:<10} {row['objective']:<28} {row['test_mae_return_pct']:10.4f} "
                f"{row['direction_accuracy']:8.3f} {hs_s:>12}"
            )

    print("\n(All rows: test MAE on next-day return, in percent points.)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
