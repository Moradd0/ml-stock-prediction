"""FastAPI service: fetch → features → scaled inference + optional test-set backtest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.project_env import load_project_env

load_project_env()

from src.data_fetch import fetch_aligned_market_data
from src.evaluate import (
    append_next_session_prices,
    build_backtest_summary,
    classification_metrics_summary,
    direction_accuracy_from_prices,
    direction_from_return_predictions,
    load_model_bundle,
    prediction_counts_summary,
    prices_from_returns,
    regression_metrics_summary,
    regression_to_user_outputs,
    return_metrics_summary,
    return_to_user_outputs,
    run_buy_and_hold_backtest,
    run_open_to_close_backtest,
)
from src.features import (
    TARGET_DIRECTION_COLUMN,
    TARGET_PRICE_COLUMN,
    TARGET_RETURN_COLUMN,
    engineer_features,
)
from src.preprocess import to_float_numpy

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(title="Tech stock price forecast API", version="0.3.0")


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Target equity ticker", examples=["MSFT"])
    benchmark: str = Field(default="QQQ", description="Sector benchmark ticker")


def _bundle_path(ticker: str) -> Path:
    return MODELS_DIR / f"bundle_{ticker.upper()}.joblib"


def _split_test_mask(n_rows: int) -> tuple[int, int]:
    i_train = int(n_rows * 0.7)
    i_val_end = i_train + int(n_rows * 0.15)
    return i_train, i_val_end


def _bundle_direction_threshold(bundle: dict[str, Any]) -> float:
    if "direction_threshold" in bundle:
        return float(bundle["direction_threshold"])
    meta = bundle.get("tune_meta") or {}
    return float(meta.get("direction_threshold", 0.0))


def _predict_user_outputs(
    model,
    xs: np.ndarray,
    today_adj: float,
    *,
    target_mode: str,
    direction_threshold: float = 0.0,
) -> dict[str, Any]:
    if target_mode == "return":
        pred_return = float(model.predict(xs)[0])
        return return_to_user_outputs(
            pred_return, today_adj, direction_threshold=direction_threshold
        )
    pred_price = float(model.predict(xs)[0])
    return regression_to_user_outputs(pred_price, today_adj)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def run_predict(req: PredictRequest) -> dict[str, Any]:
    path = _bundle_path(req.ticker)
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained bundle at {path}. Run: python scripts/run.py train {req.ticker.upper()}"
            ),
        )

    bundle = load_model_bundle(path)
    task = str(bundle.get("task", "regression"))
    target_mode = str(bundle.get("target_mode", "return"))
    target_col = str(bundle.get("target_column", TARGET_RETURN_COLUMN))
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns: list[str] = list(bundle["feature_columns"])

    bm = str(bundle.get("benchmark", req.benchmark)).upper()
    period = str(bundle.get("period", "10y"))
    raw = fetch_aligned_market_data(req.ticker.upper(), bm, period=period)
    feat = engineer_features(raw, keep_incomplete_target=True, target_ticker=req.ticker.upper())
    last = feat.tail(1)
    if last.height == 0:
        raise HTTPException(status_code=400, detail="Could not build features for the latest row.")

    X_last = to_float_numpy(last.select(feature_columns))
    xs = scaler.transform(X_last)
    raw_sorted = raw.sort("Date")
    today_adj = float(raw_sorted["target_Adj Close"][-1])

    direction_threshold = _bundle_direction_threshold(bundle)

    if task == "regression":
        user = _predict_user_outputs(
            model,
            xs,
            today_adj,
            target_mode=target_mode,
            direction_threshold=direction_threshold,
        )
    else:
        proba_up = float(model.predict_proba(xs)[0, 1])
        pred_cls = int(model.predict(xs)[0])
        user = {
            "prediction": "Up" if pred_cls == 1 else "Down",
            "direction": pred_cls,
            "predicted_next_close": None,
            "predicted_pct_change": None,
            "probability_up": round(proba_up, 6),
        }

    as_of = last["Date"][0]
    as_of_str = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)

    label_col = TARGET_RETURN_COLUMN if target_col == TARGET_RETURN_COLUMN else target_col
    feat_eval = feat.filter(pl.col(label_col).is_not_null())
    fe = append_next_session_prices(feat_eval, raw)
    n = fe.height
    i_train, i_val_end = _split_test_mask(n)
    if i_val_end >= n:
        raise HTTPException(status_code=400, detail="Not enough rows for train/val/test split.")

    test_cols = feature_columns + [
        "today_adj_close",
        "next_adj_close",
        TARGET_PRICE_COLUMN,
        TARGET_RETURN_COLUMN,
        TARGET_DIRECTION_COLUMN,
        "next_open",
        "next_close",
    ]
    test_df = fe[i_val_end:].drop_nulls(subset=test_cols)
    if test_df.height == 0:
        raise HTTPException(status_code=400, detail="Empty test window after alignment.")

    X_test = to_float_numpy(test_df.select(feature_columns))
    today_test = test_df["today_adj_close"].to_numpy().astype(float)
    y_true_price = test_df[TARGET_PRICE_COLUMN].to_numpy().astype(float)
    y_true_return = test_df[TARGET_RETURN_COLUMN].to_numpy().astype(float)
    y_true_dir = test_df[TARGET_DIRECTION_COLUMN].to_numpy().astype(int)
    next_open = test_df["next_open"].to_numpy()
    next_close = test_df["next_close"].to_numpy()

    initial_capital = 10_000.0

    if task == "regression":
        y_pred_raw = model.predict(scaler.transform(X_test)).astype(float)
        if target_mode == "return":
            y_pred_return = y_pred_raw
            y_pred_price = prices_from_returns(today_test, y_pred_return)
            ret_metrics = return_metrics_summary(y_true_return, y_pred_return)
            reg_metrics = regression_metrics_summary(y_true_price, y_pred_price)
            y_pred_dir = direction_from_return_predictions(
                y_pred_return, threshold=direction_threshold
            )
            dir_metrics = classification_metrics_summary(y_true_dir, y_pred_dir)
        else:
            y_pred_price = y_pred_raw
            y_pred_return = np.where(today_test > 0, y_pred_price / today_test - 1.0, 0.0)
            ret_metrics = return_metrics_summary(y_true_return, y_pred_return)
            reg_metrics = regression_metrics_summary(y_true_price, y_pred_price)
            y_pred_dir = (y_pred_price > today_test).astype(int)
            dir_metrics = direction_accuracy_from_prices(y_true_price, y_pred_price, today_test)
    else:
        y_pred_dir = model.predict(scaler.transform(X_test)).astype(int)
        ret_metrics = {}
        reg_metrics = {}
        dir_metrics = classification_metrics_summary(y_true_dir, y_pred_dir)

    bt_occ = run_open_to_close_backtest(
        y_pred_dir, next_open, next_close, initial_capital=initial_capital
    )
    bt_bh = run_buy_and_hold_backtest(test_df["Date"], raw, initial_capital=initial_capital)
    pred_counts = prediction_counts_summary(y_pred_dir)
    backtest_summary = build_backtest_summary(
        bt_occ,
        bt_bh,
        initial_capital=initial_capital,
        test_sessions=test_df.height,
        prediction_counts=pred_counts,
        direction_threshold=direction_threshold if task == "regression" and target_mode == "return" else None,
    )
    dates = [str(d) for d in test_df["Date"].to_list()]

    return {
        "ticker": req.ticker.upper(),
        "benchmark": bm,
        "period": period,
        "task": task,
        "target_mode": target_mode,
        "loss": bundle.get("loss", "huber"),
        "tune_meta": bundle.get("tune_meta"),
        "direction_threshold": direction_threshold,
        "as_of_date": as_of_str,
        "today_adj_close": round(today_adj, 4),
        **user,
        "test_return_metrics": ret_metrics,
        "test_regression_metrics": reg_metrics,
        "test_direction_metrics": dir_metrics,
        "backtest_summary": backtest_summary,
        "backtest": {
            **bt_occ,
            **bt_bh,
            "dates": dates,
            "initial_capital": initial_capital,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    return run_predict(req)
