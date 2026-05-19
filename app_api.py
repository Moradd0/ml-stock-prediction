"""FastAPI service: fetch → features → scaled inference + optional test-set backtest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.project_env import load_project_env

load_project_env()

from src.data_fetch import fetch_aligned_market_data
from src.evaluate import (
    append_next_session_prices,
    classification_metrics_summary,
    direction_accuracy_from_prices,
    load_model_bundle,
    regression_metrics_summary,
    regression_to_user_outputs,
    run_buy_and_hold_backtest,
    run_open_to_close_backtest,
)
from src.features import TARGET_DIRECTION_COLUMN, TARGET_PRICE_COLUMN, engineer_features
from src.preprocess import to_float_numpy

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(title="Tech stock price forecast API", version="0.2.0")


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Target equity ticker", examples=["MSFT"])
    benchmark: str = Field(default="QQQ", description="Sector benchmark ticker")


def _bundle_path(ticker: str) -> Path:
    return MODELS_DIR / f"bundle_{ticker.upper()}.joblib"


def _split_test_mask(n_rows: int) -> tuple[int, int]:
    """Same boundaries as :func:`src.preprocess.chronological_train_val_test_split`."""
    i_train = int(n_rows * 0.7)
    i_val_end = i_train + int(n_rows * 0.15)
    return i_train, i_val_end


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def run_predict(req: PredictRequest) -> dict[str, Any]:
    """
    Same payload as ``POST /predict`` (for CLI or scripts). Raises :class:`HTTPException` on errors.
    """
    path = _bundle_path(req.ticker)
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained bundle at {path}. From the project root run: "
                f"python scripts/run.py train {req.ticker.upper()}"
            ),
        )

    bundle = load_model_bundle(path)
    task = str(bundle.get("task", "regression"))
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns: list[str] = list(bundle["feature_columns"])
    target_col = str(bundle.get("target_column", TARGET_PRICE_COLUMN))

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

    if task == "regression":
        pred_price = float(model.predict(xs)[0])
        user = regression_to_user_outputs(pred_price, today_adj)
    else:
        # Legacy classification bundles
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

    feat_eval = feat.filter(pl.col(target_col).is_not_null())
    fe = append_next_session_prices(feat_eval, raw)
    n = fe.height
    i_train, i_val_end = _split_test_mask(n)
    if i_val_end >= n:
        raise HTTPException(status_code=400, detail="Not enough rows for train/val/test split.")

    test_cols = feature_columns + [
        "today_adj_close",
        "next_adj_close",
        "next_open",
        "next_close",
        target_col,
        TARGET_DIRECTION_COLUMN,
    ]
    test_df = fe[i_val_end:].drop_nulls(subset=test_cols)
    if test_df.height == 0:
        raise HTTPException(status_code=400, detail="Empty test window after alignment.")

    X_test = to_float_numpy(test_df.select(feature_columns))
    y_true_price = test_df[target_col].to_numpy().astype(float)
    today_test = test_df["today_adj_close"].to_numpy().astype(float)
    next_open = test_df["next_open"].to_numpy()
    next_close = test_df["next_close"].to_numpy()

    if task == "regression":
        y_pred_price = model.predict(scaler.transform(X_test)).astype(float)
        reg_metrics = regression_metrics_summary(y_true_price, y_pred_price)
        dir_metrics = direction_accuracy_from_prices(y_true_price, y_pred_price, today_test)
        y_pred_dir = (y_pred_price > today_test).astype(int)
    else:
        y_pred_dir = model.predict(scaler.transform(X_test)).astype(int)
        reg_metrics = {}
        y_true_price = test_df[TARGET_DIRECTION_COLUMN].to_numpy()
        dir_metrics = classification_metrics_summary(y_true_price, y_pred_dir)

    bt_occ = run_open_to_close_backtest(y_pred_dir, next_open, next_close, initial_capital=10_000.0)
    bt_bh = run_buy_and_hold_backtest(test_df["Date"], raw, initial_capital=10_000.0)
    dates = [str(d) for d in test_df["Date"].to_list()]

    return {
        "ticker": req.ticker.upper(),
        "benchmark": bm,
        "period": period,
        "task": task,
        "loss": bundle.get("loss", "mae"),
        "as_of_date": as_of_str,
        "today_adj_close": round(today_adj, 4),
        **user,
        "test_regression_metrics": reg_metrics,
        "test_direction_metrics": dir_metrics,
        "backtest": {
            **bt_occ,
            **bt_bh,
            "dates": dates,
            "initial_capital": 10_000.0,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    return run_predict(req)
