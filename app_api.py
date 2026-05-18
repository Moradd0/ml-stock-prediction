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
    load_model_bundle,
    run_buy_and_hold_backtest,
    run_open_to_close_backtest,
)
from src.features import engineer_features
from src.preprocess import to_float_numpy

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(title="Tech stock direction API", version="0.1.0")


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
                f"PYTHONPATH=. python -m src.train_bundle {req.ticker.upper()}"
            ),
        )

    bundle = load_model_bundle(path)
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
    proba_up = float(model.predict_proba(xs)[0, 1])
    pred_cls = int(model.predict(xs)[0])
    as_of = last["Date"][0]
    as_of_str = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)

    # --- Hold-out backtest (same 70/15/15 split as training); reuse one feature build ---
    feat_eval = feat.filter(pl.col("Target_Direction").is_not_null())
    fe = append_next_session_prices(feat_eval, raw)
    n = fe.height
    i_train, i_val_end = _split_test_mask(n)
    if i_val_end >= n:
        raise HTTPException(status_code=400, detail="Not enough rows for train/val/test split.")

    test_df = fe[i_val_end:].drop_nulls(
        subset=feature_columns + ["next_open", "next_close", "Target_Direction"]
    )
    if test_df.height == 0:
        raise HTTPException(status_code=400, detail="Empty test window after alignment.")

    X_test = to_float_numpy(test_df.select(feature_columns))
    y_test = test_df["Target_Direction"].to_numpy()
    next_open = test_df["next_open"].to_numpy()
    next_close = test_df["next_close"].to_numpy()

    y_pred = model.predict(scaler.transform(X_test)).astype(int)
    metrics = classification_metrics_summary(y_test, y_pred)
    bt_occ = run_open_to_close_backtest(y_pred, next_open, next_close, initial_capital=10_000.0)
    bt_bh = run_buy_and_hold_backtest(test_df["Date"], raw, initial_capital=10_000.0)

    dates = [str(d) for d in test_df["Date"].to_list()]

    return {
        "ticker": req.ticker.upper(),
        "benchmark": bm,
        "period": period,
        "as_of_date": as_of_str,
        "prediction": "Up" if pred_cls == 1 else "Down",
        "probability_up": round(proba_up, 6),
        "test_classification_metrics": metrics,
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
