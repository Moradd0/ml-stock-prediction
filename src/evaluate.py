"""Evaluation: classification metrics and simple open-to-close backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def regression_metrics_summary(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
) -> dict[str, Any]:
    """Price-level errors: MAE (primary loss), RMSE, MAPE, R²."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.maximum(np.abs(yt), 1e-12)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
        "mean_signed_error": float(np.mean(err)),
    }


def regression_to_user_outputs(
    predicted_next_close: float,
    today_adj_close: float,
) -> dict[str, Any]:
    """
    Map a predicted next-session adj. close to UI fields.

    ``predicted_pct_change`` is percent move from today's close to the prediction.
    """
    today = float(today_adj_close)
    pred = float(predicted_next_close)
    if today <= 0:
        pct = 0.0
    else:
        pct = (pred / today - 1.0) * 100.0
    direction = 1 if pred > today else 0
    return {
        "predicted_next_close": round(pred, 4),
        "predicted_pct_change": round(pct, 4),
        "prediction": "Up" if direction == 1 else "Down",
        "direction": direction,
    }


def direction_accuracy_from_prices(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    today_adj_close: np.ndarray,
) -> dict[str, float]:
    """Compare direction implied by predicted vs true next close (vs today's close)."""
    today = np.asarray(today_adj_close, dtype=float).ravel()
    yt = np.asarray(y_true_price, dtype=float).ravel()
    yp = np.asarray(y_pred_price, dtype=float).ravel()
    true_dir = (yt > today).astype(int)
    pred_dir = (yp > today).astype(int)
    return classification_metrics_summary(true_dir, pred_dir)


def classification_metrics_summary(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
) -> dict[str, Any]:
    """
    Standard binary classification metrics plus a confusion matrix (nested list).

    Labels are assumed to be ``0`` / ``1`` (``Target_Direction`` convention).
    """
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1]).tolist(),
    }


def append_next_session_prices(feat: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """
    Join next session ``target_Open`` / ``target_Close`` onto each feature row (by ``Date``).

    For session ``Date = D``, ``next_open`` / ``next_close`` are the target stock's open and
    close on the following row of ``raw`` (next trading session).
    """
    r = raw.sort("Date").select(
        [
            "Date",
            pl.col("target_Adj Close").alias("today_adj_close"),
            pl.col("target_Adj Close").shift(-1).alias("next_adj_close"),
            pl.col("target_Open").shift(-1).alias("next_open"),
            pl.col("target_Close").shift(-1).alias("next_close"),
        ]
    )
    return feat.join(r, on="Date", how="left")


def annualized_sharpe(
    daily_returns: np.ndarray,
    *,
    periods_per_year: int = 252,
) -> float:
    """Sample Sharpe of per-period simple returns, scaled by ``sqrt(periods_per_year)``."""
    r = np.asarray(daily_returns, dtype=float).ravel()
    if r.size < 2:
        return float("nan")
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    if sig == 0.0 or not np.isfinite(sig):
        return float("nan")
    return float(np.sqrt(periods_per_year) * mu / sig)


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    """Maximum drawdown (most negative peak-to-trough) from an equity level series."""
    eq = np.asarray(equity, dtype=float).ravel()
    if eq.size == 0:
        return float("nan")
    roll_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(roll_max, 1e-12) - 1.0
    return float(np.min(dd))


def run_open_to_close_backtest(
    predictions: np.ndarray | pl.Series,
    next_open: np.ndarray,
    next_close: np.ndarray,
    *,
    initial_capital: float = 10_000.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Simulate model strategy vs an always-intraday open→close benchmark.

    For each session in the evaluation window:

    * **Strategy:** if ``prediction == 1``, earn the next session's simple return from
      ``next_open`` to ``next_close``; if ``0``, earn ``0`` (cash).
    * **Open-to-close benchmark:** earn that same next-session open-to-close return every
      day (not classical buy-and-hold; see :func:`run_buy_and_hold_backtest`).

    ``predictions[i]`` must align with ``next_open[i]`` / ``next_close[i]`` (same length).
    """
    pred = np.asarray(predictions).astype(int).ravel()
    po = np.asarray(next_open, dtype=float).ravel()
    pc = np.asarray(next_close, dtype=float).ravel()
    if not (pred.size == po.size == pc.size):
        raise ValueError("predictions, next_open, and next_close must have the same length.")
    with np.errstate(divide="ignore", invalid="ignore"):
        r_occ = np.where(po > 0, (pc - po) / po, 0.0)
    r_occ = np.nan_to_num(r_occ, nan=0.0, posinf=0.0, neginf=0.0)
    r_benchmark = r_occ.copy()
    r_strat = np.where(pred == 1, r_occ, 0.0)

    eq_s = initial_capital * np.cumprod(1.0 + r_strat)
    eq_benchmark = initial_capital * np.cumprod(1.0 + r_benchmark)

    return {
        "daily_returns_strategy": r_strat.astype(float).tolist(),
        "daily_returns_open_to_close_benchmark": r_benchmark.astype(float).tolist(),
        "equity_strategy": eq_s.astype(float).tolist(),
        "equity_open_to_close_benchmark": eq_benchmark.astype(float).tolist(),
        "cumulative_return_strategy": float(eq_s[-1] / initial_capital - 1.0),
        "cumulative_return_open_to_close_benchmark": float(
            eq_benchmark[-1] / initial_capital - 1.0
        ),
        "sharpe_annualized_strategy": annualized_sharpe(r_strat, periods_per_year=periods_per_year),
        "max_drawdown_strategy": max_drawdown_from_equity(eq_s),
    }


def run_buy_and_hold_backtest(
    session_dates: np.ndarray | pl.Series,
    raw: pl.DataFrame,
    *,
    price_col: str = "target_Adj Close",
    initial_capital: float = 10_000.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Classical buy-and-hold over the evaluation window.

    Buy at the **first** session's ``price_col``, hold through close-to-close moves, mark to
    market on each session date, and sell at the **last** session's price. Total return
    is ``exit_price / entry_price - 1`` (same as the final marked equity vs initial capital).
    """
    dates = pl.DataFrame({"Date": pl.Series(session_dates)}).unique().sort("Date")
    if dates.height < 2:
        return {
            "buy_and_hold_entry_date": None,
            "buy_and_hold_exit_date": None,
            "buy_and_hold_entry_price": float("nan"),
            "buy_and_hold_exit_price": float("nan"),
            "cumulative_return_buy_and_hold": float("nan"),
            "equity_buy_and_hold": [],
            "sharpe_annualized_buy_and_hold": float("nan"),
            "max_drawdown_buy_and_hold": float("nan"),
        }

    if price_col not in raw.columns:
        raise ValueError(f"Column {price_col!r} not found in raw market frame.")

    window = (
        raw.sort("Date")
        .select(["Date", price_col])
        .join(dates, on="Date", how="inner")
    )
    px = window[price_col].to_numpy().astype(float)
    if px.size < 2 or px[0] <= 0:
        return {
            "buy_and_hold_entry_date": str(window["Date"][0]),
            "buy_and_hold_exit_date": str(window["Date"][-1]),
            "buy_and_hold_entry_price": float(px[0]) if px.size else float("nan"),
            "buy_and_hold_exit_price": float(px[-1]) if px.size else float("nan"),
            "cumulative_return_buy_and_hold": float("nan"),
            "equity_buy_and_hold": [],
            "sharpe_annualized_buy_and_hold": float("nan"),
            "max_drawdown_buy_and_hold": float("nan"),
        }

    entry_d = window["Date"][0]
    exit_d = window["Date"][-1]
    entry_p = float(px[0])
    exit_p = float(px[-1])
    eq = initial_capital * (px / entry_p)
    r_cc = np.zeros(px.size, dtype=float)
    r_cc[1:] = px[1:] / px[:-1] - 1.0

    return {
        "buy_and_hold_entry_date": entry_d.isoformat() if hasattr(entry_d, "isoformat") else str(entry_d),
        "buy_and_hold_exit_date": exit_d.isoformat() if hasattr(exit_d, "isoformat") else str(exit_d),
        "buy_and_hold_entry_price": entry_p,
        "buy_and_hold_exit_price": exit_p,
        "cumulative_return_buy_and_hold": float(exit_p / entry_p - 1.0),
        "equity_buy_and_hold": eq.astype(float).tolist(),
        "sharpe_annualized_buy_and_hold": annualized_sharpe(
            r_cc[1:], periods_per_year=periods_per_year
        ),
        "max_drawdown_buy_and_hold": max_drawdown_from_equity(eq),
    }


def save_model_bundle(path: str | Path, bundle: dict[str, Any]) -> Path:
    """Persist training artifacts (expects keys ``model``, ``scaler``, ``feature_columns``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)
    return p


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a bundle written by :func:`save_model_bundle`."""
    return joblib.load(Path(path))
