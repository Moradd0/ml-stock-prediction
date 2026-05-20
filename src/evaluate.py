"""Evaluation: classification metrics and simple open-to-close backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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


def return_to_user_outputs(
    predicted_return: float,
    today_adj_close: float,
    *,
    max_abs_return: float = 0.15,
    direction_threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Map predicted next-day simple return to price / direction (UI).

    ``max_abs_return`` clamps the return used for display (default ±15%) for plausibility.
    Direction uses ``predicted_return > direction_threshold`` (threshold tuned on validation).
    """
    today = float(today_adj_close)
    r = float(predicted_return)
    r_display = float(np.clip(r, -max_abs_return, max_abs_return))
    pred_price = today * (1.0 + r_display) if today > 0 else 0.0
    pct = r_display * 100.0
    direction = 1 if r > direction_threshold else 0
    out = {
        "predicted_next_return": round(r, 6),
        "predicted_next_close": round(pred_price, 4),
        "predicted_pct_change": round(pct, 4),
        "prediction": "Up" if direction == 1 else "Down",
        "direction": direction,
    }
    if abs(r - r_display) > 1e-9:
        out["display_clamped"] = True
    return out


def return_metrics_summary(
    y_true_return: np.ndarray | pl.Series,
    y_pred_return: np.ndarray | pl.Series,
) -> dict[str, Any]:
    """MAE / RMSE on simple returns; MAE in percent points (×100)."""
    yt = np.asarray(y_true_return, dtype=float).ravel()
    yp = np.asarray(y_pred_return, dtype=float).ravel()
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "mae_return": mae,
        "mae_return_pct": mae * 100.0,
        "rmse_return": rmse,
        "mean_signed_error_return": float(np.mean(err)),
    }


def direction_from_return_predictions(
    y_pred_return: np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Up (1) if predicted return is strictly above ``threshold``."""
    return (np.asarray(y_pred_return, dtype=float).ravel() > float(threshold)).astype(int)


def prediction_counts_summary(y_pred_dir: np.ndarray | pl.Series) -> dict[str, Any]:
    """Counts of Up / Down predictions for backtest reporting."""
    y = np.asarray(y_pred_dir, dtype=int).ravel()
    n = int(y.size)
    n_up = int(np.sum(y == 1))
    n_down = int(np.sum(y == 0))
    return {
        "n_predicted_up": n_up,
        "n_predicted_down": n_down,
        "n_total": n,
        "pct_predicted_up": round(100.0 * n_up / n, 2) if n else 0.0,
        "pct_predicted_down": round(100.0 * n_down / n, 2) if n else 0.0,
    }


def threshold_for_predicted_up_rate(
    y_pred_return: np.ndarray | pl.Series,
    target_up_frac: float,
) -> float:
    """Threshold ``t`` so about ``target_up_frac`` of rows have ``pred_return > t`` (ties-aware)."""
    y = np.sort(np.asarray(y_pred_return, dtype=float).ravel())
    n = y.size
    if n == 0:
        return 0.0
    n_up = int(round(float(target_up_frac) * n))
    n_up = max(1, min(n - 1, n_up))
    idx = n - n_up - 1
    t = float(y[idx])
    if idx + 1 < n and y[idx + 1] <= t:
        t = np.nextafter(t, -np.inf)
    return t


def tune_direction_threshold(
    y_true_dir: np.ndarray | pl.Series,
    y_pred_return: np.ndarray | pl.Series,
    *,
    up_rate_tolerance: float = 0.12,
    min_up_frac: float = 0.20,
    max_up_frac: float = 0.80,
) -> tuple[float, dict[str, Any]]:
    """
    Pick ``threshold`` so ``pred_return > threshold`` maximizes validation balanced accuracy.

    Keeps the predicted Up rate near the validation label Up rate (± ``up_rate_tolerance``),
    so the model does not collapse to always-Up or always-Down. Falls back to the prediction
    median if no percentile threshold qualifies.
    """
    yt = np.asarray(y_true_dir, dtype=int).ravel()
    yp = np.asarray(y_pred_return, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true_dir and y_pred_return must have the same length.")
    n = yp.size
    if n == 0:
        return 0.0, {"val_balanced_accuracy": float("nan"), "n_candidates": 0}

    true_up_frac = float(np.mean(yt == 1))
    up_lo = max(min_up_frac, true_up_frac - up_rate_tolerance)
    up_hi = min(max_up_frac, true_up_frac + up_rate_tolerance)

    percentiles = np.linspace(5.0, 95.0, 19)
    candidates = sorted(
        {float(np.percentile(yp, p)) for p in percentiles}
        | {0.0, threshold_for_predicted_up_rate(yp, true_up_frac)}
    )

    best_t = 0.0
    best_ba = -1.0
    best_counts: dict[str, Any] = {}

    for t in candidates:
        pred = (yp > t).astype(int)
        counts = prediction_counts_summary(pred)
        up_frac = counts["n_predicted_up"] / n
        if up_frac < up_lo or up_frac > up_hi:
            continue
        ba = float(balanced_accuracy_score(yt, pred))
        if ba > best_ba:
            best_ba = ba
            best_t = t
            best_counts = counts

    if best_ba < 0.0:
        best_t = threshold_for_predicted_up_rate(yp, true_up_frac)
        pred = (yp > best_t).astype(int)
        best_ba = float(balanced_accuracy_score(yt, pred))
        best_counts = prediction_counts_summary(pred)
        fallback = "quantile_match_label_rate"
    else:
        fallback = None

    meta = {
        "val_balanced_accuracy": best_ba,
        "val_accuracy": float(accuracy_score(yt, (yp > best_t).astype(int))),
        "val_true_up_frac": round(true_up_frac, 4),
        "val_target_up_frac_range": [round(up_lo, 4), round(up_hi, 4)],
        "n_candidates": len(candidates),
        "threshold_fallback": fallback,
        **best_counts,
    }
    return best_t, meta


def prices_from_returns(today_adj_close: np.ndarray, y_return: np.ndarray) -> np.ndarray:
    today = np.asarray(today_adj_close, dtype=float).ravel()
    r = np.asarray(y_return, dtype=float).ravel()
    return today * (1.0 + r)


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
    final_s = float(eq_s[-1])
    final_b = float(eq_benchmark[-1])

    return {
        "daily_returns_strategy": r_strat.astype(float).tolist(),
        "daily_returns_open_to_close_benchmark": r_benchmark.astype(float).tolist(),
        "equity_strategy": eq_s.astype(float).tolist(),
        "equity_open_to_close_benchmark": eq_benchmark.astype(float).tolist(),
        "cumulative_return_strategy": float(final_s / initial_capital - 1.0),
        "cumulative_return_open_to_close_benchmark": float(final_b / initial_capital - 1.0),
        "final_capital_strategy": final_s,
        "final_capital_open_to_close_benchmark": final_b,
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

    final_bh = float(eq[-1])
    return {
        "buy_and_hold_entry_date": entry_d.isoformat() if hasattr(entry_d, "isoformat") else str(entry_d),
        "buy_and_hold_exit_date": exit_d.isoformat() if hasattr(exit_d, "isoformat") else str(exit_d),
        "buy_and_hold_entry_price": entry_p,
        "buy_and_hold_exit_price": exit_p,
        "cumulative_return_buy_and_hold": float(exit_p / entry_p - 1.0),
        "final_capital_buy_and_hold": final_bh,
        "equity_buy_and_hold": eq.astype(float).tolist(),
        "sharpe_annualized_buy_and_hold": annualized_sharpe(
            r_cc[1:], periods_per_year=periods_per_year
        ),
        "max_drawdown_buy_and_hold": max_drawdown_from_equity(eq),
    }


def build_backtest_summary(
    bt_model: dict[str, Any],
    bt_buy_hold: dict[str, Any],
    *,
    initial_capital: float = 10_000.0,
    test_sessions: int | None = None,
    prediction_counts: dict[str, Any] | None = None,
    direction_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Human-readable backtest block for API / CLI (percent returns and final capital).
    """
    ret_s = float(bt_model.get("cumulative_return_strategy", 0.0)) * 100.0
    ret_bh = float(bt_buy_hold.get("cumulative_return_buy_and_hold", 0.0)) * 100.0
    ret_occ = float(bt_model.get("cumulative_return_open_to_close_benchmark", 0.0)) * 100.0

    strategies = [
        {
            "name": "Model strategy",
            "description": "If predicted Up: earn next session open→close return; else cash.",
            "return_pct": round(ret_s, 2),
            "final_capital": round(float(bt_model.get("final_capital_strategy", initial_capital)), 2),
        },
        {
            "name": "Buy and hold",
            "description": "Hold target stock from first to last test session (adj. close).",
            "return_pct": round(ret_bh, 2),
            "final_capital": round(
                float(bt_buy_hold.get("final_capital_buy_and_hold", initial_capital)), 2
            ),
        },
        {
            "name": "Open-to-close every day",
            "description": "Benchmark: take every next-session open→close return (always invested intraday).",
            "return_pct": round(ret_occ, 2),
            "final_capital": round(
                float(bt_model.get("final_capital_open_to_close_benchmark", initial_capital)), 2
            ),
        },
    ]
    out: dict[str, Any] = {
        "initial_capital": initial_capital,
        "strategies": strategies,
    }
    if test_sessions is not None:
        out["test_sessions"] = test_sessions
    if prediction_counts is not None:
        out["prediction_counts"] = prediction_counts
    if direction_threshold is not None:
        out["direction_threshold"] = round(float(direction_threshold), 8)
    return out


def save_model_bundle(path: str | Path, bundle: dict[str, Any]) -> Path:
    """Persist training artifacts (expects keys ``model``, ``scaler``, ``feature_columns``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)
    return p


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a bundle written by :func:`save_model_bundle`."""
    return joblib.load(Path(path))
