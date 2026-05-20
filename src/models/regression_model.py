"""XGBoost regressor for next-session adjusted close (price-level target)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.evaluate import tune_direction_threshold

logger = logging.getLogger(__name__)

# Training objectives supported by compare-losses / init_xgb_regressor.
LOSS_CONFIGS: dict[str, dict[str, Any]] = {
    "mae": {
        "label": "Absolute error (MAE)",
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "bundle_loss": "mae",
    },
    "squared": {
        "label": "Squared error (RMSE)",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "bundle_loss": "rmse",
    },
    "huber": {
        "label": "Huber (pseudo-Huber)",
        "objective": "reg:pseudohubererror",
        "eval_metric": "mphe",
        "bundle_loss": "huber",
    },
}


def default_huber_slope(y_train: np.ndarray | pl.Series) -> float:
    """
    Huber transition in the same units as ``y``.

    For ``Target_Next_Return`` (simple returns), use ~typical daily |return|;
    for raw prices use median price (legacy).
    """
    y = _to_numpy_y(y_train)
    med_abs = float(np.median(np.abs(y)))
    if med_abs < 0.5:
        return max(med_abs, 1e-4)
    return max(med_abs, 1.0)


# Hyperparameter grid: tuned on validation return MAE, then direction threshold on val.
TUNE_PARAM_GRID: list[dict[str, Any]] = [
    {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 5.0, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0, "min_child_weight": 3},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0, "min_child_weight": 1},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "min_child_weight": 1},
    {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 2.0, "min_child_weight": 2},
    {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "min_child_weight": 1},
]

# Among configs within this factor of the best validation MAE, pick best direction BA.
TUNE_MAE_TOLERANCE = 1.08


def init_xgb_regressor(
    *,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    random_state: int = 42,
    tree_method: str = "hist",
    n_jobs: int = -1,
    objective: str = "reg:absoluteerror",
    eval_metric: str | None = None,
    huber_slope: float | None = None,
    **kwargs: Any,
) -> XGBRegressor:
    """
    Regressor for ``Target_Next_Adj_Close``.

    ``objective`` examples: ``reg:absoluteerror`` (MAE), ``reg:squarederror`` (RMSE),
    ``reg:pseudohubererror`` (smooth Huber; set ``huber_slope`` in dollars).
    """
    if eval_metric is None:
        eval_metric = {
            "reg:absoluteerror": "mae",
            "reg:squarederror": "rmse",
            "reg:pseudohubererror": "mphe",
        }.get(objective, "mae")
    params = dict(
        objective=objective,
        eval_metric=eval_metric,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        random_state=random_state,
        tree_method=tree_method,
        n_jobs=n_jobs,
    )
    if objective == "reg:pseudohubererror" and huber_slope is not None:
        params["huber_slope"] = huber_slope
    params.update(kwargs)
    return XGBRegressor(**params)


def init_xgb_regressor_for_loss(
    loss_key: str,
    *,
    y_train_for_huber: np.ndarray | pl.Series | None = None,
    huber_slope: float | None = None,
    **kwargs: Any,
) -> XGBRegressor:
    """Build a regressor from a key in :data:`LOSS_CONFIGS` (``mae``, ``squared``, ``huber``)."""
    if loss_key not in LOSS_CONFIGS:
        raise ValueError(f"Unknown loss {loss_key!r}; choose from {list(LOSS_CONFIGS)}.")
    cfg = LOSS_CONFIGS[loss_key]
    slope = huber_slope
    if loss_key == "huber" and slope is None and y_train_for_huber is not None:
        slope = default_huber_slope(y_train_for_huber)
    elif loss_key == "huber" and slope is None:
        slope = 0.01
    return init_xgb_regressor(
        objective=cfg["objective"],
        eval_metric=cfg["eval_metric"],
        huber_slope=slope if loss_key == "huber" else None,
        **kwargs,
    )


def _to_numpy_X(X: np.ndarray | pl.DataFrame) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return X.to_numpy().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def _to_numpy_y(y: np.ndarray | pl.Series) -> np.ndarray:
    if isinstance(y, pl.Series):
        return y.to_numpy()
    return np.asarray(y, dtype=np.float64).ravel()


def train_xgb_regressor(
    reg: XGBRegressor,
    X_train: np.ndarray | pl.DataFrame,
    y_train: np.ndarray | pl.Series,
    X_val: np.ndarray | pl.DataFrame,
    y_val: np.ndarray | pl.Series,
    *,
    feature_names: list[str] | None = None,
    early_stopping_rounds: int = 20,
) -> XGBRegressor:
    """Fit with validation early stopping on MAE (or RMSE for squared-error objective)."""
    Xt_tr = _to_numpy_X(X_train)
    Xt_va = _to_numpy_X(X_val)
    if feature_names is not None and Xt_tr.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} != number of columns {Xt_tr.shape[1]}."
        )
    yt_tr = _to_numpy_y(y_train)
    yt_va = _to_numpy_y(y_val)

    logger.info("XGBoost regressor hyperparameters: %s", reg.get_params(deep=False))
    reg.set_params(early_stopping_rounds=early_stopping_rounds)
    reg.fit(Xt_tr, yt_tr, eval_set=[(Xt_va, yt_va)], verbose=False)

    evals = reg.evals_result()
    val_key = next(iter(evals)) if evals else None
    metric_key = reg.get_params().get("eval_metric", "mae")
    if isinstance(metric_key, (list, tuple)):
        metric_key = metric_key[0]
    if val_key and metric_key in evals[val_key]:
        losses = evals[val_key][metric_key]
        best_loss = min(losses) if losses else float("nan")
        final_loss = losses[-1] if losses else float("nan")
        logger.info(
            "Validation %s (best=%.6f, final=%.6f, best_iter=%s)",
            metric_key,
            best_loss,
            final_loss,
            getattr(reg, "best_iteration", None),
        )
        try:
            y_hat = reg.predict(Xt_va)
            mae = mean_absolute_error(yt_va, y_hat)
            logger.info(" sklearn.metrics.mean_absolute_error on validation: %.6f", mae)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compute sklearn MAE: %s", exc)

    return reg


def fit_huber_regressor_tuned(
    X_train: np.ndarray | pl.DataFrame,
    y_train: np.ndarray | pl.Series,
    X_val: np.ndarray | pl.DataFrame,
    y_val: np.ndarray | pl.Series,
    *,
    feature_names: list[str] | None = None,
    tune: bool = True,
    early_stopping_rounds: int = 20,
) -> tuple[XGBRegressor, dict[str, Any]]:
    """
    Huber regressor with hyperparameter search on validation return MAE.

    Among configs within :data:`TUNE_MAE_TOLERANCE` of the best MAE, keeps the one with
    highest validation balanced direction accuracy (after :func:`tune_direction_threshold`).
    The winning threshold is stored in the returned metadata for inference/backtest.
    """
    slope = default_huber_slope(y_train)
    grid = TUNE_PARAM_GRID if tune else [TUNE_PARAM_GRID[3]]
    y_val_np = _to_numpy_y(y_val)
    y_val_dir = (y_val_np > 0.0).astype(int)

    candidates: list[tuple[float, float, XGBRegressor, dict[str, Any], float, dict[str, Any]]] = []

    for params in grid:
        reg = init_xgb_regressor_for_loss(
            "huber",
            huber_slope=slope,
            y_train_for_huber=y_train,
            n_estimators=500,
            **params,
        )
        train_xgb_regressor(
            reg,
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names=feature_names,
            early_stopping_rounds=early_stopping_rounds,
        )
        pred = reg.predict(_to_numpy_X(X_val))
        mae = float(mean_absolute_error(y_val_np, pred))
        thresh, thresh_meta = tune_direction_threshold(y_val_dir, pred)
        ba = float(thresh_meta.get("val_balanced_accuracy", float("nan")))
        candidates.append((mae, ba, reg, dict(params), thresh, thresh_meta))

    if not candidates:
        raise RuntimeError("Hyperparameter search did not fit any model.")

    best_mae = min(c[0] for c in candidates)
    mae_cutoff = best_mae * TUNE_MAE_TOLERANCE
    eligible = [c for c in candidates if c[0] <= mae_cutoff]
    eligible.sort(key=lambda c: (-c[1], c[0]))
    _, best_ba, best_reg, best_params, direction_threshold, thresh_meta = eligible[0]

    mean_pred_val = float(np.mean(best_reg.predict(_to_numpy_X(X_val))))
    meta: dict[str, Any] = {
        "best_params": best_params,
        "huber_slope": slope,
        "val_mae": float(mean_absolute_error(y_val_np, best_reg.predict(_to_numpy_X(X_val)))),
        "val_mae_best_in_grid": best_mae,
        "val_balanced_accuracy": best_ba,
        "direction_threshold": direction_threshold,
        "direction_threshold_meta": thresh_meta,
        "val_mean_predicted_return": mean_pred_val,
        "loss": "huber",
    }
    logger.info(
        "Selected params %s (val MAE=%.6f, val BA=%.4f, direction_threshold=%.6f, huber_slope=%.6f)",
        best_params,
        meta["val_mae"],
        best_ba,
        direction_threshold,
        slope,
    )
    return best_reg, meta
