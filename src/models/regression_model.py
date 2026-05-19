"""XGBoost regressor for next-session adjusted close (price-level target)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

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
    Huber transition in the same units as ``y`` (dollars for adj. close).

    ``reg:pseudohubererror`` expects ``huber_slope`` near the typical target scale;
    values ≪ price level (e.g. 1–20) cause unstable fits on raw prices.
    """
    y = _to_numpy_y(y_train)
    med = float(np.median(y))
    return max(med, 1.0)


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
        slope = 10.0
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
