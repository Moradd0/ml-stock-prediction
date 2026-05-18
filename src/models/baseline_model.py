"""XGBoost baseline classifier with early stopping and feature-importance plots."""

from __future__ import annotations

import logging
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def init_xgb_classifier(
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
    **kwargs: Any,
) -> XGBClassifier:
    """
    Build an ``XGBClassifier`` for binary ``Target_Direction`` (``binary:logistic``).

    Additional keyword arguments are forwarded to ``XGBClassifier`` so you can override
    defaults without editing this function.
    """
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
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
    params.update(kwargs)
    return XGBClassifier(**params)


def _to_numpy_X(X: np.ndarray | pl.DataFrame) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return X.to_numpy().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def _to_numpy_y(y: np.ndarray | pl.Series) -> np.ndarray:
    if isinstance(y, pl.Series):
        return y.to_numpy()
    return np.asarray(y).ravel()


def train_xgb_baseline(
    clf: XGBClassifier,
    X_train: np.ndarray | pl.DataFrame,
    y_train: np.ndarray | pl.Series,
    X_val: np.ndarray | pl.DataFrame,
    y_val: np.ndarray | pl.Series,
    *,
    feature_names: list[str] | None = None,
    early_stopping_rounds: int = 20,
) -> XGBClassifier:
    """
    Fit ``clf`` on training data with validation-based early stopping on log-loss.

    Logs hyperparameters (``get_params``) and the best / final validation log-loss via
    :mod:`logging`. Does not configure handlers; configure ``logging.basicConfig`` in your
    entrypoint if you want console output.

    ``feature_names`` is optional metadata for :func:`plot_xgb_feature_importance`; training
    uses NumPy arrays so PyArrow is not required for XGBoost.
    """
    Xt_tr = _to_numpy_X(X_train)
    Xt_va = _to_numpy_X(X_val)
    if feature_names is not None and Xt_tr.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} != number of columns {Xt_tr.shape[1]}."
        )
    yt_tr = _to_numpy_y(y_train)
    yt_va = _to_numpy_y(y_val)

    logger.info("XGBoost baseline hyperparameters: %s", clf.get_params(deep=False))

    clf.set_params(early_stopping_rounds=early_stopping_rounds)
    logger.info(
        "Early stopping: early_stopping_rounds=%s",
        clf.get_params().get("early_stopping_rounds"),
    )
    clf.fit(
        Xt_tr,
        yt_tr,
        eval_set=[(Xt_va, yt_va)],
        verbose=False,
    )

    evals = clf.evals_result()
    val_key = next(iter(evals)) if evals else None
    if val_key and "logloss" in evals[val_key]:
        losses = evals[val_key]["logloss"]
        best_iter = getattr(clf, "best_iteration", None)
        best_loss = min(losses) if losses else float("nan")
        final_loss = losses[-1] if losses else float("nan")
        logger.info(
            "Validation log-loss (best=%.6f at iter=%s, final=%.6f, n_trees=%s)",
            best_loss,
            best_iter,
            final_loss,
            getattr(clf, "n_estimators", None),
        )
        # sklearn log_loss on validation for a single scalar audit line
        try:
            y_proba = clf.predict_proba(Xt_va)[:, 1]
            ll = log_loss(yt_va, y_proba, labels=[0, 1])
            logger.info(" sklearn.metrics.log_loss on validation set: %.6f", ll)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not compute sklearn log_loss: %s", exc)
    else:
        logger.warning("evals_result missing logloss keys: %s", evals)

    return clf


def plot_xgb_feature_importance(
    clf: XGBClassifier,
    *,
    feature_names: list[str] | None = None,
    importance_type: Literal["gain", "weight", "cover"] = "gain",
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot XGBoost feature importance (``gain``, ``weight``, or ``cover``).

    Uses the native booster scores when available; otherwise falls back to
    ``feature_importances_`` from the sklearn wrapper (gain-like).
    """
    booster = clf.get_booster()
    scores = booster.get_score(importance_type=importance_type)
    if not scores:
        imp = getattr(clf, "feature_importances_", None)
        if imp is None:
            raise RuntimeError("No feature importances available from this model.")
        names = (
            feature_names
            if feature_names is not None
            else [f"f{i}" for i in range(len(imp))]
        )
        pairs = list(zip(names, imp.astype(float), strict=True))
    else:
        def _name(fkey: str) -> str:
            if fkey.startswith("f") and fkey[1:].isdigit():
                idx = int(fkey[1:])
                if feature_names is not None and idx < len(feature_names):
                    return feature_names[idx]
            return fkey

        pairs = [(_name(k), float(v)) for k, v in scores.items()]

    pairs.sort(key=lambda kv: kv[1], reverse=True)
    pairs = pairs[:top_n]
    if not pairs:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_title("XGBoost feature importance (empty)")
        fig.tight_layout()
        return fig, ax
    labels, values = zip(*pairs, strict=True)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels))))
    else:
        fig = ax.figure

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(importance_type)
    ax.set_title(f"XGBoost feature importance ({importance_type})")
    fig.tight_layout()
    return fig, ax
