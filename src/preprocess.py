"""Prepare engineered features for ML: X/y split, chronological splits, leakage-safe scaling."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from src.features import TARGET_LABEL_COLUMNS, TARGET_RETURN_COLUMN

_RAW_FIELD_NAMES = frozenset({"Open", "High", "Low", "Close", "Volume", "Adj Close"})


def _is_raw_price_column(name: str) -> bool:
    if name in _RAW_FIELD_NAMES:
        return True
    if "_" in name:
        _, right = name.rsplit("_", 1)
        if right in _RAW_FIELD_NAMES:
            return True
    return False


def _raw_price_columns(names: Iterable[str]) -> list[str]:
    return [c for c in names if _is_raw_price_column(c)]


def split_features_and_target(
    df: pl.DataFrame,
    *,
    target_column: str = TARGET_RETURN_COLUMN,
    date_column: str = "Date",
) -> tuple[pl.DataFrame, pl.Series]:
    """
    Split ``df`` into feature matrix ``X`` and target ``y``.

    ``X`` excludes **all** label columns (``TARGET_LABEL_COLUMNS``), the date column, and raw
    OHLCV columns so models never see tomorrow's close or direction as inputs.

    Rows are sorted by ``date_column`` when that column exists, so row order is chronological.
    """
    if target_column not in df.columns:
        raise ValueError(f"Missing target column {target_column!r}.")

    ordered = df.sort(date_column) if date_column in df.columns else df

    drop_cols = set(TARGET_LABEL_COLUMNS)
    drop_cols.add(target_column)
    if date_column in ordered.columns:
        drop_cols.add(date_column)
    drop_cols.update(_raw_price_columns(ordered.columns))

    feature_names = [c for c in ordered.columns if c not in drop_cols]
    if not feature_names:
        raise ValueError(
            "No feature columns left after dropping target, date, and raw prices."
        )

    leaked = [c for c in TARGET_LABEL_COLUMNS if c in feature_names]
    if leaked:
        raise ValueError(f"Label columns must not be model inputs: {leaked}")

    X = ordered.select(feature_names)
    y = ordered[target_column]
    return X, y


def chronological_train_val_test_split(
    X: pl.DataFrame,
    y: pl.Series,
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series, pl.Series, pl.Series]:
    """
    Split ``X`` and ``y`` in strict time order (no shuffling).

    Uses the first ``train_fraction`` of rows for training, the next ``val_fraction`` for
    validation, and the remainder for testing. Indices are computed from ``int(n * frac)``
    so the test block absorbs any rounding remainder.
    """
    if not math.isclose(
        train_fraction + val_fraction + test_fraction, 1.0, rel_tol=1e-6
    ):
        raise ValueError(
            "train_fraction, val_fraction, and test_fraction must sum to 1.0; "
            f"got {train_fraction + val_fraction + test_fraction}."
        )

    n = X.height
    if y.len() != n:
        raise ValueError(f"Length mismatch: X has {n} rows, y has {y.len()}.")

    i_train = int(n * train_fraction)
    i_val_end = i_train + int(n * val_fraction)

    if i_train == 0 or i_val_end <= i_train or i_val_end >= n:
        raise ValueError(
            f"Split boundaries invalid for n={n}: train_end={i_train}, val_end={i_val_end}. "
            "Need more rows or different fractions."
        )

    X_train = X.slice(0, i_train)
    X_val = X.slice(i_train, i_val_end - i_train)
    X_test = X.slice(i_val_end, n - i_val_end)

    y_train = y.slice(0, i_train)
    y_val = y.slice(i_train, i_val_end - i_train)
    y_test = y.slice(i_val_end, n - i_val_end)

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_float_numpy(X: pl.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return X.to_numpy().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def scale_train_val_test(
    X_train: pl.DataFrame | np.ndarray,
    X_val: pl.DataFrame | np.ndarray,
    X_test: pl.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standard-scale features using training statistics only.

    Fits ``StandardScaler`` on ``X_train``, then transforms train, validation, and test so
    validation and test rows never influence the mean or variance.
    """
    scaler = StandardScaler()
    Xt_train = to_float_numpy(X_train)
    Xt_val = to_float_numpy(X_val)
    Xt_test = to_float_numpy(X_test)

    scaler.fit(Xt_train)
    return (
        scaler.transform(Xt_train),
        scaler.transform(Xt_val),
        scaler.transform(Xt_test),
        scaler,
    )
