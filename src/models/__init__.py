"""Model building blocks (XGBoost baseline). Import ``dl_utils`` only when PyTorch is installed."""

from src.models.baseline_model import (
    init_xgb_classifier,
    plot_xgb_feature_importance,
    train_xgb_baseline,
)

__all__ = [
    "init_xgb_classifier",
    "plot_xgb_feature_importance",
    "train_xgb_baseline",
]
