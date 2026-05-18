"""PyTorch Dataset and DataLoaders for overlapping sequence windows over tabular features."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceTabularDataset(Dataset):
    """
    Sliding windows over rows of a 2D feature matrix.

    For end index ``t`` (inclusive window ``[t - sequence_length + 1, t]``), the label is
    ``y[t + 1]`` (next-row target). Requires ``sequence_length <= n_rows`` and produces
    ``n_rows - sequence_length`` samples (no lookahead into future *features* beyond the
    aligned label convention).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y).ravel(), dtype=torch.long)
        self.seq_len = sequence_length
        n = int(self.X.shape[0])
        if self.y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows.")
        if n < sequence_length + 1:
            raise ValueError(
                f"Need at least sequence_length + 1 rows (got n={n}, sequence_length={sequence_length})."
            )
        # End indices t: window X[t-seq_len+1 : t+1], label y[t+1]
        self._starts = torch.arange(0, n - sequence_length, dtype=torch.long)

    def __len__(self) -> int:
        return int(self._starts.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self._starts[idx].item())
        end = start + self.seq_len  # exclusive end index; slice [start:end] has length seq_len
        x_seq = self.X[start:end]
        y_next = self.y[end]
        return x_seq, y_next


def _to_numpy_X(X: np.ndarray | pl.DataFrame) -> np.ndarray:
    if isinstance(X, pl.DataFrame):
        return X.to_numpy().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _to_numpy_y(y: np.ndarray | pl.Series) -> np.ndarray:
    if isinstance(y, pl.Series):
        return y.to_numpy()
    return np.asarray(y).ravel()


def make_sequence_dataloaders(
    X_train: np.ndarray | pl.DataFrame,
    y_train: np.ndarray | pl.Series,
    X_val: np.ndarray | pl.DataFrame,
    y_val: np.ndarray | pl.Series,
    X_test: np.ndarray | pl.DataFrame,
    y_test: np.ndarray | pl.Series,
    sequence_length: int,
    *,
    batch_size: int = 64,
    train_shuffle: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build ``DataLoader``s with overlapping windows.

    Validation and test loaders always use ``shuffle=False``. Training defaults to
    ``shuffle=True`` (batch order only; each batch item remains a contiguous time window).
    """
    train_ds = SequenceTabularDataset(_to_numpy_X(X_train), _to_numpy_y(y_train), sequence_length)
    val_ds = SequenceTabularDataset(_to_numpy_X(X_val), _to_numpy_y(y_val), sequence_length)
    test_ds = SequenceTabularDataset(_to_numpy_X(X_test), _to_numpy_y(y_test), sequence_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
