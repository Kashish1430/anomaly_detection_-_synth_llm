from __future__ import annotations

import numpy as np
import pandas as pd


def time_ordered_split(
    timestamps: pd.Series, train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits into three contiguous, time-ordered blocks: train / validation /
    test. Not random - PLAN.md §07 treats a shuffled split as a false signal
    for a problem where behaviour drifts over time. Returns integer positions
    into the original (unsorted) array, so callers can index any DataFrame
    that shares that row order.
    """
    order = np.argsort(timestamps.to_numpy())
    n = len(order)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return order[:train_end], order[train_end:val_end], order[val_end:]


def expanding_window_splits(
    timestamps: pd.Series, n_folds: int = 4, min_train_frac: float = 0.4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward time-based CV: fold i always trains on everything up to a
    cutoff and tests on the next contiguous slice after it (an expanding
    window). This is what PLAN.md §07 means by "time-based cross-validation" -
    a model that only looks good under random k-fold shuffling, but falls
    apart here, would have failed silently in production.

    Returns integer positions into the original (unsorted) `timestamps`
    array, same convention as `time_ordered_split`.
    """
    order = np.argsort(timestamps.to_numpy())
    n = len(order)
    first_test_start = int(n * min_train_frac)
    remaining = n - first_test_start
    fold_size = remaining // n_folds
    if fold_size <= 0:
        raise ValueError("not enough rows to form the requested number of folds")

    splits = []
    for i in range(n_folds):
        test_start = first_test_start + i * fold_size
        test_end = n if i == n_folds - 1 else test_start + fold_size
        train_idx, test_idx = order[:test_start], order[test_start:test_end]
        if len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits
