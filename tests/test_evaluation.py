from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.splits import expanding_window_splits, time_ordered_split
from evaluation.stats import precision_confidence_interval, two_proportion_ztest
from evaluation.threshold_tuning import rates_at_threshold, select_threshold_by_ztest


def _timestamps(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))


def test_time_ordered_split_is_contiguous_and_chronological():
    ts = _timestamps(100)
    train_idx, val_idx, test_idx = time_ordered_split(ts, train_frac=0.7, val_frac=0.15)

    assert len(train_idx) == 70
    assert len(val_idx) == 15
    assert len(test_idx) == 15
    # every train timestamp precedes every val timestamp precedes every test timestamp
    assert ts.iloc[train_idx].max() <= ts.iloc[val_idx].min()
    assert ts.iloc[val_idx].max() <= ts.iloc[test_idx].min()


def test_time_ordered_split_handles_unsorted_input():
    ts = _timestamps(20)
    shuffled = ts.sample(frac=1.0, random_state=0).reset_index(drop=True)
    train_idx, val_idx, test_idx = time_ordered_split(shuffled, train_frac=0.5, val_frac=0.25)

    assert shuffled.iloc[train_idx].max() <= shuffled.iloc[val_idx].min()
    assert shuffled.iloc[val_idx].max() <= shuffled.iloc[test_idx].min()


def test_expanding_window_splits_train_always_precedes_test():
    ts = _timestamps(200)
    splits = expanding_window_splits(ts, n_folds=4, min_train_frac=0.4)

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert ts.iloc[train_idx].max() <= ts.iloc[test_idx].min()

    # expanding window: each fold's train set is a superset of the previous fold's
    for (train_a, _), (train_b, _) in zip(splits, splits[1:], strict=False):
        assert len(train_b) > len(train_a)
        assert set(train_a).issubset(set(train_b))


def test_precision_confidence_interval_contains_point_estimate():
    lower, upper = precision_confidence_interval(n_flagged=100, n_true_positive=30)
    assert lower < 0.30 < upper


def test_precision_confidence_interval_narrows_with_more_data():
    small_lower, small_upper = precision_confidence_interval(n_flagged=20, n_true_positive=6)
    large_lower, large_upper = precision_confidence_interval(
        n_flagged=20_000, n_true_positive=6_000
    )
    assert (large_upper - large_lower) < (small_upper - small_lower)


def test_two_proportion_ztest_detects_a_real_difference():
    # 500/1000 vs 100/1000 - not a subtle difference, should be extremely significant
    z_stat, p_value = two_proportion_ztest(count1=500, nobs1=1000, count2=100, nobs2=1000)
    assert p_value < 0.001
    assert z_stat > 0  # proportion1 > proportion2


def test_two_proportion_ztest_no_difference_is_not_significant():
    z_stat, p_value = two_proportion_ztest(count1=100, nobs1=1000, count2=102, nobs2=1000)
    assert p_value > 0.5


def test_rates_at_threshold_hand_crafted():
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])

    result = rates_at_threshold(y_true, y_score, threshold=0.5)
    assert result["n_flagged"] == 3  # scores 0.9, 0.8, 0.7
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)


def test_select_threshold_by_ztest_prefers_lower_fp_at_equal_recall():
    rng = np.random.default_rng(0)
    n = 5000
    y_true = (rng.random(n) < 0.05).astype(int)
    # score correlates with label but isn't perfect, so there's a real
    # precision/recall tradeoff across the threshold grid
    y_score = y_true * rng.uniform(0.5, 1.0, n) + (1 - y_true) * rng.uniform(0.0, 0.6, n)

    baseline_threshold = float(np.quantile(y_score, 0.95))
    thresholds = np.quantile(y_score, np.linspace(0.80, 0.999, 40))

    result = select_threshold_by_ztest(y_true, y_score, thresholds, baseline_threshold)

    assert result["chosen_metrics"]["recall"] >= result["baseline_metrics"]["recall"] - 1e-9
    assert result["chosen_metrics"]["fp_rate"] <= result["baseline_metrics"]["fp_rate"] + 1e-9
    assert 0.0 <= result["p_value"] <= 1.0
