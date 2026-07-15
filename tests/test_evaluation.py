from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.fairness import (
    flagging_rate_by_group,
    parity_tests_vs_reference,
    performance_by_group,
)
from evaluation.monitoring import population_stability_index, psi_report
from evaluation.sensitivity import decision_flip_rate, feature_sensitivity, perturb_feature
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


def test_flagging_rate_by_group_hand_crafted():
    group = pd.Series(["retail", "retail", "retail", "retail", "sme", "sme"])
    flagged = pd.Series([True, True, False, False, True, False])

    result = flagging_rate_by_group(flagged, group).set_index("group")

    assert result.loc["retail", "n"] == 4
    assert result.loc["retail", "n_flagged"] == 2
    assert result.loc["retail", "flagging_rate"] == pytest.approx(0.5)
    assert result.loc["sme", "n"] == 2
    assert result.loc["sme", "flagging_rate"] == pytest.approx(0.5)
    # CI should bracket the point estimate for both groups
    assert result.loc["retail", "ci_low"] < 0.5 < result.loc["retail", "ci_high"]


def test_flagging_rate_by_group_ci_narrows_with_more_data():
    small_group = pd.Series(["a"] * 20)
    small_flagged = pd.Series([True] * 6 + [False] * 14)
    large_group = pd.Series(["a"] * 20_000)
    large_flagged = pd.Series([True] * 6_000 + [False] * 14_000)

    small = flagging_rate_by_group(small_flagged, small_group).iloc[0]
    large = flagging_rate_by_group(large_flagged, large_group).iloc[0]

    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])


def test_parity_tests_vs_reference_detects_a_real_difference():
    # reference group flags 10%, "high" group flags 50% - not subtle
    group = pd.Series(["reference"] * 1000 + ["high"] * 1000)
    flagged = pd.Series([True] * 100 + [False] * 900 + [True] * 500 + [False] * 500)

    result = parity_tests_vs_reference(flagged, group, reference="reference").set_index("group")

    assert result.loc["high", "significant"]
    assert result.loc["high", "p_value"] < 0.001
    assert result.loc["high", "rate_diff"] == pytest.approx(0.4)
    # the reference group itself should not appear as a comparison row
    assert "reference" not in result.index


def test_parity_tests_vs_reference_no_difference_is_not_significant():
    group = pd.Series(["reference"] * 1000 + ["other"] * 1000)
    flagged = pd.Series([True] * 100 + [False] * 900 + [True] * 102 + [False] * 898)

    result = parity_tests_vs_reference(flagged, group, reference="reference").set_index("group")

    assert not result.loc["other", "significant"]
    assert result.loc["other", "p_value"] > 0.5


def test_parity_tests_vs_reference_raises_for_unknown_reference():
    group = pd.Series(["a", "b"])
    flagged = pd.Series([True, False])

    with pytest.raises(ValueError, match="not present"):
        parity_tests_vs_reference(flagged, group, reference="does_not_exist")


def test_performance_by_group_hand_crafted():
    group = pd.Series(["a"] * 5)
    y_true = pd.Series([True, True, False, False, False])
    flagged = pd.Series([True, False, True, False, False])

    result = performance_by_group(y_true, flagged, group).set_index("group")

    assert result.loc["a", "n"] == 5
    assert result.loc["a", "true_anomaly_rate"] == pytest.approx(2 / 5)
    assert result.loc["a", "flagging_rate"] == pytest.approx(2 / 5)
    assert result.loc["a", "precision"] == pytest.approx(0.5)  # 1 tp / 2 flagged
    assert result.loc["a", "recall"] == pytest.approx(0.5)  # 1 tp / 2 actual positive


def test_performance_by_group_reveals_bias_masked_by_equal_true_rate():
    # both groups have the SAME true anomaly rate (20%) - so a flagging-rate
    # disparity here can't be explained by "group b just has more anomalies",
    # unlike a raw flagging-rate comparison would suggest without this check
    group = pd.Series(["a"] * 10 + ["b"] * 10)
    y_true = pd.Series([True] * 2 + [False] * 8 + [True] * 2 + [False] * 8)
    # group a: flags exactly the 2 true anomalies (perfect precision/recall)
    # group b: flags the 2 true anomalies AND 4 false ones (over-flagged)
    flagged = pd.Series(
        [True, True] + [False] * 8 + [True, True, True, True, True, True] + [False] * 4
    )

    result = performance_by_group(y_true, flagged, group).set_index("group")

    assert result.loc["a", "true_anomaly_rate"] == pytest.approx(
        result.loc["b", "true_anomaly_rate"]
    )
    assert result.loc["a", "recall"] == pytest.approx(1.0)
    assert result.loc["b", "recall"] == pytest.approx(1.0)
    # same underlying anomaly prevalence and same recall, but group b's
    # precision is dragged down by extra false positives - a real disparity,
    # not one explained away by "group b has more true anomalies"
    assert result.loc["b", "precision"] < result.loc["a", "precision"]


def test_perturb_feature_shifts_by_std_and_leaves_others_untouched():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 10.0, 10.0, 10.0, 10.0]})
    std_a = X["a"].std()

    perturbed = perturb_feature(X, "a", delta_in_std=1.0)

    assert perturbed["a"].to_numpy() == pytest.approx((X["a"] + std_a).to_numpy())
    assert (perturbed["b"] == X["b"]).all()
    assert (X["a"] == [1.0, 2.0, 3.0, 4.0, 5.0]).all()  # original untouched


def test_feature_sensitivity_ranks_more_influential_feature_first():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return 5.0 * df["a"].to_numpy() + 0.1 * df["b"].to_numpy()

    result = feature_sensitivity(predict_fn, X, ["a", "b"], delta_in_std=1.0)

    assert result.iloc[0]["feature"] == "a"
    by_feature = result.set_index("feature")
    assert by_feature.loc["a", "mean_abs_delta_up"] > by_feature.loc["b", "mean_abs_delta_up"]


def test_decision_flip_rate_zero_when_scores_never_cross_threshold():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.9)

    flip_rate = decision_flip_rate(predict_fn, X, threshold=0.5, noise_std_frac=0.5)
    assert flip_rate == 0.0


def test_decision_flip_rate_positive_when_scores_sit_at_threshold():
    X = pd.DataFrame({"a": np.linspace(4.9, 5.1, 1000)})

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return df["a"].to_numpy()

    flip_rate = decision_flip_rate(predict_fn, X, threshold=5.0, noise_std_frac=0.1)
    assert 0.0 < flip_rate < 1.0


def test_population_stability_index_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 5000)
    same_dist = rng.normal(0, 1, 5000)

    psi = population_stability_index(baseline, same_dist)
    assert psi < 0.05


def test_population_stability_index_handles_boolean_arrays():
    # feature columns like is_round_amount arrive as bool dtype - np.quantile
    # can't subtract booleans, so this would previously raise a TypeError
    expected = np.array([True, False, False, False, False] * 200)
    actual = np.array([True, True, False, False, False] * 200)

    psi = population_stability_index(expected, actual)
    assert psi >= 0.0


def test_population_stability_index_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 5000)
    shifted = rng.normal(3, 1, 5000)  # not a subtle shift

    psi = population_stability_index(baseline, shifted)
    assert psi > 0.25


def test_psi_report_labels_stability_status_correctly():
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame(
        {
            "stable_feature": rng.normal(0, 1, 5000),
            "drifted_feature": rng.normal(0, 1, 5000),
        }
    )
    current = pd.DataFrame(
        {
            "stable_feature": rng.normal(0, 1, 5000),
            "drifted_feature": rng.normal(4, 1, 5000),
        }
    )
    columns = ["stable_feature", "drifted_feature"]

    result = psi_report(baseline, current, columns).set_index("feature")
    assert result.loc["stable_feature", "stability_status"] == "stable"
    assert result.loc["drifted_feature", "stability_status"] == "significant_shift"

    ranked = psi_report(baseline, current, columns)
    assert ranked.iloc[0]["feature"] == "drifted_feature"  # sorted by severity


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
