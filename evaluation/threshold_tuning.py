from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.stats import two_proportion_ztest


def rates_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    flagged = y_score >= threshold
    n_flagged = int(flagged.sum())
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    n_negative = int((y_true == 0).sum())
    n_positive = int((y_true == 1).sum())
    return {
        "threshold": float(threshold),
        "n_flagged": n_flagged,
        "tp": tp,
        "fp": fp,
        "n_negative": n_negative,
        "precision": tp / n_flagged if n_flagged else 0.0,
        "recall": tp / n_positive if n_positive else 0.0,
        "fp_rate": fp / n_negative if n_negative else 0.0,
    }


def threshold_sweep(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame([rates_at_threshold(y_true, y_score, t) for t in thresholds])


def select_threshold_by_ztest(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    baseline_threshold: float,
    alpha: float = 0.05,
) -> dict:
    """Among candidates that recall at least as well as `baseline_threshold`,
    picks the one with the lowest false-positive rate, and tests whether that
    FP-rate improvement over the baseline is statistically significant via a
    two-proportion z-test - PLAN.md §07's "tuning decision thresholds through
    hypothesis testing", not just eyeballing a sweep table for the smallest
    number.
    """
    baseline = rates_at_threshold(y_true, y_score, baseline_threshold)
    sweep = threshold_sweep(y_true, y_score, thresholds)

    eligible = sweep[sweep["recall"] >= baseline["recall"]]
    if eligible.empty:
        eligible = sweep

    best = eligible.sort_values("fp_rate").iloc[0]

    z_stat, p_value = two_proportion_ztest(
        count1=baseline["fp"],
        nobs1=baseline["n_negative"],
        count2=int(best["fp"]),
        nobs2=int(best["n_negative"]),
    )
    is_improvement = best["fp_rate"] < baseline["fp_rate"]
    is_significant = bool(p_value < alpha) and is_improvement

    return {
        "baseline_threshold": baseline_threshold,
        "baseline_metrics": baseline,
        "chosen_threshold": float(best["threshold"]),
        "chosen_metrics": best.to_dict(),
        "z_stat": z_stat,
        "p_value": p_value,
        "fp_reduction_significant_at_alpha": is_significant,
        "sweep": sweep,
    }
