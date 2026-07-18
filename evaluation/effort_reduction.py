from __future__ import annotations

import numpy as np

from models.evaluate import evaluate_at_capacity


def capacity_for_target_recall(
    y_true: np.ndarray, anomaly_score: np.ndarray, target_recall: float
) -> float:
    """Smallest review-capacity fraction (share of transactions flagged,
    highest score first) needed to reach at least `target_recall`. Recall is
    a non-decreasing step function of how many top-scored rows are flagged,
    so this is exact cumulative counting, not a threshold grid search.
    """
    y_true = np.asarray(y_true, dtype=bool)
    n = len(y_true)
    n_positive = int(y_true.sum())
    if n_positive == 0:
        return 0.0

    order = np.argsort(anomaly_score)[::-1]
    cum_true_positives = np.cumsum(y_true[order])
    recall_at_k = cum_true_positives / n_positive

    reaches_target = np.flatnonzero(recall_at_k >= target_recall)
    if len(reaches_target) == 0:
        return 1.0  # even reviewing everything doesn't reach target_recall
    k = int(reaches_target[0]) + 1
    return k / n


def effort_reduction_summary(
    y_true: np.ndarray,
    baseline_score: np.ndarray,
    tuned_score: np.ndarray,
    capacity_frac: float = 0.02,
) -> dict:
    """Compares the tuned model against the baseline on the *same* TEST rows,
    at the same fixed review capacity, from two angles:

    1. False positives avoided at fixed capacity (same review volume for
       both - PLAN.md's original CV-draft "cut false positives" bullet).
    2. Review volume ("effort") needed by the tuned model to match the
       baseline's recall at that capacity - the actual "reduced manual
       review effort" bullet, since #1 holds volume fixed rather than
       measuring a volume reduction directly.
    """
    baseline_metrics = evaluate_at_capacity(baseline_score, y_true, capacity_frac)
    tuned_metrics = evaluate_at_capacity(tuned_score, y_true, capacity_frac)

    baseline_fp_share = 1 - baseline_metrics["precision"]
    tuned_fp_share = 1 - tuned_metrics["precision"]
    fp_reduction_at_fixed_capacity = (
        1 - (tuned_fp_share / baseline_fp_share) if baseline_fp_share > 0 else 0.0
    )

    equivalent_capacity = capacity_for_target_recall(
        y_true, tuned_score, target_recall=baseline_metrics["recall"]
    )
    effort_reduction = 1 - (equivalent_capacity / capacity_frac)

    return {
        "capacity_frac": capacity_frac,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "fp_reduction_at_fixed_capacity": fp_reduction_at_fixed_capacity,
        "equivalent_capacity_frac_for_tuned_model": equivalent_capacity,
        "manual_review_effort_reduction": effort_reduction,
    }
