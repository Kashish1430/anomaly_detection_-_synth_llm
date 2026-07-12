from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def evaluate_at_capacity(
    anomaly_score: np.ndarray, y_true: np.ndarray, capacity_frac: float
) -> dict:
    """Simulates a fixed investigator review capacity: flag the top
    `capacity_frac` share of transactions by anomaly score, and score that
    against ground truth. This is the naive, untuned fixed-threshold baseline
    from PLAN.md §05 - real threshold tuning (two-proportion z-tests across
    candidate cut-points, PLAN.md §07) is Week 3, not this.
    """
    n = len(anomaly_score)
    k = max(1, int(round(n * capacity_frac)))
    order = np.argsort(anomaly_score)[::-1]
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True

    return {
        "capacity_frac": capacity_frac,
        "n_scored": n,
        "n_flagged": int(flagged.sum()),
        "n_true_anomalies": int(y_true.sum()),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, anomaly_score)),
    }
