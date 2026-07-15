from __future__ import annotations

import numpy as np
import pandas as pd

# Conventional PSI thresholds (PLAN.md §08): below this, no meaningful shift.
PSI_STABLE_THRESHOLD = 0.1
# Above this, a significant shift - the standard retraining/recalibration trigger.
PSI_SIGNIFICANT_THRESHOLD = 0.25


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index comparing a baseline distribution ("expected",
    typically the TRAIN-time population) to a later one ("actual") - the
    standard model-monitoring metric for detecting drift that would erode a
    model's validity after deployment. Bin edges are quantiles of `expected`
    so each baseline bin starts with ~equal mass; PSI then measures how much
    `actual` has shifted away from that baseline binning.

    Conventional thresholds: PSI < 0.1 no meaningful shift, 0.1-0.25 moderate
    shift worth investigating, > 0.25 significant shift.
    """
    # boolean feature columns (e.g. is_round_amount) arrive as bool dtype,
    # which numpy can't take quantiles/differences of - treat as 0/1 floats
    expected = expected.astype(float)
    actual = actual.astype(float)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf  # catch actual values outside the expected range

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # avoid log(0) / division by 0 for empty bins - a small floor is the
    # standard fix, since a truly-empty bin still carries real information
    # (a total absence of a value range that used to occur)
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _stability_status(psi: float) -> str:
    if psi < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate_shift"
    return "significant_shift"


def psi_report(
    baseline: pd.DataFrame, current: pd.DataFrame, columns: list[str], bins: int = 10
) -> pd.DataFrame:
    """PSI for each of `columns`, comparing `baseline` to `current` (PLAN.md
    §08 monitoring plan) - one row per column with a stability_status label
    using the conventional PSI thresholds, sorted by severity.
    """
    rows = []
    for col in columns:
        psi = population_stability_index(
            baseline[col].to_numpy(), current[col].to_numpy(), bins=bins
        )
        rows.append({"feature": col, "psi": psi, "stability_status": _stability_status(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
