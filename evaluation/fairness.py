from __future__ import annotations

import pandas as pd

from evaluation.stats import two_proportion_ztest, wilson_confidence_interval


def flagging_rate_by_group(flagged: pd.Series, group: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Statistical-parity check (PLAN.md §08): the model's flagging rate within
    each customer segment, with a Wilson confidence interval - the bias/fairness
    analogue of the precision CI in evaluation/stats.py, measuring P(flagged)
    per group instead of P(true positive | flagged) overall.
    """
    df = pd.DataFrame({"group": group.to_numpy(), "flagged": flagged.to_numpy()})
    rows = []
    for name, sub in df.groupby("group", sort=True):
        n = len(sub)
        n_flagged = int(sub["flagged"].sum())
        ci_low, ci_high = wilson_confidence_interval(n_flagged, n, alpha=alpha)
        rows.append(
            {
                "group": name,
                "n": n,
                "n_flagged": n_flagged,
                "flagging_rate": n_flagged / n if n else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def parity_tests_vs_reference(
    flagged: pd.Series, group: pd.Series, reference: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Two-proportion z-test of each group's flagging rate against a reference
    group (PLAN.md §08 bias/fairness check) - the same z-test methodology already
    used in evaluation/threshold_tuning.py to compare FP rates between candidate
    thresholds, applied here across customer segments instead.
    """
    df = pd.DataFrame({"group": group.to_numpy(), "flagged": flagged.to_numpy()})
    counts = df.groupby("group")["flagged"].agg(["sum", "count"])
    if reference not in counts.index:
        raise ValueError(f"reference group {reference!r} not present in data")
    ref_count, ref_n = int(counts.loc[reference, "sum"]), int(counts.loc[reference, "count"])

    rows = []
    for name, row in counts.iterrows():
        if name == reference:
            continue
        count, n = int(row["sum"]), int(row["count"])
        z_stat, p_value = two_proportion_ztest(count, n, ref_count, ref_n)
        rows.append(
            {
                "group": name,
                "reference": reference,
                "rate_diff": (count / n if n else 0.0) - (ref_count / ref_n if ref_n else 0.0),
                "z_stat": z_stat,
                "p_value": p_value,
                "significant": p_value < alpha,
            }
        )
    return pd.DataFrame(rows)
