from __future__ import annotations

import numpy as np
import pandas as pd


def sort_by_customer_time(transactions: pd.DataFrame) -> pd.DataFrame:
    return transactions.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def expanding_prior_mean_std(
    df: pd.DataFrame, group_col: str, value_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of `value_col` using only *prior* rows within each group
    (the current row's own value is excluded). `df` must already be sorted by
    time within each group.

    This is the leakage-safe building block behind every "customer's own
    baseline" feature below: at scoring time in production you would only ever
    know a customer's history up to, not including, the transaction being
    scored. A customer's first transaction has no prior history, so its stats
    are NaN by construction - callers decide how to fill that.
    """
    grouped_value = df.groupby(group_col)[value_col]
    n_prior = grouped_value.cumcount().to_numpy(dtype=float)
    cum_sum = (grouped_value.cumsum() - df[value_col]).to_numpy()

    sq = df[value_col] ** 2
    cum_sum_sq = (sq.groupby(df[group_col]).cumsum() - sq).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = cum_sum / n_prior
        var = np.clip(cum_sum_sq / n_prior - mean**2, a_min=0, a_max=None)
    std = np.sqrt(var)
    mean[n_prior == 0] = np.nan
    std[n_prior == 0] = np.nan
    return mean, std


def safe_zscore(
    value: pd.Series, mean: np.ndarray, std: np.ndarray, min_std: float | np.ndarray = 0.0
) -> np.ndarray:
    """z-score with a floor on the denominator.

    `expanding_prior_mean_std`'s variance is E[X^2] - E[X]^2, which is prone to
    catastrophic cancellation for customers whose prior values happen to be
    nearly identical: a tiny (sometimes near-zero-but-not-quite) std blows the
    z-score up to the hundreds of thousands. `min_std` floors the denominator -
    pass an absolute value, or an array scaled to each row's own `mean` so the
    floor tracks the value's natural magnitude (see behavioral.py).
    """
    std_floor = np.maximum(std, min_std)
    std_safe = np.where((std_floor == 0) | np.isnan(std_floor), np.nan, std_floor)
    z = (value.to_numpy() - mean) / std_safe
    return np.nan_to_num(z, nan=0.0)
