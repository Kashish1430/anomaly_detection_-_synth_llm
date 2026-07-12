from __future__ import annotations

import pandas as pd

from features.utils import expanding_prior_mean_std, safe_zscore, sort_by_customer_time


def compute_contextual_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = sort_by_customer_time(transactions)
    df = df.assign(hour=df["timestamp"].dt.hour)

    prior_mean, prior_std = expanding_prior_mean_std(df, "customer_id", "hour")
    # floor at 1 hour of natural spread - see the matching note in behavioral.py
    hour_zscore = safe_zscore(df["hour"], prior_mean, prior_std, min_std=1.0)

    return pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].to_numpy(),
            "hour_of_day": df["hour"].to_numpy(),
            "hour_zscore": hour_zscore,
            "is_cross_border": df["is_cross_border"].to_numpy(),
        }
    )
