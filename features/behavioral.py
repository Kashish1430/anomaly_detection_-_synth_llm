from __future__ import annotations

import numpy as np
import pandas as pd

from features.utils import expanding_prior_mean_std, safe_zscore, sort_by_customer_time


def compute_behavioral_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = sort_by_customer_time(transactions)

    prior_mean, prior_std = expanding_prior_mean_std(df, "customer_id", "amount")
    amount_to_avg_ratio = (df["amount"].to_numpy() / prior_mean).astype(float)
    # no prior average on a first transaction - 1.0 (neither high nor low) is the neutral fill
    amount_to_avg_ratio[pd.isna(prior_mean)] = 1.0
    # floor the z-score denominator at 5% of the customer's own prior mean (min 1.0) -
    # otherwise a customer whose first few amounts happen to be nearly identical gets a
    # near-zero std and the z-score explodes into the hundreds of thousands (see ADR note
    # in features/utils.py:safe_zscore)
    amount_std_floor = np.maximum(np.abs(np.nan_to_num(prior_mean, nan=0.0)) * 0.05, 1.0)
    personal_amount_zscore = safe_zscore(df["amount"], prior_mean, prior_std, amount_std_floor)

    is_new_counterparty = ~df.duplicated(subset=["customer_id", "counterparty_id"], keep="first")

    prev_channel = df.groupby("customer_id")["channel"].shift(1)
    is_channel_switch = (df["channel"] != prev_channel) & prev_channel.notna()

    return pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].to_numpy(),
            "amount_to_avg_ratio": amount_to_avg_ratio,
            "personal_amount_zscore": personal_amount_zscore,
            "is_new_counterparty": is_new_counterparty.to_numpy(),
            "is_channel_switch": is_channel_switch.to_numpy(),
        }
    )
