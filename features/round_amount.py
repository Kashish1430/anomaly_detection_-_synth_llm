from __future__ import annotations

import pandas as pd

from features.utils import sort_by_customer_time

ROUND_STEP = 100


def compute_round_amount_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = sort_by_customer_time(transactions)
    is_round = (df["amount"] % ROUND_STEP == 0).astype(bool)

    indexed = df.assign(is_round=is_round).set_index("timestamp")
    rolling_round_count = (
        indexed.groupby("customer_id")["is_round"].rolling("30D").sum().reset_index(drop=True)
    )

    return pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].to_numpy(),
            "is_round_amount": is_round.to_numpy(),
            "round_amount_count_30d": rolling_round_count.to_numpy(),
        }
    )
