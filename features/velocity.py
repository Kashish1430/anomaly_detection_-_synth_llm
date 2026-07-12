from __future__ import annotations

import pandas as pd

from features.utils import sort_by_customer_time

WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7D", "30d": "30D"}


def compute_velocity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Rolling transaction count/sum per customer over trailing time windows.

    Windows are right-closed (t-window, t] - they include the transaction
    being scored itself, same convention as "how much has this customer done
    up to and including now", which is what a live monitoring system would ask.
    """
    cols = ["transaction_id", "customer_id", "timestamp", "amount"]
    df = sort_by_customer_time(transactions)[cols]
    indexed = df.set_index("timestamp")
    grouped = indexed.groupby("customer_id")["amount"]

    out = pd.DataFrame({"transaction_id": df["transaction_id"].to_numpy()})
    for label, window in WINDOWS.items():
        out[f"velocity_count_{label}"] = grouped.rolling(window).count().reset_index(drop=True)
        out[f"velocity_sum_{label}"] = grouped.rolling(window).sum().reset_index(drop=True)
    return out
