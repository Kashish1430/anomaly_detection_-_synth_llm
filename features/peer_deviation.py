from __future__ import annotations

import numpy as np
import pandas as pd

from features.utils import sort_by_customer_time

MAD_TO_STD = 1.4826  # scale factor so MAD approximates std under normality


def compute_peer_group_stats(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Median/MAD of transaction amount per peer_group (segment x home
    country) - the group-level statistics compute_peer_deviation_features
    scores each transaction against. Exposed separately so
    api/load_full_history.py can persist these once, offline, into Postgres's
    peer_group_stats table - live per-transaction scoring (api/live_features.py)
    looks them up instead of recomputing over a full peer group's rows on
    every request (see CLAUDE.md's Week 7 load_data.py incident on why that
    matters on this box).
    """
    df = transactions.merge(customers[["customer_id", "peer_group"]], on="customer_id", how="left")
    return df.groupby("peer_group")["amount"].agg(
        peer_median="median",
        peer_mad=lambda s: (s - s.median()).abs().median(),
    )


def compute_peer_deviation_features(
    transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    """Robust z-score of each transaction's amount against its customer's peer
    group (segment x home country).

    KNOWN SIMPLIFICATION: peer-group median/MAD are computed once over the
    *entire* dataset, including transactions that happen after the one being
    scored. Unlike the customer-level features in behavioral.py/contextual.py,
    this is not yet point-in-time safe. That's acceptable for the Week 2
    baseline but must be revisited in Week 3 alongside time-based CV (PLAN.md
    §07) - the fix is computing peer statistics per fold, from the training
    period only. Tracked here rather than silently ignored.
    """
    df = sort_by_customer_time(transactions).merge(
        customers[["customer_id", "peer_group"]], on="customer_id", how="left"
    )

    peer_stats = compute_peer_group_stats(transactions, customers)
    df = df.merge(peer_stats, on="peer_group", how="left")

    mad_safe = df["peer_mad"].replace(0, np.nan) * MAD_TO_STD
    peer_zscore = (df["amount"] - df["peer_median"]) / mad_safe
    peer_zscore = peer_zscore.fillna(0.0)

    return pd.DataFrame(
        {
            "transaction_id": df["transaction_id"].to_numpy(),
            "peer_zscore": peer_zscore.to_numpy(),
        }
    )
