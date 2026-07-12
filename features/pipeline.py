from __future__ import annotations

import pandas as pd

from features.behavioral import compute_behavioral_features
from features.contextual import compute_contextual_features
from features.peer_deviation import compute_peer_deviation_features
from features.round_amount import compute_round_amount_features
from features.velocity import compute_velocity_features

FEATURE_COLUMNS = [
    "velocity_count_1h",
    "velocity_sum_1h",
    "velocity_count_24h",
    "velocity_sum_24h",
    "velocity_count_7d",
    "velocity_sum_7d",
    "velocity_count_30d",
    "velocity_sum_30d",
    "amount_to_avg_ratio",
    "personal_amount_zscore",
    "is_new_counterparty",
    "is_channel_switch",
    "is_round_amount",
    "round_amount_count_30d",
    "peer_zscore",
    "hour_of_day",
    "hour_zscore",
    "is_cross_border",
]


def build_feature_table(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Builds the full engineered feature table, one row per transaction_id.

    Ground truth (`is_anomalous`, `typology`) is deliberately NOT included here
    - it's joined back separately, only for evaluation, so a bug here can't
    accidentally hand the label to a model as a feature. See PLAN.md §03 and
    the leakage check in notebooks/01_eda.ipynb.
    """
    velocity = compute_velocity_features(transactions)
    behavioral = compute_behavioral_features(transactions)
    round_amount = compute_round_amount_features(transactions)
    peer = compute_peer_deviation_features(transactions, customers)
    contextual = compute_contextual_features(transactions)

    features = velocity
    for other in (behavioral, round_amount, peer, contextual):
        features = features.merge(other, on="transaction_id", how="inner", validate="one_to_one")

    if len(features) != len(transactions):
        raise ValueError("feature table row count drifted from transactions during merge")

    return features[["transaction_id", *FEATURE_COLUMNS]]
