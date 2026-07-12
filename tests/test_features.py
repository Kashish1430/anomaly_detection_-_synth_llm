from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_sim.config import SimConfig
from data_sim.simulate import run
from features.behavioral import compute_behavioral_features
from features.pipeline import build_feature_table
from features.round_amount import compute_round_amount_features
from features.schemas import FeatureTableSchema
from features.velocity import compute_velocity_features


def _hand_crafted_transactions() -> pd.DataFrame:
    # Customer C1: designed to exercise velocity windows, new-counterparty,
    # channel-switch, round-amount, and amount_to_avg_ratio with known answers.
    rows = [
        # customer, timestamp,                amount, direction, channel,  counterparty
        ("C1", "2024-01-01 00:00:00", 100.0, "debit", "card", "X"),
        ("C1", "2024-01-01 00:30:00", 250.0, "debit", "card", "Y"),
        ("C1", "2024-01-01 02:00:00", 100.0, "debit", "online", "X"),
        ("C1", "2024-01-02 01:00:00", 333.0, "debit", "online", "Z"),
        # customer C2: only present to confirm grouping doesn't cross customers
        ("C2", "2024-01-01 00:15:00", 999.0, "debit", "wire", "W"),
    ]
    columns = ["customer_id", "timestamp", "amount", "direction", "channel", "counterparty_id"]
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["counterparty_country"] = "GB"
    df["is_cross_border"] = False
    df["is_anomalous"] = False
    df["typology"] = None
    df.insert(0, "transaction_id", [f"TXN{i:03d}" for i in range(len(df))])
    return df


def test_velocity_windows_hand_crafted():
    tx = _hand_crafted_transactions()
    out = compute_velocity_features(tx).set_index("transaction_id")

    assert out.loc["TXN000", "velocity_count_1h"] == 1  # t0: only itself
    assert out.loc["TXN001", "velocity_count_1h"] == 2  # t1: t0 (30m ago) + itself
    assert out.loc["TXN002", "velocity_count_1h"] == 1  # t2: t0/t1 fall outside the 1h window
    assert out.loc["TXN002", "velocity_count_24h"] == 3  # t2: all of C1's txns so far

    # C2's single transaction must not appear in C1's counts and vice versa
    assert out.loc["TXN004", "velocity_count_24h"] == 1


def test_new_counterparty_and_channel_switch_hand_crafted():
    tx = _hand_crafted_transactions()
    out = compute_behavioral_features(tx).set_index("transaction_id")

    assert list(out.loc[["TXN000", "TXN001", "TXN002", "TXN003"], "is_new_counterparty"]) == [
        True,  # X, first time ever
        True,  # Y, first time ever
        False,  # X again
        True,  # Z, first time ever
    ]
    assert list(out.loc[["TXN000", "TXN001", "TXN002", "TXN003"], "is_channel_switch"]) == [
        False,  # no prior transaction
        False,  # card -> card
        True,  # card -> online
        False,  # online -> online
    ]


def test_amount_to_avg_ratio_hand_crafted():
    tx = _hand_crafted_transactions()
    out = compute_behavioral_features(tx).set_index("transaction_id")

    assert out.loc["TXN000", "amount_to_avg_ratio"] == pytest.approx(1.0)  # no prior history
    assert out.loc["TXN001", "amount_to_avg_ratio"] == pytest.approx(250 / 100)
    assert out.loc["TXN002", "amount_to_avg_ratio"] == pytest.approx(100 / ((100 + 250) / 2))
    assert out.loc["TXN003", "amount_to_avg_ratio"] == pytest.approx(333 / ((100 + 250 + 100) / 3))


def test_round_amount_flag_hand_crafted():
    tx = _hand_crafted_transactions()
    out = compute_round_amount_features(tx).set_index("transaction_id")

    assert list(out.loc[["TXN000", "TXN001", "TXN002", "TXN003"], "is_round_amount"]) == [
        True,  # 100
        False,  # 250
        True,  # 100
        False,  # 333
    ]


@pytest.fixture(scope="module")
def small_sim_data():
    config = SimConfig(seed=11, n_customers=300, target_n_transactions=8_000)
    customers, transactions, _ = run(config)
    return customers, transactions


def test_feature_table_matches_schema_and_row_count(small_sim_data):
    customers, transactions = small_sim_data
    features = build_feature_table(transactions, customers)
    FeatureTableSchema.validate(features)
    assert len(features) == len(transactions)
    assert set(features["transaction_id"]) == set(transactions["transaction_id"])


def test_prior_only_features_are_not_affected_by_future_rows(small_sim_data):
    """Leakage guard: changing a later transaction must not change the
    engineered features of an earlier transaction for the same customer."""
    customers, transactions = small_sim_data

    counts = transactions["customer_id"].value_counts()
    target_customer = counts[counts >= 4].index[0]
    cust_mask = transactions["customer_id"] == target_customer
    cust_rows = transactions[cust_mask].sort_values("timestamp")
    early_txn_id = cust_rows.iloc[0]["transaction_id"]
    late_txn_id = cust_rows.iloc[-1]["transaction_id"]

    baseline = build_feature_table(transactions, customers).set_index("transaction_id")

    mutated = transactions.copy()
    mutated.loc[mutated["transaction_id"] == late_txn_id, "amount"] = 999_999.0
    mutated_features = build_feature_table(mutated, customers).set_index("transaction_id")

    prior_only_cols = [
        "velocity_count_1h",
        "amount_to_avg_ratio",
        "personal_amount_zscore",
        "is_new_counterparty",
        "hour_zscore",
    ]
    for col in prior_only_cols:
        before = baseline.loc[early_txn_id, col]
        after = mutated_features.loc[early_txn_id, col]
        if isinstance(before, float) and np.isnan(before):
            assert np.isnan(after)
        else:
            assert before == after, f"{col} leaked future information into an earlier row"
