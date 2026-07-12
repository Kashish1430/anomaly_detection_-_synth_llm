from __future__ import annotations

import numpy as np
import pytest

from data_sim.config import SimConfig
from data_sim.customers import generate_customers
from data_sim.schemas import CustomerSchema, TransactionSchema
from data_sim.simulate import run
from data_sim.transactions import generate_base_transactions
from data_sim.typologies import inject_structuring, inject_velocity_spike


@pytest.fixture
def small_config() -> SimConfig:
    return SimConfig(seed=7, n_customers=200, target_n_transactions=5_000)


def test_generate_customers_shape_and_schema(small_config):
    rng = np.random.default_rng(small_config.seed)
    customers = generate_customers(small_config, rng)
    assert len(customers) == small_config.n_customers
    assert customers["customer_id"].is_unique
    CustomerSchema.validate(customers)


def test_base_transactions_hit_target_within_tolerance(small_config):
    rng = np.random.default_rng(small_config.seed)
    customers = generate_customers(small_config, rng)
    base = generate_base_transactions(customers, small_config, rng)
    assert base["amount"].gt(0).all()
    assert set(base["direction"].unique()) <= {"debit", "credit"}
    # Poisson-sampled totals won't hit the target exactly, but should be close
    target = small_config.target_n_transactions
    assert abs(len(base) - target) / target < 0.1


def test_every_customer_home_country_reachable(small_config):
    rng = np.random.default_rng(small_config.seed)
    customers = generate_customers(small_config, rng)
    assert customers["home_country"].nunique() > 1


def test_structuring_injector_labels_are_consistent(small_config):
    rng = np.random.default_rng(small_config.seed)
    customers = generate_customers(small_config, rng)
    rows = inject_structuring(customers, rng, small_config, n_target=20)
    assert len(rows) > 0
    assert (rows["is_anomalous"]).all()
    assert (rows["typology"] == "structuring").all()
    assert (rows["amount"] < small_config.reporting_threshold).all()


def test_velocity_spike_injector_clusters_in_time(small_config):
    rng = np.random.default_rng(small_config.seed)
    customers = generate_customers(small_config, rng)
    rows = inject_velocity_spike(customers, rng, small_config, n_target=16)
    assert len(rows) > 0
    for _, group in rows.groupby("customer_id"):
        span_hours = (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 3600
        assert span_hours <= 72


def test_full_run_produces_valid_labeled_dataset(small_config):
    customers, transactions, manifest = run(small_config)

    TransactionSchema.validate(transactions)
    assert transactions["transaction_id"].is_unique

    # label consistency: typology set iff is_anomalous is True
    assert (transactions.loc[transactions["is_anomalous"], "typology"].notna()).all()
    assert (transactions.loc[~transactions["is_anomalous"], "typology"].isna()).all()

    assert 0 < manifest["anomaly_rate_actual"] < 0.10
    assert set(manifest["typology_counts"]) == {
        "structuring",
        "layering",
        "round_amount",
        "velocity_spike",
        "peer_deviation",
        "geographic_risk",
    }
    assert all(c > 0 for c in manifest["typology_counts"].values())


def test_run_is_reproducible_given_same_seed(small_config):
    _, tx_a, _ = run(small_config)
    _, tx_b, _ = run(small_config)
    assert tx_a["amount"].tolist() == tx_b["amount"].tolist()
    assert tx_a["customer_id"].tolist() == tx_b["customer_id"].tolist()


def test_different_seeds_produce_different_data(small_config):
    other_config = SimConfig(
        seed=small_config.seed + 1,
        n_customers=small_config.n_customers,
        target_n_transactions=small_config.target_n_transactions,
    )
    _, tx_a, _ = run(small_config)
    _, tx_b, _ = run(other_config)
    assert tx_a["amount"].tolist() != tx_b["amount"].tolist()
