from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_sim.config import SimConfig
from models.baseline import fit_isolation_forest, score_anomaly
from models.evaluate import evaluate_at_capacity
from models.train_baseline import run


def test_evaluate_at_capacity_hand_crafted():
    # 10 items, 2 true anomalies at the highest scores - a perfect model
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05])
    y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])

    result = evaluate_at_capacity(scores, y_true, capacity_frac=0.2)  # flags top 2

    assert result["n_flagged"] == 2
    assert result["n_true_anomalies"] == 2
    assert result["precision"] == pytest.approx(1.0)  # both flagged are true anomalies
    assert result["recall"] == pytest.approx(1.0)


def test_evaluate_at_capacity_imperfect_model():
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])  # top 2 by score are index 0, 1
    y_true = np.array([0, 0, 0, 0, 1])  # but the one true anomaly is index 4

    result = evaluate_at_capacity(scores, y_true, capacity_frac=0.4)  # flags top 2

    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)


def test_isolation_forest_scores_have_expected_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=500),
            "b": rng.normal(size=500),
            "flag": rng.integers(0, 2, size=500).astype(bool),
        }
    )
    model = fit_isolation_forest(X, n_estimators=50, random_state=1)
    scores = score_anomaly(model, X)

    assert scores.shape == (500,)
    assert np.isfinite(scores).all()


@pytest.fixture(scope="module")
def small_run_result(tmp_path_factory):
    from data_sim.simulate import run as simulate_run

    config = SimConfig(seed=21, n_customers=400, target_n_transactions=10_000)
    customers, transactions, _ = simulate_run(config)

    data_dir = tmp_path_factory.mktemp("baseline_data")
    customers.to_parquet(data_dir / "customers.parquet", index=False)
    transactions.to_parquet(data_dir / "transactions.parquet", index=False)

    return run(
        data_dir, capacity_frac=0.02, test_size=0.3, n_estimators=100, seed=21, log_to_mlflow=False
    )


def test_baseline_run_produces_sane_metrics(small_run_result):
    metrics = small_run_result["metrics"]
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["n_flagged"] > 0


def test_baseline_beats_random_guessing(small_run_result):
    """Regression guard: if the features + IsolationForest carry no signal,
    precision at a fixed capacity should equal roughly the base anomaly rate.
    This asserts the pipeline is doing meaningfully better than that floor."""
    metrics = small_run_result["metrics"]
    base_rate = metrics["n_true_anomalies"] / metrics["n_scored"]
    assert metrics["precision"] > base_rate * 2
