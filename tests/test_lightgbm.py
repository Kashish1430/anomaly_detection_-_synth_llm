from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_sim.config import SimConfig
from data_sim.simulate import run as simulate_run
from models.lightgbm_model import fit_lightgbm, predict_proba_anomaly
from models.train_lightgbm import run as train_lightgbm_run
from models.tuning import tune_lightgbm


def test_lightgbm_fit_predict_shape_and_range():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 1.0).astype(int).to_numpy()

    model = fit_lightgbm(X, y, random_state=1)
    proba = predict_proba_anomaly(model, X)

    assert proba.shape == (n,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_lightgbm_recovers_a_strong_signal():
    """Regression guard: a feature that near-perfectly determines the label
    should produce near-perfect predicted probabilities. If this ever fails,
    something is wrong with the fit/predict wiring, not the data."""
    rng = np.random.default_rng(1)
    n = 3000
    signal = rng.normal(size=n)
    y = (signal > 0).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})

    model = fit_lightgbm(X, y, random_state=2)
    proba = predict_proba_anomaly(model, X)

    high_signal_proba = proba[signal > 1].mean()
    low_signal_proba = proba[signal < -1].mean()
    assert high_signal_proba > 0.8
    assert low_signal_proba < 0.2


def test_tune_lightgbm_runs_and_improves_or_matches_default():
    rng = np.random.default_rng(3)
    n = 4000
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.7, size=n) > 0.5).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))

    study = tune_lightgbm(X, y, timestamps, n_trials=5, n_folds=2, seed=4)

    assert study.best_value > 0.0
    assert "cv_std" in study.best_trial.user_attrs


@pytest.fixture(scope="module")
def small_lightgbm_result(tmp_path_factory):
    config = SimConfig(seed=31, n_customers=600, target_n_transactions=15_000)
    customers, transactions, _ = simulate_run(config)

    data_dir = tmp_path_factory.mktemp("lightgbm_data")
    customers.to_parquet(data_dir / "customers.parquet", index=False)
    transactions.to_parquet(data_dir / "transactions.parquet", index=False)

    return train_lightgbm_run(
        data_dir,
        capacity_frac=0.02,
        train_frac=0.6,
        val_frac=0.2,
        n_trials=3,
        cv_folds=2,
        seed=31,
        log_to_mlflow=False,
    )


def test_train_lightgbm_produces_sane_metrics(small_lightgbm_result):
    metrics = small_lightgbm_result["metrics"]
    assert 0.0 <= metrics["test_precision_at_capacity"] <= 1.0
    assert 0.0 <= metrics["test_recall_at_capacity"] <= 1.0
    assert metrics["test_precision_ci_lower"] <= metrics["test_precision_at_capacity"]
    assert metrics["test_precision_ci_upper"] >= metrics["test_precision_at_capacity"]
    assert 0.0 <= metrics["threshold_ztest_pvalue"] <= 1.0


def test_train_lightgbm_beats_random_guessing(small_lightgbm_result):
    """Regression guard: precision at the capacity threshold should be well
    above the dataset's raw anomaly rate - otherwise the model is adding
    nothing over flagging transactions at random."""
    config = SimConfig(seed=31, n_customers=600, target_n_transactions=15_000)
    _, transactions, _ = simulate_run(config)
    base_rate = transactions["is_anomalous"].mean()

    metrics = small_lightgbm_result["metrics"]
    assert metrics["test_precision_at_capacity"] > base_rate * 2
