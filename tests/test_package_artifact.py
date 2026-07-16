from __future__ import annotations

import joblib
import pandas as pd

from data_sim.config import SimConfig
from data_sim.simulate import run as simulate_run
from features.pipeline import FEATURE_COLUMNS
from models.lightgbm_model import fit_lightgbm
from models.package_artifact import LGBM_FEATURE_COLUMNS, build_bundle


def _write_small_dataset(tmp_path):
    config = SimConfig(seed=11, n_customers=400, target_n_transactions=8_000)
    customers, transactions, _ = simulate_run(config)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    customers.to_parquet(data_dir / "customers.parquet", index=False)
    transactions.to_parquet(data_dir / "transactions.parquet", index=False)
    return data_dir


def test_build_bundle_contains_a_fitted_isolation_forest_and_the_loaded_lightgbm_model(
    tmp_path, monkeypatch
):
    data_dir = _write_small_dataset(tmp_path)

    # build_bundle only needs *some* LightGBM model + threshold back from
    # load_tuned_model_and_threshold - stub it out with a cheaply-fit model
    # instead of needing a real MLflow run for this test.
    rng_X = [[0.0] * len(LGBM_FEATURE_COLUMNS)] * 50
    rng_y = ([0] * 25) + ([1] * 25)
    fake_model = fit_lightgbm(pd.DataFrame(rng_X, columns=LGBM_FEATURE_COLUMNS), rng_y)
    fake_threshold = 0.5
    monkeypatch.setattr(
        "models.package_artifact.load_tuned_model_and_threshold",
        lambda run_id: (fake_model, fake_threshold),
    )

    bundle = build_bundle(data_dir, run_id="fake-run", train_frac=0.6, val_frac=0.2, seed=11)

    assert bundle["lightgbm_model"] is fake_model
    assert bundle["capacity_threshold"] == fake_threshold
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    assert bundle["lgbm_feature_columns"] == [*FEATURE_COLUMNS, "if_anomaly_score"]
    assert bundle["metadata"]["mlflow_run_id"] == "fake-run"
    assert bundle["metadata"]["seed"] == 11

    # the IsolationForest is a real fit, not a stub - it must be able to score
    isolation_forest = bundle["isolation_forest"]
    assert hasattr(isolation_forest, "estimators_")


def test_bundle_round_trips_through_joblib(tmp_path, monkeypatch):
    data_dir = _write_small_dataset(tmp_path)

    fake_model = fit_lightgbm(
        pd.DataFrame([[0.0] * len(LGBM_FEATURE_COLUMNS)] * 10, columns=LGBM_FEATURE_COLUMNS),
        [0, 1] * 5,
    )
    monkeypatch.setattr(
        "models.package_artifact.load_tuned_model_and_threshold",
        lambda run_id: (fake_model, 0.7),
    )

    bundle = build_bundle(data_dir, run_id="fake-run", train_frac=0.6, val_frac=0.2, seed=11)

    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, bundle_path)
    reloaded = joblib.load(bundle_path)

    assert reloaded["capacity_threshold"] == bundle["capacity_threshold"]
    assert reloaded["feature_columns"] == bundle["feature_columns"]
    assert hasattr(reloaded["isolation_forest"], "estimators_")
