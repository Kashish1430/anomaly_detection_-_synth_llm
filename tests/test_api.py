from __future__ import annotations

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from data_sim.config import SimConfig
from data_sim.simulate import run as simulate_run
from models.lightgbm_model import fit_lightgbm
from models.package_artifact import LGBM_FEATURE_COLUMNS, build_bundle

SAMPLE_FEATURES = {
    "velocity_count_1h": 1.0,
    "velocity_sum_1h": 100.0,
    "velocity_count_24h": 3.0,
    "velocity_sum_24h": 500.0,
    "velocity_count_7d": 10.0,
    "velocity_sum_7d": 2000.0,
    "velocity_count_30d": 40.0,
    "velocity_sum_30d": 8000.0,
    "amount_to_avg_ratio": 1.2,
    "personal_amount_zscore": 0.5,
    "is_new_counterparty": False,
    "is_channel_switch": False,
    "is_round_amount": False,
    "round_amount_count_30d": 0.0,
    "peer_zscore": 0.3,
    "hour_of_day": 14,
    "hour_zscore": 0.1,
    "is_cross_border": False,
}


def _write_bundle(tmp_path, monkeypatch):
    config = SimConfig(seed=21, n_customers=300, target_n_transactions=6_000)
    customers, transactions, _ = simulate_run(config)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    customers.to_parquet(data_dir / "customers.parquet", index=False)
    transactions.to_parquet(data_dir / "transactions.parquet", index=False)

    fake_model = fit_lightgbm(
        pd.DataFrame([[0.0] * len(LGBM_FEATURE_COLUMNS)] * 10, columns=LGBM_FEATURE_COLUMNS),
        [0, 1] * 5,
    )
    monkeypatch.setattr(
        "models.package_artifact.load_tuned_model_and_threshold",
        lambda run_id: (fake_model, 0.5),
    )

    bundle = build_bundle(data_dir, run_id="fake-run", train_frac=0.6, val_frac=0.2, seed=21)
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, bundle_path)
    return bundle_path


def test_health_and_score_endpoints(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)

    with TestClient(api_main.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model_run_id": "fake-run",
            "model_packaged_at": health.json()["model_packaged_at"],
        }

        response = client.post(
            "/score", json={"transaction_id": "txn-1", "features": SAMPLE_FEATURES}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["transaction_id"] == "txn-1"
        assert 0.0 <= body["anomaly_probability"] <= 1.0
        assert isinstance(body["is_flagged"], bool)
        assert body["threshold"] == 0.5


def test_score_rejects_out_of_range_hour(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)

    bad_features = {**SAMPLE_FEATURES, "hour_of_day": 25}
    with TestClient(api_main.app) as client:
        response = client.post("/score", json={"transaction_id": "txn-1", "features": bad_features})
        assert response.status_code == 422


def test_health_returns_503_when_bundle_missing(tmp_path):
    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(tmp_path / "does_not_exist.joblib")

    with TestClient(api_main.app) as client:
        response = client.get("/health")
        assert response.status_code == 503
