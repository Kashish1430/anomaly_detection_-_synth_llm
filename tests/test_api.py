from __future__ import annotations

from datetime import UTC, datetime

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.db import open_pool
from data_sim.config import SimConfig
from data_sim.simulate import run as simulate_run
from llm.costs import TokenUsage
from llm.schemas import ExplanationOutput
from models.lightgbm_model import fit_lightgbm
from models.package_artifact import LGBM_FEATURE_COLUMNS, build_bundle


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    # api.main's `pool` is a process-lifetime singleton in production (opened
    # once, closed once) - but psycopg_pool pools can't reopen after closing,
    # and every test's `with TestClient(...)` cycles the lifespan (open+close)
    # against whatever pool object api.main currently holds. Swap in a fresh,
    # never-yet-opened pool per test so tests can run in any order/count.
    import api.main as api_main

    monkeypatch.setattr(api_main, "pool", open_pool(api_main.config.database_url))


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


class _FakeClient:
    """Mirrors tests/test_llm.py's _FakeClient test double - stands in for
    AnthropicClient/OpenAICompatibleClient without any real network call."""

    def __init__(self, explanation=None, raises: bool = False) -> None:
        self.model_name = "fake-model"
        self._explanation = explanation
        self._raises = raises

    async def generate_explanation(self, transaction, features, shap_values, shap_base_value):
        if self._raises:
            raise RuntimeError("simulated LLM failure (e.g. missing API key / Ollama down)")
        return self._explanation, TokenUsage(input_tokens=10, output_tokens=10)


SAMPLE_TRANSACTION = {
    "amount": 9000.0,
    "direction": "debit",
    "channel": "online",
    "counterparty_country": "PA",
}


def test_explain_falls_back_when_llm_call_fails(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)
    monkeypatch.setattr("api.explain.get_llm_client", lambda config: _FakeClient(raises=True))

    with TestClient(api_main.app) as client:
        response = client.post(
            "/explain",
            json={
                "transaction_id": "txn-1",
                "transaction": SAMPLE_TRANSACTION,
                "features": SAMPLE_FEATURES,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback"
    assert "fallback" in body["explanation"].lower()
    assert body["fact_check_passed"] is None


SAMPLE_DB_ROW = {
    "transaction_id": "txn-db-1",
    "customer_id": "cust-1",
    "timestamp": datetime(2025, 1, 1, tzinfo=UTC),
    "amount": 500.0,
    "direction": "debit",
    "channel": "online",
    "counterparty_id": "cp-1",
    "counterparty_country": "GB",
    "is_cross_border": False,
    "features": SAMPLE_FEATURES,
    "anomaly_probability": 0.91,
    "is_flagged": True,
    "is_anomalous": True,
    "typology": "layering",
}


async def _fake_list_flagged_transactions(pool, limit=50):
    return [SAMPLE_DB_ROW]


async def _fake_get_transaction(pool, transaction_id):
    return SAMPLE_DB_ROW if transaction_id == "txn-db-1" else None


def test_list_transactions_endpoint(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)
    monkeypatch.setattr(api_main, "list_flagged_transactions", _fake_list_flagged_transactions)

    with TestClient(api_main.app) as client:
        response = client.get("/transactions?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["transaction_id"] == "txn-db-1"
    assert body[0]["features"]["is_new_counterparty"] is False


def test_transaction_detail_endpoint_found(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)
    monkeypatch.setattr(api_main, "get_transaction", _fake_get_transaction)

    with TestClient(api_main.app) as client:
        response = client.get("/transactions/txn-db-1")

    assert response.status_code == 200
    assert response.json()["transaction_id"] == "txn-db-1"


async def _fake_insert_feedback(pool, transaction_id, verdict, note):
    return {
        "id": 1,
        "transaction_id": transaction_id,
        "verdict": verdict,
        "note": note,
        "submitted_at": datetime(2025, 1, 2, tzinfo=UTC),
    }


def test_submit_feedback_endpoint(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)
    monkeypatch.setattr(api_main, "get_transaction", _fake_get_transaction)
    monkeypatch.setattr(api_main, "insert_feedback", _fake_insert_feedback)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/transactions/txn-db-1/feedback",
            json={"verdict": "false_positive", "note": "customer confirmed, benign"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "txn-db-1"
    assert body["verdict"] == "false_positive"
    assert body["note"] == "customer confirmed, benign"


def test_submit_feedback_404_for_unknown_transaction(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)
    monkeypatch.setattr(api_main, "get_transaction", _fake_get_transaction)
    monkeypatch.setattr(api_main, "insert_feedback", _fake_insert_feedback)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/transactions/does-not-exist/feedback",
            json={"verdict": "true_positive"},
        )

    assert response.status_code == 404


def test_transaction_detail_endpoint_404(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)
    monkeypatch.setattr(api_main, "get_transaction", _fake_get_transaction)

    with TestClient(api_main.app) as client:
        response = client.get("/transactions/does-not-exist")

    assert response.status_code == 404


def test_explain_returns_llm_explanation_when_call_succeeds(tmp_path, monkeypatch):
    bundle_path = _write_bundle(tmp_path, monkeypatch)

    import api.main as api_main

    fake_explanation = ExplanationOutput(
        explanation="This transaction of 9000.0 is a first payment to a new counterparty.",
        typology="layering",
        confidence=0.8,
        likely_false_positive=False,
    )
    api_main.load_bundle.cache_clear()
    api_main.config.model_bundle_path = str(bundle_path)
    monkeypatch.setattr(
        "api.explain.get_llm_client",
        lambda config: _FakeClient(explanation=fake_explanation),
    )

    with TestClient(api_main.app) as client:
        response = client.post(
            "/explain",
            json={
                "transaction_id": "txn-1",
                "transaction": SAMPLE_TRANSACTION,
                "features": SAMPLE_FEATURES,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["typology"] == "layering"
    assert body["fact_check_passed"] is True
