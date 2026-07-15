from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from pydantic import ValidationError

from llm.cache import cache_key, get_cached, set_cached
from llm.client import AnthropicClient, LLMClient, OpenAICompatibleClient, get_llm_client
from llm.config import LLMConfig
from llm.costs import TokenUsage
from llm.fact_checker import check_explanation
from llm.fallback import rule_based_explanation
from llm.generate_explanations import (
    CUSTOMER_COLUMNS,
    LGBM_FEATURE_COLUMNS,
    SHAP_COLUMNS,
    TRANSACTION_COLUMNS,
    _explain_one,
)
from llm.schemas import ExplanationOutput


def test_get_llm_client_returns_anthropic_client():
    client = get_llm_client(LLMConfig(provider="anthropic", anthropic_api_key="test-key"))
    assert isinstance(client, AnthropicClient)


def test_get_llm_client_returns_ollama_client():
    client = get_llm_client(LLMConfig(provider="ollama"))
    assert isinstance(client, OpenAICompatibleClient)


def test_get_llm_client_rejects_unknown_provider():
    config = LLMConfig()
    config.provider = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        get_llm_client(config)


def test_explanation_output_rejects_invalid_typology():
    with pytest.raises(ValidationError):
        ExplanationOutput(
            explanation="test",
            typology="not_a_real_typology",
            confidence=0.5,
            likely_false_positive=False,
        )


def test_explanation_output_accepts_valid_typology():
    output = ExplanationOutput(
        explanation="test", typology="velocity_spike", confidence=0.9, likely_false_positive=False
    )
    assert output.typology == "velocity_spike"


def test_check_explanation_flags_hallucinated_number():
    transaction = {"amount": 500.0}
    features = {"velocity_count_1h": 2}
    result = check_explanation(
        "This transaction of 500.0 is unusual, having occurred 9999 times before.",
        transaction,
        features,
    )
    assert not result.is_clean
    assert 9999.0 in result.mismatched_numbers


def test_check_explanation_passes_clean_explanation():
    transaction = {"amount": 500.0}
    features = {"velocity_count_1h": 2}
    result = check_explanation(
        "This transaction of 500.0 occurred after 2 prior transactions in the last hour.",
        transaction,
        features,
    )
    assert result.is_clean


def test_cache_round_trip(tmp_path):
    cache_path = str(tmp_path / "cache.sqlite3")
    key = cache_key("txn_1", "AnthropicClient", "claude-haiku-4-5")
    assert get_cached(cache_path, key) is None

    explanation = ExplanationOutput(
        explanation="test", typology="none", confidence=0.5, likely_false_positive=True
    )
    set_cached(cache_path, key, explanation)

    assert get_cached(cache_path, key) == explanation


def _feature_defaults() -> dict:
    return {
        "velocity_count_1h": 0,
        "velocity_count_24h": 0,
        "is_round_amount": False,
        "round_amount_count_30d": 0,
        "peer_zscore": 0.0,
        "is_cross_border": False,
        "is_new_counterparty": False,
        "personal_amount_zscore": 0.0,
    }


def test_rule_based_explanation_detects_velocity_spike():
    transaction = {"amount": 100.0}
    features = {**_feature_defaults(), "velocity_count_1h": 6}
    result = rule_based_explanation(transaction, features)
    assert result.typology == "velocity_spike"


def test_rule_based_explanation_defaults_to_none():
    transaction = {"amount": 100.0}
    result = rule_based_explanation(transaction, _feature_defaults())
    assert result.typology == "none"


class _FakeClient(LLMClient):
    def __init__(self, explanation: ExplanationOutput | None = None, raises: bool = False) -> None:
        self.model_name = "fake-model"
        self._explanation = explanation
        self._raises = raises
        self.call_count = 0

    async def generate_explanation(
        self,
        transaction: dict,
        features: dict,
        shap_values: dict[str, float],
        shap_base_value: float,
    ) -> tuple[ExplanationOutput, TokenUsage]:
        self.call_count += 1
        if self._raises:
            raise RuntimeError("simulated failure")
        assert self._explanation is not None
        return self._explanation, TokenUsage(input_tokens=100, output_tokens=50)


def _sample_row(transaction_id: str = "txn_1") -> pd.Series:
    data: dict = {
        "transaction_id": transaction_id,
        "customer_id": "cust_1",
        "timestamp": pd.Timestamp("2025-01-01"),
        "amount": 500.0,
        "direction": "debit",
        "channel": "online",
        "counterparty_id": "cp_1",
        "counterparty_country": "GB",
        "is_cross_border": False,
        "segment": "retail",
        "home_country": "GB",
        "declared_risk_rating": "low",
        "peer_group": "retail_GB",
    }
    for col in LGBM_FEATURE_COLUMNS:
        data.setdefault(col, 0)
    for shap_col in SHAP_COLUMNS:
        data.setdefault(shap_col, 0.0)
    data.setdefault("shap_base_value", 0.0)
    required = TRANSACTION_COLUMNS + CUSTOMER_COLUMNS[1:] + LGBM_FEATURE_COLUMNS + SHAP_COLUMNS
    assert set(required) <= set(data)
    return pd.Series(data)


def test_explain_one_uses_cache_hit_without_calling_client(tmp_path):
    cache_path = str(tmp_path / "cache.sqlite3")
    row = _sample_row()
    client = _FakeClient()
    key = cache_key(row["transaction_id"], client.__class__.__name__, client.model_name)
    cached_explanation = ExplanationOutput(
        explanation="cached", typology="none", confidence=0.9, likely_false_positive=False
    )
    set_cached(cache_path, key, cached_explanation)

    semaphore = asyncio.Semaphore(1)
    result = asyncio.run(_explain_one(client, semaphore, row, cache_path))

    assert client.call_count == 0
    assert result["explanation"] == cached_explanation


def test_explain_one_falls_back_on_repeated_llm_failure(tmp_path):
    cache_path = str(tmp_path / "cache.sqlite3")
    row = _sample_row(transaction_id="txn_2")
    client = _FakeClient(raises=True)

    semaphore = asyncio.Semaphore(1)
    result = asyncio.run(_explain_one(client, semaphore, row, cache_path))

    assert result["cost_usd"] == 0.0
    assert result["explanation"].typology == "none"
    assert "fallback" in result["explanation"].explanation.lower()


def test_explain_one_caches_successful_result(tmp_path):
    cache_path = str(tmp_path / "cache.sqlite3")
    row = _sample_row(transaction_id="txn_3")
    explanation = ExplanationOutput(
        explanation="This transaction of 500.0 is unusual.",
        typology="none",
        confidence=0.8,
        likely_false_positive=False,
    )
    client = _FakeClient(explanation=explanation)

    semaphore = asyncio.Semaphore(1)
    result = asyncio.run(_explain_one(client, semaphore, row, cache_path))

    assert result["explanation"] == explanation
    key = cache_key(row["transaction_id"], client.__class__.__name__, client.model_name)
    assert get_cached(cache_path, key) == explanation
