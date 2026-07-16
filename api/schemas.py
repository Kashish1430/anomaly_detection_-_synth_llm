from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llm.schemas import Typology


class TransactionFeatures(BaseModel):
    """The 18 engineered features `features.pipeline.FEATURE_COLUMNS` produces for
    one transaction. The skeleton inference endpoint takes these pre-computed
    rather than a raw transaction, since several of them (velocity, peer
    z-score) need a customer's transaction history to compute - that lookup is
    Postgres's job once it's wired in, not this endpoint's.
    """

    velocity_count_1h: float
    velocity_sum_1h: float
    velocity_count_24h: float
    velocity_sum_24h: float
    velocity_count_7d: float
    velocity_sum_7d: float
    velocity_count_30d: float
    velocity_sum_30d: float
    amount_to_avg_ratio: float
    personal_amount_zscore: float
    is_new_counterparty: bool
    is_channel_switch: bool
    is_round_amount: bool
    round_amount_count_30d: float
    peer_zscore: float
    hour_of_day: int = Field(ge=0, le=23)
    hour_zscore: float
    is_cross_border: bool


class ScoreRequest(BaseModel):
    transaction_id: str
    features: TransactionFeatures


class ScoreResponse(BaseModel):
    transaction_id: str
    anomaly_probability: float = Field(ge=0.0, le=1.0)
    is_flagged: bool
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_run_id: str
    model_packaged_at: str


class ExplainRequest(BaseModel):
    transaction_id: str
    # Raw transaction context for the prompt (amount, channel, counterparty,
    # country, ...) - deliberately not a fixed schema since llm/prompts.py just
    # formats whatever keys are present, the same way llm/generate_explanations.py
    # passes through a flexible row of columns.
    transaction: dict[str, Any]
    features: TransactionFeatures


class ExplainResponse(BaseModel):
    transaction_id: str
    explanation: str
    typology: Typology
    confidence: float = Field(ge=0.0, le=1.0)
    likely_false_positive: bool
    source: Literal["llm", "fallback"]
    fact_check_passed: bool | None = None
