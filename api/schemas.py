from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm.schemas import Typology


class TransactionFeatures(BaseModel):
    """The 18 engineered features `features.pipeline.FEATURE_COLUMNS` produces for
    one transaction. /score and /explain take these pre-computed rather than a
    raw transaction, since several of them (velocity, peer z-score) need a
    customer's transaction history to compute - api/load_data.py does that
    offline, in bulk, against the TEST split, rather than this endpoint
    computing it live per request.
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


Verdict = Literal["true_positive", "false_positive", "needs_review"]


class FeedbackRequest(BaseModel):
    verdict: Verdict
    note: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    transaction_id: str
    verdict: Verdict
    note: str | None
    submitted_at: datetime


class RawTransactionRequest(BaseModel):
    """A genuinely new transaction, as it would arrive in production - raw
    fields only, none of the 18 engineered features. /transactions/predict
    computes those from this customer's stored history (api/live_features.py)
    before scoring, unlike /score and /explain which take features pre-computed.

    new_customer_segment/home_country/declared_risk_rating are only used if
    customer_id doesn't already exist in `customers` - they let a genuinely
    new customer be registered on the fly rather than rejected outright, since
    a live system can't assume every customer_id it ever sees was already
    known at deploy time.
    """

    customer_id: str
    timestamp: datetime
    amount: float
    direction: str
    channel: str
    counterparty_id: str | None = None
    counterparty_country: str | None = None
    is_cross_border: bool = False
    new_customer_segment: str | None = None
    new_customer_home_country: str | None = None
    new_customer_declared_risk_rating: str | None = None


class PredictResponse(BaseModel):
    transaction_id: str
    anomaly_probability: float = Field(ge=0.0, le=1.0)
    is_flagged: bool
    explanation: str | None = None
    typology: Typology | None = None
    source: Literal["llm", "fallback"] | None = None
    fact_check_passed: bool | None = None


class TransactionResponse(BaseModel):
    """A row from Postgres's `transactions` table (infra/db/schema.sql) - the
    dashboard's browsable, flagged-transaction universe, loaded once offline by
    api/load_data.py rather than computed on the fly.
    """

    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float
    direction: str
    channel: str
    counterparty_id: str | None
    counterparty_country: str | None
    is_cross_border: bool
    features: TransactionFeatures
    anomaly_probability: float = Field(ge=0.0, le=1.0)
    is_flagged: bool
    is_anomalous: bool | None
    typology: str | None
