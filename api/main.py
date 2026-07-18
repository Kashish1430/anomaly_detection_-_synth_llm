from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from api.config import ApiConfig
from api.db import (
    get_customer,
    get_peer_group_stats,
    get_transaction,
    insert_customer,
    insert_explanation,
    insert_feedback,
    insert_scored_transaction,
    list_customer_transactions,
    list_feedback,
    list_flagged_transactions,
    open_pool,
)
from api.explain import explain_transaction
from api.live_features import compute_live_features
from api.model_bundle import load_bundle, score_features
from api.schemas import (
    ExplainRequest,
    ExplainResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictResponse,
    RawTransactionRequest,
    ScoreRequest,
    ScoreResponse,
    TransactionResponse,
)
from llm.config import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

config = ApiConfig.from_env()
llm_config = LLMConfig.from_env()
pool = open_pool(config.database_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load-and-cache the bundle at startup so the first request isn't slow -
    # a missing bundle doesn't crash the app (a health check should still be
    # able to report *why* it's unhealthy), it just leaves every request
    # failing loudly until `models.package_artifact` is run.
    try:
        load_bundle(config.model_bundle_path)
        log.info("Model bundle loaded from %s", config.model_bundle_path)
    except FileNotFoundError as exc:
        log.warning("%s", exc)

    # wait=False (the default): returns immediately and retries connecting in
    # the background, so a not-yet-reachable Postgres doesn't crash startup -
    # /transactions just fails per-request until it's up, same philosophy as
    # the model bundle above.
    await pool.open()
    log.info("Postgres pool opened for %s", config.database_url)
    yield
    await pool.close()


app = FastAPI(
    title="Anomaly Detection Engine API",
    description="Inference endpoint for the LLM-augmented transaction anomaly "
    "detection engine (PLAN.md §02).",
    version="0.1.0",
    lifespan=lifespan,
    root_path=config.root_path,
)


def _load_bundle_or_503() -> dict:
    try:
        return load_bundle(config.model_bundle_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = _load_bundle_or_503()
    return HealthResponse(
        status="ok",
        model_run_id=bundle["metadata"]["mlflow_run_id"],
        model_packaged_at=bundle["metadata"]["packaged_at"],
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    bundle = _load_bundle_or_503()
    proba, is_flagged = score_features(bundle, request.features)
    return ScoreResponse(
        transaction_id=request.transaction_id,
        anomaly_probability=proba,
        is_flagged=is_flagged,
        threshold=bundle["capacity_threshold"],
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    bundle = _load_bundle_or_503()
    explanation, used_llm, fact_check_passed = await explain_transaction(
        bundle, llm_config, request.transaction, request.features
    )
    return ExplainResponse(
        transaction_id=request.transaction_id,
        explanation=explanation.explanation,
        typology=explanation.typology,
        confidence=explanation.confidence,
        likely_false_positive=explanation.likely_false_positive,
        source="llm" if used_llm else "fallback",
        fact_check_passed=fact_check_passed,
    )


@app.get("/transactions", response_model=list[TransactionResponse])
async def transactions(limit: int = 50) -> list[TransactionResponse]:
    rows = await list_flagged_transactions(pool, limit=min(limit, 500))
    return [TransactionResponse.model_validate(row) for row in rows]


@app.get("/transactions/{transaction_id}", response_model=TransactionResponse)
async def transaction_detail(transaction_id: str) -> TransactionResponse:
    row = await get_transaction(pool, transaction_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"transaction {transaction_id!r} not found")
    return TransactionResponse.model_validate(row)


@app.post("/transactions/{transaction_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(transaction_id: str, request: FeedbackRequest) -> FeedbackResponse:
    if await get_transaction(pool, transaction_id) is None:
        raise HTTPException(status_code=404, detail=f"transaction {transaction_id!r} not found")
    row = await insert_feedback(pool, transaction_id, request.verdict, request.note)
    return FeedbackResponse.model_validate(row)


@app.get("/transactions/{transaction_id}/feedback", response_model=list[FeedbackResponse])
async def get_feedback(transaction_id: str) -> list[FeedbackResponse]:
    if await get_transaction(pool, transaction_id) is None:
        raise HTTPException(status_code=404, detail=f"transaction {transaction_id!r} not found")
    rows = await list_feedback(pool, transaction_id)
    return [FeedbackResponse.model_validate(row) for row in rows]


@app.post("/transactions/predict", response_model=PredictResponse)
async def predict_transaction(request: RawTransactionRequest) -> PredictResponse:
    """The live predict pipeline: raw transaction -> engineered features
    (api/live_features.py, computed from this customer's stored history) ->
    score (the same score_features /score uses) -> explain if flagged (the
    same explain_transaction /explain uses) -> persisted, so it shows up in
    GET /transactions and the dashboard exactly like the batch-loaded
    transactions do. Unlike /score and /explain, this takes a genuinely raw
    transaction - no pre-computed features required.
    """
    bundle = _load_bundle_or_503()

    customer = await get_customer(pool, request.customer_id)
    if customer is None:
        segment = request.new_customer_segment
        home_country = request.new_customer_home_country
        declared_risk_rating = request.new_customer_declared_risk_rating
        if segment is None or home_country is None or declared_risk_rating is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"customer {request.customer_id!r} not found - provide "
                    "new_customer_segment, new_customer_home_country, and "
                    "new_customer_declared_risk_rating to register a new customer"
                ),
            )
        peer_group = f"{segment}_{home_country}"
        customer = await insert_customer(
            pool, request.customer_id, segment, home_country, declared_risk_rating, peer_group
        )

    transaction_id = f"LIVE{uuid4().hex[:10].upper()}"
    history = await list_customer_transactions(pool, request.customer_id)
    peer_stats = await get_peer_group_stats(pool, customer["peer_group"])
    features = compute_live_features(
        history, peer_stats, request, transaction_id, request.customer_id
    )

    proba, is_flagged = score_features(bundle, features)

    explanation_output = None
    used_llm = False
    fact_check_passed = None
    if is_flagged:
        transaction_context = {
            "transaction_id": transaction_id,
            "customer_id": request.customer_id,
            "timestamp": request.timestamp.isoformat(),
            "amount": request.amount,
            "direction": request.direction,
            "channel": request.channel,
            "counterparty_id": request.counterparty_id,
            "counterparty_country": request.counterparty_country,
            "is_cross_border": request.is_cross_border,
        }
        explanation_output, used_llm, fact_check_passed = await explain_transaction(
            bundle, llm_config, transaction_context, features
        )

    await insert_scored_transaction(
        pool,
        transaction_id,
        request.customer_id,
        request.timestamp,
        request.amount,
        request.direction,
        request.channel,
        request.counterparty_id,
        request.counterparty_country,
        request.is_cross_border,
        features.model_dump(),
        proba,
        is_flagged,
    )

    if explanation_output is not None:
        await insert_explanation(
            pool,
            transaction_id,
            explanation_output.explanation,
            explanation_output.typology,
            explanation_output.confidence,
            explanation_output.likely_false_positive,
            "llm" if used_llm else "fallback",
            fact_check_passed,
        )

    return PredictResponse(
        transaction_id=transaction_id,
        anomaly_probability=proba,
        is_flagged=is_flagged,
        explanation=explanation_output.explanation if explanation_output else None,
        typology=explanation_output.typology if explanation_output else None,
        source=("llm" if used_llm else "fallback") if explanation_output else None,
        fact_check_passed=fact_check_passed,
    )
