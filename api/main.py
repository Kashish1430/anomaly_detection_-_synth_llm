from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.config import ApiConfig
from api.explain import explain_transaction
from api.model_bundle import load_bundle, score_features
from api.schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from llm.config import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

config = ApiConfig.from_env()
llm_config = LLMConfig.from_env()


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
    yield


app = FastAPI(
    title="Anomaly Detection Engine API",
    description="Inference endpoint for the LLM-augmented transaction anomaly "
    "detection engine (PLAN.md §02).",
    version="0.1.0",
    lifespan=lifespan,
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
