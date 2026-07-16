from __future__ import annotations

import logging

from api.model_bundle import build_lgbm_row
from api.schemas import TransactionFeatures
from llm.client import get_llm_client
from llm.config import LLMConfig
from llm.fact_checker import check_explanation
from llm.fallback import rule_based_explanation
from llm.schemas import ExplanationOutput
from models.lightgbm_model import predict_shap_contributions

log = logging.getLogger(__name__)


async def explain_transaction(
    bundle: dict,
    llm_config: LLMConfig,
    transaction: dict,
    features: TransactionFeatures,
) -> tuple[ExplanationOutput, bool, bool | None]:
    """Explains one transaction the same way llm/generate_explanations.py's batch
    pipeline does: ground the prompt in the model's exact TreeSHAP contributions,
    then fall back to the deterministic rule-based template on *any* failure -
    missing Anthropic API key, Ollama not reachable, rate limits, repeated
    schema-validation failures - so a broken or unconfigured LLM backend
    degrades this endpoint gracefully instead of erroring the caller out
    (PLAN.md §06's reliability requirement). Returns (explanation, used_llm,
    fact_check_passed) - the latter is None for a fallback explanation, since
    the fact-checker only applies to LLM-generated prose.
    """
    lgbm_row = build_lgbm_row(bundle, features)
    contributions = predict_shap_contributions(bundle["lightgbm_model"], lgbm_row)[0]
    shap_values = dict(zip(bundle["lgbm_feature_columns"], contributions[:-1], strict=True))
    shap_base_value = float(contributions[-1])
    feature_dict = lgbm_row.iloc[0].to_dict()

    try:
        client = get_llm_client(llm_config)
        explanation, _usage = await client.generate_explanation(
            transaction, feature_dict, shap_values, shap_base_value
        )
    except Exception:
        log.exception(
            "LLM explanation failed for transaction %s, falling back to rule-based template",
            transaction.get("transaction_id"),
        )
        return rule_based_explanation(transaction, feature_dict), False, None

    fact_check_context = {**feature_dict, **shap_values, "shap_base_value": shap_base_value}
    fact_check = check_explanation(explanation.explanation, transaction, fact_check_context)
    if not fact_check.is_clean:
        log.warning(
            "Fact-check flagged transaction %s: mismatched numbers %s",
            transaction.get("transaction_id"),
            fact_check.mismatched_numbers,
        )
    return explanation, True, fact_check.is_clean
