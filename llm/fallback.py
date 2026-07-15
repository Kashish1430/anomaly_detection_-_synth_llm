from __future__ import annotations

from llm.schemas import ExplanationOutput, Typology


def rule_based_explanation(transaction: dict, features: dict) -> ExplanationOutput:
    """Deterministic templated explanation used when every LLM call attempt has
    failed - keeps the pipeline (and any UI built on it) working rather than
    broken, per PLAN.md §06.
    """
    amount = transaction.get("amount")
    typology: Typology
    detail: str

    if features.get("is_round_amount") and features.get("round_amount_count_30d", 0) >= 3:
        typology = "round_amount"
        detail = (
            f"Transaction amount {amount} is a round figure, and this account has "
            f"{features.get('round_amount_count_30d')} other round-amount transactions "
            "in the trailing 30 days."
        )
    elif features.get("velocity_count_1h", 0) >= 5 or features.get("velocity_count_24h", 0) >= 15:
        typology = "velocity_spike"
        detail = (
            f"Unusually high transaction velocity: {features.get('velocity_count_1h')} "
            f"transactions in the last hour, {features.get('velocity_count_24h')} in 24h."
        )
    elif abs(features.get("peer_zscore", 0.0)) >= 3:
        typology = "peer_deviation"
        detail = (
            "Transaction deviates sharply from this customer's peer group "
            f"(peer z-score {features.get('peer_zscore'):.2f})."
        )
    elif features.get("is_cross_border"):
        typology = "geographic_risk"
        detail = f"Cross-border transaction to {transaction.get('counterparty_country')}."
    elif features.get("is_new_counterparty"):
        typology = "layering"
        detail = "First transaction to a new counterparty, flagged for review."
    elif abs(features.get("personal_amount_zscore", 0.0)) >= 3:
        typology = "structuring"
        detail = (
            f"Amount {amount} is a significant deviation from this customer's usual "
            f"transaction size (z-score {features.get('personal_amount_zscore'):.2f})."
        )
    else:
        typology = "none"
        detail = "Flagged by the scoring model; no single rule-based signal dominates."

    return ExplanationOutput(
        explanation=f"{detail} (Auto-generated fallback - LLM explanation unavailable.)",
        typology=typology,
        confidence=0.3,
        likely_false_positive=False,
    )
