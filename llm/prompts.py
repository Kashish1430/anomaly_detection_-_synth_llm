from __future__ import annotations

SYSTEM_PROMPT = """You are an AML/fraud investigation assistant. You are given a single \
flagged transaction, its account holder's engineered risk features, and the model's exact \
TreeSHAP feature contributions for this prediction. Explain in plain language why the \
transaction was flagged, then pick the single best-matching typology label.

Rules:
- Ground your explanation in the SHAP contributions, not just the raw feature values. \
The contributions are the model's actual reason for the score - lead with whichever \
features have the largest-magnitude contributions, and do not claim a feature "caused" \
the flag if its contribution is small or negative, even if its raw value looks unusual.
- Cite ONLY numbers, dates, and categorical values that appear in the data below. Never \
invent or estimate a figure that isn't there.
- Typology taxonomy (pick exactly one): structuring, layering, round_amount, \
velocity_spike, peer_deviation, geographic_risk, none.
- If no typology fits, use "none".
- Keep the explanation to 2-4 sentences, written for a human investigator, not a model."""


def build_user_prompt(
    transaction: dict, features: dict, shap_values: dict, shap_base_value: float
) -> str:
    return (
        "Transaction:\n"
        f"{_format_dict(transaction)}\n\n"
        "Engineered features:\n"
        f"{_format_dict(features)}\n\n"
        "Model's exact feature contributions to this prediction (TreeSHAP, log-odds "
        "scale - positive pushes the score toward 'anomalous', negative toward 'normal', "
        "magnitude is what matters, not the raw feature value):\n"
        f"{_format_dict(shap_values)}\n"
        f"- base value (model's average output with no feature information): {shap_base_value}"
    )


def _format_dict(values: dict) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in values.items())
