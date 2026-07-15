from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Typology = Literal[
    "structuring",
    "layering",
    "round_amount",
    "velocity_spike",
    "peer_deviation",
    "geographic_risk",
    "none",
]


class ExplanationOutput(BaseModel):
    explanation: str = Field(
        description=(
            "Plain-language explanation of why this transaction was flagged, in 2-4 "
            "sentences, citing only values present in the supplied transaction and "
            "feature data."
        )
    )
    typology: Typology = Field(
        description="Best-matching typology label from the fixed taxonomy, or 'none' if none fit."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the typology label, 0-1.")
    likely_false_positive: bool = Field(
        description="True if the explanation itself suggests this flag is probably benign."
    )
