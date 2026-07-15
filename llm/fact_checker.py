from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


@dataclass
class FactCheckResult:
    mismatched_numbers: list[float]
    is_clean: bool


def check_explanation(
    explanation_text: str, transaction: dict, features: dict, tolerance: float = 0.01
) -> FactCheckResult:
    """Regex-extracts every number cited in the LLM's explanation text and confirms
    each one matches some value in the source transaction/feature data (within
    `tolerance`), catching hallucinated figures before they reach an investigator
    (PLAN.md §06). Numbers embedded in feature key names (e.g. the "24" in
    `velocity_count_24h`) count as legitimate source values too, since prose like
    "24 hours" is a paraphrase of the window name, not a fabricated figure.
    """
    merged = {**transaction, **features}
    source_values = [
        float(v) for v in merged.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    source_values += [float(n) for key in merged for n in re.findall(r"\d+", key)]

    mismatches = []
    for match in _NUMBER_RE.findall(explanation_text):
        number = float(match)
        if not any(
            abs(number - source) <= tolerance * max(abs(source), 1.0) for source in source_values
        ):
            mismatches.append(number)

    return FactCheckResult(mismatched_numbers=mismatches, is_clean=not mismatches)
