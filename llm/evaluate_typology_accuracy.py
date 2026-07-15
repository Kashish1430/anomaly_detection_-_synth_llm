from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from llm.client import AnthropicClient
from llm.config import LLMConfig
from llm.generate_explanations import LGBM_FEATURE_COLUMNS, SHAP_COLUMNS, assemble_flagged_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


async def run(data_dir: Path, sample_size: int = 50, seed: int = 42) -> dict:
    """Compares Haiku's bulk-generated typology label against a Sonnet-generated
    label on the same sample, benchmarking the cheap bulk model's accuracy
    against a stronger one (PLAN.md §06). Always runs against Anthropic - this
    is a Haiku-vs-Sonnet comparison, not part of the local-swap surface.
    """
    config = LLMConfig.from_env()
    sample = assemble_flagged_sample(data_dir, sample_size=sample_size, seed=seed)

    haiku = AnthropicClient(config, model=config.anthropic_model_bulk)
    sonnet = AnthropicClient(config, model=config.anthropic_model_accuracy)

    agreements = 0
    for _, row in sample.iterrows():
        transaction = row.drop(LGBM_FEATURE_COLUMNS + SHAP_COLUMNS + ["shap_base_value"]).to_dict()
        features = row[LGBM_FEATURE_COLUMNS].to_dict()
        shap_values = {col: float(row[f"shap_{col}"]) for col in LGBM_FEATURE_COLUMNS}
        shap_base_value = float(row["shap_base_value"])
        haiku_explanation, _ = await haiku.generate_explanation(
            transaction, features, shap_values, shap_base_value
        )
        sonnet_explanation, _ = await sonnet.generate_explanation(
            transaction, features, shap_values, shap_base_value
        )
        if haiku_explanation.typology == sonnet_explanation.typology:
            agreements += 1

    accuracy = agreements / len(sample) if len(sample) else 0.0
    log.info(
        "Haiku/Sonnet typology agreement: %d/%d (%.1f%%)", agreements, len(sample), accuracy * 100
    )
    return {"n_sample": len(sample), "agreements": agreements, "agreement_rate": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Haiku's typology labels against Sonnet's on a sample."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = asyncio.run(run(Path(args.data_dir), sample_size=args.sample_size, seed=args.seed))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
