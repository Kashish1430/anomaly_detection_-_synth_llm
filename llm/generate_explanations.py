from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.splits import time_ordered_split
from features.pipeline import FEATURE_COLUMNS, build_feature_table
from llm.cache import cache_key, get_cached, set_cached
from llm.client import LLMClient, get_llm_client
from llm.config import LLMConfig
from llm.costs import estimate_cost_usd
from llm.fact_checker import check_explanation
from llm.fallback import rule_based_explanation
from models.baseline import fit_isolation_forest, score_anomaly
from models.lightgbm_model import fit_lightgbm, predict_proba_anomaly, predict_shap_contributions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRANSACTION_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "direction",
    "channel",
    "counterparty_id",
    "counterparty_country",
    "is_cross_border",
]
CUSTOMER_COLUMNS = ["customer_id", "segment", "home_country", "declared_risk_rating", "peer_group"]
LGBM_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "if_anomaly_score"]
SHAP_COLUMNS = [f"shap_{col}" for col in LGBM_FEATURE_COLUMNS]


def _load_features(
    data_dir: Path, transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    features_path = data_dir / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return build_feature_table(transactions, customers)


def assemble_flagged_sample(
    data_dir: Path,
    sample_size: int = 25,
    capacity_frac: float = 0.02,
    train_frac: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Scores the held-out portion of the dataset with a plain LightGBM model (no
    Optuna tuning - this assembles a demo sample for the explanation pipeline, not
    a research result) and returns up to `sample_size` flagged transactions with
    their raw fields and engineered features, for the LLM to explain.
    """
    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)

    # `is_cross_border` is a raw transaction field AND a passthrough engineered
    # feature (features/contextual.py) - drop the feature-table copy so the merge
    # doesn't collide and suffix both into `is_cross_border_x`/`_y`.
    features = features.drop(columns=["is_cross_border"])
    merged = transactions.merge(features, on="transaction_id", validate="one_to_one")
    merged = merged.merge(customers[CUSTOMER_COLUMNS], on="customer_id", validate="many_to_one")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    y_all = merged["is_anomalous"].astype(int).to_numpy()

    train_idx, _, test_idx = time_ordered_split(
        merged["timestamp"], train_frac=train_frac, val_frac=0.0
    )

    if_model = fit_isolation_forest(
        merged.iloc[train_idx][FEATURE_COLUMNS], n_estimators=200, random_state=seed
    )
    merged["if_anomaly_score"] = score_anomaly(if_model, merged[FEATURE_COLUMNS])

    model = fit_lightgbm(
        merged.iloc[train_idx][LGBM_FEATURE_COLUMNS], y_all[train_idx], random_state=seed
    )
    proba_test = predict_proba_anomaly(model, merged.iloc[test_idx][LGBM_FEATURE_COLUMNS])

    threshold = np.quantile(proba_test, 1 - capacity_frac)
    flagged_positions = test_idx[proba_test >= threshold]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        flagged_positions, size=min(sample_size, len(flagged_positions)), replace=False
    )

    # TreeSHAP contributions for the model's actual decision, not just the raw
    # feature values - grounds the LLM's explanation in why the model flagged the
    # transaction rather than having it guess from a wall of numbers.
    shap_test = predict_shap_contributions(model, merged.iloc[test_idx][LGBM_FEATURE_COLUMNS])
    shap_columns_with_base = [*SHAP_COLUMNS, "shap_base_value"]
    merged.loc[merged.index[test_idx], shap_columns_with_base] = shap_test

    columns = [
        *TRANSACTION_COLUMNS,
        *CUSTOMER_COLUMNS[1:],
        *LGBM_FEATURE_COLUMNS,
        *shap_columns_with_base,
    ]
    return merged.iloc[chosen][columns].reset_index(drop=True)


async def _explain_one(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    row: pd.Series,
    cache_path: str,
) -> dict:
    transaction = row[TRANSACTION_COLUMNS + CUSTOMER_COLUMNS[1:]].to_dict()
    features = row[LGBM_FEATURE_COLUMNS].to_dict()
    shap_values = {col: float(row[f"shap_{col}"]) for col in LGBM_FEATURE_COLUMNS}
    shap_base_value = float(row["shap_base_value"])
    key = cache_key(
        str(transaction["transaction_id"]), client.__class__.__name__, client.model_name
    )

    cached = get_cached(cache_path, key)
    if cached is not None:
        return {
            "transaction_id": transaction["transaction_id"],
            "explanation": cached,
            "cost_usd": 0.0,
        }

    async with semaphore:
        try:
            explanation, usage = await client.generate_explanation(
                transaction, features, shap_values, shap_base_value
            )
            fact_check_context = {**features, **shap_values, "shap_base_value": shap_base_value}
            fact_check = check_explanation(explanation.explanation, transaction, fact_check_context)
            if not fact_check.is_clean:
                log.warning(
                    "Fact-check flagged transaction %s: mismatched numbers %s",
                    transaction["transaction_id"],
                    fact_check.mismatched_numbers,
                )
            cost = estimate_cost_usd(client.model_name, usage.input_tokens, usage.output_tokens)
        except Exception:
            log.exception(
                "LLM call failed for transaction %s after retries, using rule-based fallback",
                transaction["transaction_id"],
            )
            explanation = rule_based_explanation(transaction, features)
            cost = 0.0

    set_cached(cache_path, key, explanation)
    return {
        "transaction_id": transaction["transaction_id"],
        "explanation": explanation,
        "cost_usd": cost,
    }


async def run(
    data_dir: Path,
    sample_size: int = 25,
    capacity_frac: float = 0.02,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    config = LLMConfig.from_env()
    client = get_llm_client(config)
    sample = assemble_flagged_sample(
        data_dir, sample_size=sample_size, capacity_frac=capacity_frac, seed=seed
    )
    log.info(
        "Assembled %d flagged transactions for explanation (provider=%s)",
        len(sample),
        config.provider,
    )

    semaphore = asyncio.Semaphore(config.concurrency)
    results = await asyncio.gather(
        *(_explain_one(client, semaphore, row, config.cache_path) for _, row in sample.iterrows())
    )

    total_cost = sum(r["cost_usd"] for r in results)
    output_rows = [
        {"transaction_id": r["transaction_id"], **r["explanation"].model_dump()} for r in results
    ]
    output_df = pd.DataFrame(output_rows)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_path, index=False)

    log.info("Generated %d explanations, total cost $%.4f", len(results), total_cost)
    return {
        "n_explanations": len(results),
        "total_cost_usd": total_cost,
        "provider": config.provider,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM explanations for a sample of flagged transactions."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--capacity-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="data/llm_explanations_sample.parquet")
    args = parser.parse_args()

    result = asyncio.run(
        run(
            Path(args.data_dir),
            sample_size=args.sample_size,
            capacity_frac=args.capacity_frac,
            seed=args.seed,
            output_path=Path(args.output_path),
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
