from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from data_sim.config import HIGH_RISK_COUNTRIES
from evaluation.fairness import (
    flagging_rate_by_group,
    parity_tests_vs_reference,
    performance_by_group,
)
from evaluation.splits import time_ordered_split
from features.pipeline import FEATURE_COLUMNS, build_feature_table
from models.baseline import fit_isolation_forest, score_anomaly
from models.lightgbm_model import load_tuned_model_and_threshold, predict_proba_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LGBM_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "if_anomaly_score"]
CUSTOMER_COLUMNS = ["customer_id", "segment", "home_country", "declared_risk_rating"]

# The Week 3 run that produced the reported 58.3% "after" precision (see
# CLAUDE.md "Measured results") - reused as-is instead of re-running the
# 30-trial Optuna search + OOF refitting that train_lightgbm.py needs for
# threshold *selection*, which this fairness check has no reason to repeat.
DEFAULT_RUN_ID = "e12a18e78ab144cea58c39d513d23007"

# Groups checked for statistical-parity-style flagging-rate disparities
# (PLAN.md §08), each with a reference category to compare the others against.
GROUP_COLUMNS = {
    "segment": "retail",
    "declared_risk_rating": "low",
    "country_risk_bucket": "standard_country",
}


def _load_features(
    data_dir: Path, transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    features_path = data_dir / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return build_feature_table(transactions, customers)


def run(
    data_dir: Path,
    run_id: str = DEFAULT_RUN_ID,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:

    model, threshold = load_tuned_model_and_threshold(run_id)

    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)
    # transactions and features both carry is_cross_border - drop the feature-table
    # copy before merging (same fix as generate_explanations.py:assemble_flagged_sample)
    features = features.drop(columns=["is_cross_border"])

    merged = transactions.merge(features, on="transaction_id", validate="one_to_one")
    merged = merged.merge(customers[CUSTOMER_COLUMNS], on="customer_id", validate="many_to_one")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["country_risk_bucket"] = (
        merged["home_country"]
        .isin(HIGH_RISK_COUNTRIES)
        .map({True: "high_risk_country", False: "standard_country"})
    )

    train_idx, _, test_idx = time_ordered_split(
        merged["timestamp"], train_frac=train_frac, val_frac=val_frac
    )

    # IsolationForest isn't saved by train_lightgbm.py's MLflow run (only the
    # final LightGBM is) - refit it here to reproduce the if_anomaly_score
    # feature the tuned model expects. Deterministic given the same seed and
    # split, and cheap next to the Optuna search this script is avoiding.
    if_model = fit_isolation_forest(
        merged.iloc[train_idx][FEATURE_COLUMNS], n_estimators=200, random_state=seed
    )
    merged["if_anomaly_score"] = score_anomaly(if_model, merged[FEATURE_COLUMNS])

    test = merged.iloc[test_idx].reset_index(drop=True)
    proba_test = predict_proba_anomaly(model, test[LGBM_FEATURE_COLUMNS])
    flagged = pd.Series(proba_test >= threshold)
    log.info(
        "TEST rows: %d | flagged: %d (%.2f%%) at threshold %.4f",
        len(test),
        int(flagged.sum()),
        100 * flagged.mean(),
        threshold,
    )

    y_true = test["is_anomalous"].astype(bool)

    results: dict = {}
    for group_col, reference in GROUP_COLUMNS.items():
        group = test[group_col]
        rates = flagging_rate_by_group(flagged, group, alpha=alpha)
        parity = parity_tests_vs_reference(flagged, group, reference=reference, alpha=alpha)
        # ground-truth-aware complement to the two checks above: tells apart a
        # flagging-rate gap explained by a genuinely different true anomaly
        # rate from one that isn't (see evaluation/fairness.py docstring)
        performance = performance_by_group(y_true, flagged, group)
        results[group_col] = {
            "reference": reference,
            "rates": rates.to_dict(orient="records"),
            "parity_tests": parity.to_dict(orient="records"),
            "performance": performance.to_dict(orient="records"),
        }
        log.info("%s flagging rates:\n%s", group_col, rates.to_string(index=False))
        log.info("%s parity tests vs %r:\n%s", group_col, reference, parity.to_string(index=False))
        log.info(
            "%s true-rate/precision/recall:\n%s", group_col, performance.to_string(index=False)
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bias/fairness check: flagging-rate parity across customer segments "
        "(PLAN.md §08)."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-path", type=str, default="data/fairness_check_results.json")
    args = parser.parse_args()

    result = run(
        Path(args.data_dir),
        run_id=args.run_id,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        alpha=args.alpha,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Wrote fairness results to %s", output_path)


if __name__ == "__main__":
    main()
