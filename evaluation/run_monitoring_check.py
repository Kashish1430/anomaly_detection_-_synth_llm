from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from evaluation.monitoring import population_stability_index, psi_report
from evaluation.splits import time_ordered_split
from features.pipeline import FEATURE_COLUMNS, build_feature_table
from models.baseline import fit_isolation_forest, score_anomaly
from models.lightgbm_model import load_tuned_model_and_threshold, predict_proba_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LGBM_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "if_anomaly_score"]

# Same Week 3 run as evaluation/run_fairness_check.py - see that file's comment
# for why this is loaded rather than re-tuned.
DEFAULT_RUN_ID = "e12a18e78ab144cea58c39d513d23007"


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
) -> dict:
    model, _ = load_tuned_model_and_threshold(run_id)

    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)
    features = features.drop(columns=["is_cross_border"])

    merged = transactions.merge(features, on="transaction_id", validate="one_to_one")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    train_idx, _, test_idx = time_ordered_split(
        merged["timestamp"], train_frac=train_frac, val_frac=val_frac
    )

    if_model = fit_isolation_forest(
        merged.iloc[train_idx][FEATURE_COLUMNS], n_estimators=200, random_state=seed
    )
    merged["if_anomaly_score"] = score_anomaly(if_model, merged[FEATURE_COLUMNS])

    train = merged.iloc[train_idx].reset_index(drop=True)
    test = merged.iloc[test_idx].reset_index(drop=True)

    # PSI on every model input feature - TRAIN is the "baseline" population the
    # model was fit on, TEST is a later, disjoint time window standing in for
    # "the live population today" (PLAN.md §08 monitoring plan).
    feature_psi = psi_report(train, test, LGBM_FEATURE_COLUMNS)
    log.info(
        "Feature PSI (TRAIN vs. TEST, sorted by severity):\n%s", feature_psi.to_string(index=False)
    )

    # PSI on the model's own output score - the single most important number
    # to monitor in production, since it can drift even when no individual
    # input feature crosses the significant-shift threshold on its own.
    score_psi = population_stability_index(
        predict_proba_anomaly(model, train[LGBM_FEATURE_COLUMNS]),
        predict_proba_anomaly(model, test[LGBM_FEATURE_COLUMNS]),
    )
    log.info("Score PSI (TRAIN vs. TEST): %.4f", score_psi)

    return {
        "feature_psi": feature_psi.to_dict(orient="records"),
        "score_psi": score_psi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitoring check: PSI of model inputs and output score, TRAIN vs. TEST "
        "(PLAN.md §08)."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="data/monitoring_check_results.json")
    args = parser.parse_args()

    result = run(
        Path(args.data_dir),
        run_id=args.run_id,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Wrote monitoring results to %s", output_path)


if __name__ == "__main__":
    main()
