from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from evaluation.effort_reduction import effort_reduction_summary
from evaluation.splits import time_ordered_split
from features.pipeline import FEATURE_COLUMNS, build_feature_table
from models.baseline import fit_isolation_forest, score_anomaly
from models.lightgbm_model import load_tuned_model_and_threshold, predict_proba_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LGBM_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "if_anomaly_score"]

# Same run as run_fairness_check.py / run_sensitivity_check.py - the Week 3
# run behind the reported 58.3% "after" precision (CLAUDE.md "Measured
# results"), reused rather than re-run.
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
    capacity_frac: float = 0.02,
    seed: int = 42,
) -> dict:
    model, _threshold = load_tuned_model_and_threshold(run_id)

    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)
    # transactions and features both carry is_cross_border - drop the feature-table
    # copy before merging (same fix as generate_explanations.py:assemble_flagged_sample)
    features = features.drop(columns=["is_cross_border"])

    merged = transactions.merge(features, on="transaction_id", validate="one_to_one")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    train_idx, _, test_idx = time_ordered_split(
        merged["timestamp"], train_frac=train_frac, val_frac=val_frac
    )

    # Same IsolationForest serves two roles here, both fit on TRAIN only: the
    # standalone "before" baseline this check compares against, and the
    # if_anomaly_score input feature the tuned LightGBM model expects - it
    # isn't persisted by train_lightgbm.py's MLflow run, only refit
    # (deterministic given seed+split, same trick run_fairness_check.py uses).
    if_model = fit_isolation_forest(
        merged.iloc[train_idx][FEATURE_COLUMNS], n_estimators=200, random_state=seed
    )
    merged["if_anomaly_score"] = score_anomaly(if_model, merged[FEATURE_COLUMNS])

    test = merged.iloc[test_idx].reset_index(drop=True)
    y_true = test["is_anomalous"].to_numpy()
    baseline_score = test["if_anomaly_score"].to_numpy()
    tuned_score = predict_proba_anomaly(model, test[LGBM_FEATURE_COLUMNS])

    result = effort_reduction_summary(y_true, baseline_score, tuned_score, capacity_frac)

    log.info(
        "Baseline @ %.1f%% capacity: precision=%.3f recall=%.3f",
        capacity_frac * 100,
        result["baseline_metrics"]["precision"],
        result["baseline_metrics"]["recall"],
    )
    log.info(
        "Tuned @ %.1f%% capacity: precision=%.3f recall=%.3f",
        capacity_frac * 100,
        result["tuned_metrics"]["precision"],
        result["tuned_metrics"]["recall"],
    )
    log.info(
        "FP reduction at fixed capacity: %.1f%%", 100 * result["fp_reduction_at_fixed_capacity"]
    )
    log.info(
        "Tuned model needs %.3f%% capacity to match baseline's recall -> "
        "manual review effort reduction: %.1f%%",
        100 * result["equivalent_capacity_frac_for_tuned_model"],
        100 * result["manual_review_effort_reduction"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual review effort reduction check: how much less review "
        "volume the tuned model needs to match the baseline's recall, plus false "
        "positives avoided at fixed capacity (PLAN.md §15 CV bullets)."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--capacity-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path", type=str, default="data/effort_reduction_check_results.json"
    )
    args = parser.parse_args()

    result = run(
        Path(args.data_dir),
        run_id=args.run_id,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        capacity_frac=args.capacity_frac,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Wrote effort reduction results to %s", output_path)


if __name__ == "__main__":
    main()
