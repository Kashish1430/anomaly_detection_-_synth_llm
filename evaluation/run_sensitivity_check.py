from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from evaluation.sensitivity import decision_flip_rate, feature_sensitivity
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
    delta_in_std: float = 1.0,
    noise_std_frac: float = 0.01,
    sample_size: int = 20_000,
) -> dict:
    model, threshold = load_tuned_model_and_threshold(run_id)

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

    # A random sample of TEST, not all of it - feature_sensitivity calls
    # predict_fn 2 * n_features times, so this is deliberately capped for speed;
    # sample_size=20,000 is still large enough for the mean deltas to be stable.
    test = merged.iloc[test_idx].reset_index(drop=True)
    if len(test) > sample_size:
        test = test.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    X_test = test[LGBM_FEATURE_COLUMNS]

    def predict_fn(X: pd.DataFrame) -> pd.Series:
        return predict_proba_anomaly(model, X)

    log.info(
        "Running feature sensitivity on %d rows x %d features",
        len(X_test),
        len(LGBM_FEATURE_COLUMNS),
    )
    sensitivity = feature_sensitivity(
        predict_fn, X_test, LGBM_FEATURE_COLUMNS, delta_in_std=delta_in_std
    )
    log.info("Feature sensitivity (sorted by impact):\n%s", sensitivity.to_string(index=False))

    flip_rate = decision_flip_rate(
        predict_fn, X_test, threshold=threshold, noise_std_frac=noise_std_frac, seed=seed
    )
    log.info(
        "Decision flip rate under %.1f%% feature noise: %.4f%% of rows",
        noise_std_frac * 100,
        flip_rate * 100,
    )

    return {
        "n_rows": len(X_test),
        "delta_in_std": delta_in_std,
        "noise_std_frac": noise_std_frac,
        "threshold": threshold,
        "feature_sensitivity": sensitivity.to_dict(orient="records"),
        "decision_flip_rate": flip_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: feature perturbation and decision-flip robustness "
        "(PLAN.md §08)."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in-std", type=float, default=1.0)
    parser.add_argument("--noise-std-frac", type=float, default=0.01)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--output-path", type=str, default="data/sensitivity_check_results.json")
    args = parser.parse_args()

    result = run(
        Path(args.data_dir),
        run_id=args.run_id,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        delta_in_std=args.delta_in_std,
        noise_std_frac=args.noise_std_frac,
        sample_size=args.sample_size,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Wrote sensitivity results to %s", output_path)


if __name__ == "__main__":
    main()
