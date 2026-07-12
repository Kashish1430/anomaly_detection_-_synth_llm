from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from features.pipeline import FEATURE_COLUMNS, build_feature_table
from models.baseline import fit_isolation_forest, score_anomaly
from models.evaluate import evaluate_at_capacity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_features(
    data_dir: Path, transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    features_path = data_dir / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return build_feature_table(transactions, customers)


def run(
    data_dir: Path,
    capacity_frac: float = 0.02,
    test_size: float = 0.2,
    n_estimators: int = 200,
    seed: int = 42,
    log_to_mlflow: bool = True,
) -> dict:
    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)

    merged = transactions[["transaction_id", "timestamp", "is_anomalous"]].merge(
        features, on="transaction_id", validate="one_to_one"
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # time-based split, not random - PLAN.md §07 treats shuffled-data validation
    # as a false signal for a problem where patterns drift over time
    split_idx = int(len(merged) * (1 - test_size))
    split_time = str(merged.loc[split_idx, "timestamp"])
    train, test = merged.iloc[:split_idx], merged.iloc[split_idx:]

    model = fit_isolation_forest(
        train[FEATURE_COLUMNS], n_estimators=n_estimators, random_state=seed
    )
    anomaly_score = score_anomaly(model, test[FEATURE_COLUMNS])
    y_test = test["is_anomalous"].to_numpy()
    metrics = evaluate_at_capacity(anomaly_score, y_test, capacity_frac)

    params = {
        "model": "IsolationForest",
        "n_estimators": n_estimators,
        "n_features": len(FEATURE_COLUMNS),
        "train_rows": len(train),
        "test_rows": len(test),
        "split_time": split_time,
        "seed": seed,
    }

    if log_to_mlflow:
        mlflow.set_experiment("anomaly-detection-baseline")
        with mlflow.start_run(run_name="isolation_forest_naive_threshold"):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
            mlflow.sklearn.log_model(model, "isolation_forest")

    log.info("train=%d test=%d split_time=%s", len(train), len(test), split_time)
    log.info(
        "Baseline ('before') precision @ %.1f%% capacity: %.3f | recall: %.3f | f1: %.3f | "
        "pr_auc: %.3f",
        capacity_frac * 100,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["pr_auc"],
    )
    return {"params": params, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the IsolationForest baseline.")
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--capacity-frac", type=float, default=0.02)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    result = run(
        Path(args.data_dir),
        capacity_frac=args.capacity_frac,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        seed=args.seed,
        log_to_mlflow=not args.no_mlflow,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
