from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from evaluation.splits import time_ordered_split
from features.pipeline import FEATURE_COLUMNS, build_feature_table
from models.baseline import fit_isolation_forest
from models.lightgbm_model import load_tuned_model_and_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LGBM_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "if_anomaly_score"]

# The Week 3 run that produced the reported 58.3% "after" precision (see
# CLAUDE.md "Measured results") - same run_id run_fairness_check.py and
# run_sensitivity_check.py already load from, reused here rather than
# re-running the 30-trial Optuna search.
DEFAULT_RUN_ID = "e12a18e78ab144cea58c39d513d23007"


def _load_features(
    data_dir: Path, transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    features_path = data_dir / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return build_feature_table(transactions, customers)


def build_bundle(
    data_dir: Path,
    run_id: str = DEFAULT_RUN_ID,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """Bundles everything api/ will need to score a live transaction into one
    self-contained object: the tuned LightGBM model + its capacity threshold
    (already logged to MLflow by train_lightgbm.py), plus an IsolationForest
    fit on the same TRAIN split. train_lightgbm.py never persists the
    IsolationForest - it only needed its score as a transient input feature -
    so every downstream script that needs it (run_fairness_check.py,
    run_sensitivity_check.py) refits it themselves. This does the same
    deterministic refit (same seed + split) once, at packaging time, so api/
    can load a single artifact instead of refitting on every process start.
    """
    lightgbm_model, capacity_threshold = load_tuned_model_and_threshold(run_id)

    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)

    merged = transactions[["transaction_id", "timestamp"]].merge(
        features, on="transaction_id", validate="one_to_one"
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    train_idx, _, _ = time_ordered_split(
        merged["timestamp"], train_frac=train_frac, val_frac=val_frac
    )
    isolation_forest = fit_isolation_forest(
        merged.iloc[train_idx][FEATURE_COLUMNS], n_estimators=200, random_state=seed
    )

    return {
        "isolation_forest": isolation_forest,
        "lightgbm_model": lightgbm_model,
        "capacity_threshold": capacity_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "lgbm_feature_columns": LGBM_FEATURE_COLUMNS,
        "metadata": {
            "mlflow_run_id": run_id,
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "packaged_at": datetime.now(UTC).isoformat(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the tuned LightGBM model + a matching IsolationForest into one "
        "artifact file for api/ to load, instead of refitting from raw data (PLAN.md §09)."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="artifacts/model_bundle.joblib")
    args = parser.parse_args()

    bundle = build_bundle(
        Path(args.data_dir),
        run_id=args.run_id,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    log.info(
        "Wrote model bundle to %s (%.1f MB)",
        output_path,
        output_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
