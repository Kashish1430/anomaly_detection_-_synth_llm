from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from api.config import ApiConfig
from api.model_bundle import load_bundle
from evaluation.splits import time_ordered_split
from features.pipeline import FEATURE_COLUMNS, build_feature_table
from models.baseline import score_anomaly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LGBM_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "if_anomaly_score"]
CUSTOMER_COLUMNS = ["customer_id", "segment", "home_country", "declared_risk_rating", "peer_group"]


def _load_features(
    data_dir: Path, transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    features_path = data_dir / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return build_feature_table(transactions, customers)


def build_scored_flagged_transactions(
    data_dir: Path,
    bundle: dict,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """Scores the TEST split - the same untouched split package_artifact.py's
    bundle was evaluated on - with the real tuned model, and returns only the
    flagged rows plus their engineered features as a JSON-ready dict per row.

    This is what Postgres's `transactions` table gets loaded with: real flagged
    cases from the actual model, not a separate hand-curated sample file - see
    PLAN.md §06, whose "~300 curated cases" idea becomes, once Postgres exists,
    just a subset of these rows getting explanations pre-generated and cached.
    """
    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)
    # is_cross_border is both a raw transaction field and a passthrough
    # engineered feature - same collision fix used everywhere else this merge
    # happens (generate_explanations.py, run_fairness_check.py).
    features = features.drop(columns=["is_cross_border"])

    merged = transactions.merge(features, on="transaction_id", validate="one_to_one")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    _, _, test_idx = time_ordered_split(
        merged["timestamp"], train_frac=train_frac, val_frac=val_frac
    )
    test = merged.iloc[test_idx].reset_index(drop=True)

    test["if_anomaly_score"] = score_anomaly(bundle["isolation_forest"], test[FEATURE_COLUMNS])
    proba = bundle["lightgbm_model"].predict_proba(test[LGBM_FEATURE_COLUMNS])[:, 1]
    test["anomaly_probability"] = proba
    test["is_flagged"] = proba >= bundle["capacity_threshold"]

    flagged = test[test["is_flagged"]].reset_index(drop=True)
    # pandas' own JSON serializer, not a manual dict conversion - engineered
    # feature columns are a mix of numpy float64/bool_/int32, none of which
    # `json.dumps` (used by Jsonb below) can serialize directly.
    flagged["features"] = json.loads(flagged[FEATURE_COLUMNS].to_json(orient="records"))
    return flagged


def _none_if_nan(value):
    return None if pd.isna(value) else value


def _transaction_row_to_tuple(row: pd.Series) -> tuple:
    return (
        row["transaction_id"],
        row["customer_id"],
        row["timestamp"].to_pydatetime(),
        float(row["amount"]),
        row["direction"],
        row["channel"],
        _none_if_nan(row["counterparty_id"]),
        _none_if_nan(row["counterparty_country"]),
        bool(row["is_cross_border"]),
        Jsonb(row["features"]),
        float(row["anomaly_probability"]),
        bool(row["is_flagged"]),
        bool(row["is_anomalous"]),
        _none_if_nan(row["typology"]),
    )


def load_into_postgres(
    database_url: str, customers: pd.DataFrame, flagged_transactions: pd.DataFrame
) -> None:
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO customers
                (customer_id, segment, home_country, declared_risk_rating, peer_group)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            list(customers[CUSTOMER_COLUMNS].itertuples(index=False, name=None)),
        )
        cur.executemany(
            """
            INSERT INTO transactions (
                transaction_id, customer_id, "timestamp", amount, direction, channel,
                counterparty_id, counterparty_country, is_cross_border, features,
                anomaly_probability, is_flagged, is_anomalous, typology
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            [_transaction_row_to_tuple(row) for _, row in flagged_transactions.iterrows()],
        )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the TEST split with the tuned model and load the flagged "
        "transactions into Postgres for the dashboard to browse (PLAN.md §02, §11)."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    config = ApiConfig.from_env()
    bundle = load_bundle(config.model_bundle_path)

    data_dir = Path(args.data_dir)
    customers = pd.read_parquet(data_dir / "customers.parquet")
    flagged = build_scored_flagged_transactions(
        data_dir, bundle, train_frac=args.train_frac, val_frac=args.val_frac
    )
    log.info("Scored TEST split: %d flagged transactions to load", len(flagged))

    load_into_postgres(config.database_url, customers, flagged)
    log.info(
        "Loaded %d customers and %d flagged transactions into Postgres",
        len(customers),
        len(flagged),
    )


if __name__ == "__main__":
    main()
