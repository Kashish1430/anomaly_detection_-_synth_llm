from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from api.config import ApiConfig
from api.load_data import CUSTOMER_COLUMNS, build_scored_flagged_transactions
from api.model_bundle import load_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the TEST split with the tuned model and write the flagged "
        "transactions + customers to a small parquet sample (PLAN.md §09: heavy work - "
        "loading/merging/sorting the full 1.2M-row dataset and running model inference - "
        "happens offline/locally, not on the small EC2 serving box). Run this locally, "
        "then ship data/scored_sample/ as a GitHub Release asset; api/load_data.py loads "
        "it into Postgres on the target box without touching the full dataset."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--out-dir", type=str, default="data/scored_sample")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    config = ApiConfig.from_env()
    bundle = load_bundle(config.model_bundle_path)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flagged = build_scored_flagged_transactions(
        data_dir, bundle, train_frac=args.train_frac, val_frac=args.val_frac
    )
    log.info("Scored TEST split: %d flagged transactions", len(flagged))

    # All customers, not just those with a flagged transaction - matches the
    # original load_data.py behaviour and what the dashboard/docs document
    # (12,000 customers loaded regardless of flagging).
    customers = pd.read_parquet(data_dir / "customers.parquet")[CUSTOMER_COLUMNS]

    flagged.to_parquet(out_dir / "flagged_transactions.parquet", index=False)
    customers.to_parquet(out_dir / "customers.parquet", index=False)
    log.info(
        "Wrote %d flagged transactions and %d customers to %s",
        len(flagged),
        len(customers),
        out_dir,
    )


if __name__ == "__main__":
    main()
