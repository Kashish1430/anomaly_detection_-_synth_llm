from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from data_sim.config import SimConfig
from data_sim.customers import generate_customers
from data_sim.schemas import CustomerSchema, TransactionSchema
from data_sim.transactions import TRANSACTION_COLUMNS, generate_base_transactions
from data_sim.typologies import (
    inject_geographic_risk,
    inject_layering,
    inject_peer_deviation,
    inject_round_amount,
    inject_structuring,
    inject_velocity_spike,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INJECTORS = {
    "structuring": inject_structuring,
    "layering": inject_layering,
    "round_amount": inject_round_amount,
    "velocity_spike": inject_velocity_spike,
    "peer_deviation": inject_peer_deviation,
    "geographic_risk": inject_geographic_risk,
}


def run(config: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    t0 = time.time()

    log.info("Generating %d customers", config.n_customers)
    customers = generate_customers(config, rng)
    CustomerSchema.validate(customers)

    log.info("Generating base transactions (target ~%d)", config.target_n_transactions)
    base = generate_base_transactions(customers, config, rng)
    log.info("Generated %d base transactions in %.1fs", len(base), time.time() - t0)

    anomaly_budget = int(len(base) * config.anomaly_rate)
    injected_frames = []
    actual_counts = {}
    for name, fn in INJECTORS.items():
        n_target = max(1, int(anomaly_budget * config.typology_share[name]))
        frame = fn(customers, rng, config, n_target)
        actual_counts[name] = len(frame)
        injected_frames.append(frame)
        log.info("  %-16s injected %5d rows (target ~%d)", name, len(frame), n_target)

    transactions = pd.concat([base, *injected_frames], ignore_index=True)
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)
    transactions.insert(0, "transaction_id", [f"TXN{i:08d}" for i in range(len(transactions))])
    transactions = transactions[["transaction_id", *TRANSACTION_COLUMNS]]

    TransactionSchema.validate(transactions)

    manifest = {
        "seed": config.seed,
        "n_customers": len(customers),
        "n_transactions": len(transactions),
        "n_base_transactions": len(base),
        "anomaly_rate_target": config.anomaly_rate,
        "anomaly_rate_actual": float(transactions["is_anomalous"].mean()),
        "typology_counts": actual_counts,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "reporting_threshold": config.reporting_threshold,
        "generation_seconds": round(time.time() - t0, 2),
    }
    return customers, transactions, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic transaction dataset.")
    parser.add_argument("--n-customers", type=int, default=SimConfig.n_customers)
    parser.add_argument(
        "--n-transactions",
        type=int,
        dest="target_n_transactions",
        default=SimConfig.target_n_transactions,
    )
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--output-dir", type=str, default=SimConfig.output_dir)
    args = parser.parse_args()

    config = SimConfig(
        n_customers=args.n_customers,
        target_n_transactions=args.target_n_transactions,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    customers, transactions, manifest = run(config)

    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)
    customers.to_parquet(out / "customers.parquet", index=False)
    transactions.to_parquet(out / "transactions.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    log.info("Wrote %d customers, %d transactions -> %s", len(customers), len(transactions), out)
    log.info(
        "Actual anomaly rate: %.3f%% (target %.3f%%)",
        manifest["anomaly_rate_actual"] * 100,
        config.anomaly_rate * 100,
    )


if __name__ == "__main__":
    main()
