from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from api.config import ApiConfig
from features.peer_deviation import compute_peer_group_stats
from features.pipeline import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRANSACTION_COLUMNS = """
    transaction_id, customer_id, "timestamp", amount, direction, channel,
    counterparty_id, counterparty_country, is_cross_border, features,
    anomaly_probability, is_flagged, is_anomalous, typology
"""


def _load_full_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = pd.read_parquet(data_dir / "features.parquet").drop(columns=["is_cross_border"])
    merged = transactions.merge(features, on="transaction_id", validate="one_to_one")
    return merged, customers


def _none_if_nan(value):
    return None if pd.isna(value) else value


def _history_row_tuple(row) -> tuple:
    """A row for the bulk COPY - history-only rows (everything not already
    loaded by api/load_data.py's real scoring run) get is_flagged=False,
    anomaly_probability=0.0 as an explicit placeholder, never a real score.
    These rows exist purely as feature-computation context for
    api/live_features.py - they're never surfaced by GET /transactions,
    which filters WHERE is_flagged.
    """
    features = {col: row[col] for col in FEATURE_COLUMNS}
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
        Jsonb(features),
        0.0,
        False,
        None,
        None,
    )


CHUNK_SIZE = 100_000


def load_history(
    database_url: str,
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Bulk-loads every transaction not already in Postgres via COPY (not
    executemany - see CLAUDE.md's Week 7 load_data.py incident on why bulk
    row-by-row inserts don't belong on this box) into `transactions`, and
    upserts peer_group_stats. Idempotent: existing transaction_ids (the
    ~4,060 already loaded with real scores by api/load_data.py, plus
    whatever any prior partial run of this script already committed) are
    excluded from the COPY rather than relied on to conflict-skip, since
    COPY has no native ON CONFLICT handling.

    COPYs in chunk_size-row batches, each its own transaction, rather than
    one multi-hour COPY - confirmed by real testing that a single giant COPY
    over an SSH tunnel is fragile to any connection blip on either end (two
    real failures, at 35 and 72 minutes, both a client-side "unexpected EOF"
    per Postgres's own logs - not a box or Postgres problem). Chunking means
    a drop loses minutes of progress, not over an hour, and leaves far less
    uncommitted-transaction bloat for autovacuum to clean up afterward
    (confirmed: the second failure's rollback left a zombie backend holding
    locks that needed a manual pg_terminate_backend, and made even a plain
    COUNT(*) crawl until autovacuum finished).
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT transaction_id FROM transactions")
            existing_ids = {row[0] for row in cur.fetchall()}
        to_load = transactions[~transactions["transaction_id"].isin(existing_ids)]
        log.info(
            "%d transactions already loaded, %d new to COPY in %d-row chunks",
            len(existing_ids),
            len(to_load),
            chunk_size,
        )

        loaded = 0
        for start in range(0, len(to_load), chunk_size):
            chunk = to_load.iloc[start : start + chunk_size]
            with conn.cursor() as cur:
                with cur.copy(f"COPY transactions ({TRANSACTION_COLUMNS}) FROM STDIN") as copy:
                    for _, row in chunk.iterrows():
                        copy.write_row(_history_row_tuple(row))
            conn.commit()
            loaded += len(chunk)
            log.info("Committed chunk: %d / %d rows loaded so far", loaded, len(to_load))

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS peer_group_stats (
                    peer_group TEXT PRIMARY KEY,
                    peer_median DOUBLE PRECISION NOT NULL,
                    peer_mad DOUBLE PRECISION NOT NULL
                )
                """)
            peer_stats = compute_peer_group_stats(transactions, customers).reset_index()
            cur.executemany(
                """
                INSERT INTO peer_group_stats (peer_group, peer_median, peer_mad)
                VALUES (%s, %s, %s)
                ON CONFLICT (peer_group) DO UPDATE SET
                    peer_median = EXCLUDED.peer_median, peer_mad = EXCLUDED.peer_mad
                """,
                list(
                    peer_stats[["peer_group", "peer_median", "peer_mad"]].itertuples(
                        index=False, name=None
                    )
                ),
            )
            conn.commit()
        log.info("Upserted %d peer_group_stats rows", len(peer_stats))
        return loaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time bulk load of the full transaction history + peer-group "
        "stats into Postgres, so api/live_features.py has real customer history to "
        "compute features from for a genuinely new transaction (PLAN.md's live-predict "
        "gap). Heavy step - loads the full 1.2M-row dataset - meant to run once, "
        "offline/manually, not on every deploy."
    )
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    args = parser.parse_args()

    config = ApiConfig.from_env()
    transactions, customers = _load_full_dataset(Path(args.data_dir))
    n_loaded = load_history(config.database_url, transactions, customers)
    log.info("Done: %d new transactions loaded", n_loaded)


if __name__ == "__main__":
    main()
