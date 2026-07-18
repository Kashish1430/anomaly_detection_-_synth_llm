from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

TRANSACTION_SELECT_COLUMNS = """
    transaction_id, customer_id, "timestamp", amount, direction, channel,
    counterparty_id, counterparty_country, is_cross_border, features,
    anomaly_probability, is_flagged, is_anomalous, typology
"""


def open_pool(database_url: str) -> AsyncConnectionPool:
    """Constructs (but doesn't yet connect) a pooled connection - api/main.py's
    lifespan hook awaits `.open()` at startup and `.close()` at shutdown, the
    same lifecycle as the model bundle load.
    """
    return AsyncConnectionPool(database_url, open=False, kwargs={"row_factory": dict_row})


async def list_flagged_transactions(
    pool: AsyncConnectionPool, limit: int = 50
) -> list[dict[str, Any]]:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            f"""
            SELECT {TRANSACTION_SELECT_COLUMNS}
            FROM transactions
            WHERE is_flagged
            ORDER BY anomaly_probability DESC
            LIMIT %s
            """,
            (limit,),
        )
        return await cur.fetchall()


async def get_transaction(pool: AsyncConnectionPool, transaction_id: str) -> dict[str, Any] | None:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            f"""
            SELECT {TRANSACTION_SELECT_COLUMNS}
            FROM transactions
            WHERE transaction_id = %s
            """,
            (transaction_id,),
        )
        return await cur.fetchone()


async def get_customer(pool: AsyncConnectionPool, customer_id: str) -> dict[str, Any] | None:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT customer_id, segment, home_country, declared_risk_rating, peer_group
            FROM customers
            WHERE customer_id = %s
            """,
            (customer_id,),
        )
        return await cur.fetchone()


async def insert_customer(
    pool: AsyncConnectionPool,
    customer_id: str,
    segment: str,
    home_country: str,
    declared_risk_rating: str,
    peer_group: str,
) -> dict[str, Any]:
    """Registers a customer_id /transactions/predict has never seen before -
    see api/live_features.py. peer_group must already be derived by the
    caller (segment_home_country, matching data_sim/customers.py's own
    convention) since that's what get_peer_group_stats looks up by.
    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            INSERT INTO customers
                (customer_id, segment, home_country, declared_risk_rating, peer_group)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING customer_id, segment, home_country, declared_risk_rating, peer_group
            """,
            (customer_id, segment, home_country, declared_risk_rating, peer_group),
        )
        row = await cur.fetchone()
        assert row is not None
        return row


async def list_customer_transactions(
    pool: AsyncConnectionPool, customer_id: str
) -> list[dict[str, Any]]:
    """This customer's full transaction history, oldest first - the input
    api/live_features.py appends a new raw transaction to before re-running
    features/*.py. Bounded to one customer (~100 rows typically, per the Week
    7 sizing check), not the full multi-million-row table, via the existing
    idx_transactions_customer index.
    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT transaction_id, customer_id, "timestamp", amount, direction, channel,
                   counterparty_id, counterparty_country, is_cross_border
            FROM transactions
            WHERE customer_id = %s
            ORDER BY "timestamp"
            """,
            (customer_id,),
        )
        return await cur.fetchall()


async def get_peer_group_stats(pool: AsyncConnectionPool, peer_group: str) -> dict[str, Any] | None:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT peer_group, peer_median, peer_mad FROM peer_group_stats WHERE peer_group = %s",
            (peer_group,),
        )
        return await cur.fetchone()


async def insert_scored_transaction(
    pool: AsyncConnectionPool,
    transaction_id: str,
    customer_id: str,
    timestamp: datetime,
    amount: float,
    direction: str,
    channel: str,
    counterparty_id: str | None,
    counterparty_country: str | None,
    is_cross_border: bool,
    features: dict,
    anomaly_probability: float,
    is_flagged: bool,
) -> dict[str, Any]:
    """Persists a transaction /transactions/predict just scored live -
    is_anomalous/typology are NULL (unlike the synthetic batch-loaded rows),
    since there's no ground truth for a genuinely new transaction.
    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            f"""
            INSERT INTO transactions (
                transaction_id, customer_id, "timestamp", amount, direction, channel,
                counterparty_id, counterparty_country, is_cross_border, features,
                anomaly_probability, is_flagged, is_anomalous, typology
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL)
            RETURNING {TRANSACTION_SELECT_COLUMNS}
            """,
            (
                transaction_id,
                customer_id,
                timestamp,
                amount,
                direction,
                channel,
                counterparty_id,
                counterparty_country,
                is_cross_border,
                Jsonb(features),
                anomaly_probability,
                is_flagged,
            ),
        )
        row = await cur.fetchone()
        assert row is not None
        return row


async def insert_explanation(
    pool: AsyncConnectionPool,
    transaction_id: str,
    explanation: str,
    typology: str,
    confidence: float,
    likely_false_positive: bool,
    source: str,
    fact_check_passed: bool | None,
) -> dict[str, Any]:
    """First real user of the `explanations` table - /explain (the older
    endpoint) computes an explanation but never persists it, only returns it.
    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            INSERT INTO explanations
                (transaction_id, explanation, typology, confidence, likely_false_positive,
                 source, fact_check_passed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING transaction_id, explanation, typology, confidence, likely_false_positive,
                      source, fact_check_passed, generated_at
            """,
            (
                transaction_id,
                explanation,
                typology,
                confidence,
                likely_false_positive,
                source,
                fact_check_passed,
            ),
        )
        row = await cur.fetchone()
        assert row is not None
        return row


async def list_feedback(pool: AsyncConnectionPool, transaction_id: str) -> list[dict[str, Any]]:
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT id, transaction_id, verdict, note, submitted_at
            FROM investigator_feedback
            WHERE transaction_id = %s
            ORDER BY submitted_at DESC
            """,
            (transaction_id,),
        )
        return await cur.fetchall()


async def insert_feedback(
    pool: AsyncConnectionPool, transaction_id: str, verdict: str, note: str | None
) -> dict[str, Any]:
    """Inserts one investigator verdict. Callers should check the transaction
    exists first (api/main.py does, via get_transaction) rather than relying on
    the table's foreign key to reject it - a 404 is a clearer response than a
    500 from an unhandled ForeignKeyViolation.
    """
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            INSERT INTO investigator_feedback (transaction_id, verdict, note)
            VALUES (%s, %s, %s)
            RETURNING id, transaction_id, verdict, note, submitted_at
            """,
            (transaction_id, verdict, note),
        )
        row = await cur.fetchone()
        assert row is not None
        return row
