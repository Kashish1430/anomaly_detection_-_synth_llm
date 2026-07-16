from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
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
