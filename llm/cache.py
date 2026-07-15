from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from llm.schemas import ExplanationOutput

PROMPT_VERSION = "v2"  # bumped when the SHAP-attribution prompt shape was added


def cache_key(transaction_id: str, provider: str, model: str) -> str:
    raw = f"{transaction_id}:{provider}:{model}:{PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _connect(cache_path: str) -> sqlite3.Connection:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS explanations ("
        "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    return conn


def get_cached(cache_path: str, key: str) -> ExplanationOutput | None:
    conn = _connect(cache_path)
    try:
        row = conn.execute(
            "SELECT payload FROM explanations WHERE cache_key = ?", (key,)
        ).fetchone()
        return ExplanationOutput.model_validate_json(row[0]) if row else None
    finally:
        conn.close()


def set_cached(cache_path: str, key: str, value: ExplanationOutput) -> None:
    conn = _connect(cache_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO explanations (cache_key, payload) VALUES (?, ?)",
            (key, value.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()
