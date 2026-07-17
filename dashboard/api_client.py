from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    """Raised for any non-2xx response or connection failure - app.py catches
    this at each call site and shows a clean st.error instead of a raw
    traceback, since a demo dashboard talking to a possibly-not-running API
    is the normal case, not an exceptional one.
    """


def _request(method: str, base_url: str, path: str, timeout: float = 30, **kwargs: Any) -> Any:
    try:
        response = requests.request(method, f"{base_url}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.Timeout as exc:
        raise ApiError(
            f"Request to {base_url}{path} timed out after {timeout:.0f}s - if LLM_PROVIDER=ollama, "
            "the local model may just be slow on CPU (see CLAUDE.md's Week 4 notes), not actually "
            "stuck - the call will still fall back to a rule-based explanation server-side once "
            "its own retries finish."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Could not reach the API at {base_url} - is it running?") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(str(detail))
    return response.json()


def get_health(base_url: str) -> dict:
    return _request("GET", base_url, "/health")


def list_transactions(base_url: str, limit: int = 50) -> list[dict]:
    return _request("GET", base_url, "/transactions", params={"limit": limit})


def get_transaction(base_url: str, transaction_id: str) -> dict:
    return _request("GET", base_url, f"/transactions/{transaction_id}")


def explain_transaction(
    base_url: str, transaction_id: str, transaction: dict, features: dict, timeout: float = 300
) -> dict:
    # Generous default timeout: the Anthropic path replies in a few seconds,
    # but a local Ollama backend can legitimately take several minutes (slow
    # CPU inference across up to 3 repair-retry attempts - confirmed via a
    # real run: each attempt genuinely returns a 200 from Ollama, then fails
    # schema validation and gets a repair prompt, per the known Week 4
    # qwen2.5:3b finding) before falling back - this taking a while isn't the
    # same as it being broken.
    return _request(
        "POST",
        base_url,
        "/explain",
        timeout=timeout,
        json={"transaction_id": transaction_id, "transaction": transaction, "features": features},
    )


def submit_feedback(base_url: str, transaction_id: str, verdict: str, note: str | None) -> dict:
    return _request(
        "POST",
        base_url,
        f"/transactions/{transaction_id}/feedback",
        json={"verdict": verdict, "note": note},
    )


def list_feedback(base_url: str, transaction_id: str) -> list[dict]:
    return _request("GET", base_url, f"/transactions/{transaction_id}/feedback")
