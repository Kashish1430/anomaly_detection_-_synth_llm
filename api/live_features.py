from __future__ import annotations

import json
from typing import Any

import pandas as pd

from api.schemas import RawTransactionRequest, TransactionFeatures
from features.behavioral import compute_behavioral_features
from features.contextual import compute_contextual_features
from features.peer_deviation import MAD_TO_STD
from features.pipeline import FEATURE_COLUMNS
from features.round_amount import compute_round_amount_features
from features.velocity import compute_velocity_features

PEER_FREE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "peer_zscore"]


def _build_history_dataframe(
    history: list[dict[str, Any]],
    new_txn: RawTransactionRequest,
    new_txn_id: str,
    customer_id: str,
) -> pd.DataFrame:
    """One customer's transaction history plus the new transaction as its
    latest row - the same shape features/*.py's batch functions already
    expect, just scoped to one customer instead of the full dataset. A brand
    new customer has an empty `history`, which is fine: every feature module
    already has a documented "no prior data" fallback (behavioral.py's 1.0
    neutral ratio, safe_zscore's NaN->0.0 floor - see features/utils.py).
    """
    rows = [
        *history,
        {
            "transaction_id": new_txn_id,
            "customer_id": customer_id,
            "timestamp": new_txn.timestamp,
            "amount": new_txn.amount,
            "direction": new_txn.direction,
            "channel": new_txn.channel,
            "counterparty_id": new_txn.counterparty_id,
            "counterparty_country": new_txn.counterparty_country,
            "is_cross_border": new_txn.is_cross_border,
        },
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _peer_zscore(amount: float, peer_stats: dict[str, Any] | None) -> float:
    """Same formula as features/peer_deviation.py's compute_peer_deviation_features,
    just fed a precomputed (peer_median, peer_mad) lookup instead of a full peer
    group's raw rows - see api/load_full_history.py for where that lookup is
    populated. A peer_group with no stats (only possible for a segment/country
    combination outside data_sim's known set, since load_full_history.py covers
    all of them) falls back to the neutral 0.0, matching how every other
    "no prior information" case in features/*.py already degrades.
    """
    if peer_stats is None:
        return 0.0
    mad_safe = peer_stats["peer_mad"] * MAD_TO_STD
    if not mad_safe:
        return 0.0
    return float((amount - peer_stats["peer_median"]) / mad_safe)


def compute_live_features(
    history: list[dict[str, Any]],
    peer_stats: dict[str, Any] | None,
    new_txn: RawTransactionRequest,
    new_txn_id: str,
    customer_id: str,
) -> TransactionFeatures:
    """Computes the 18 engineered features for one new transaction, reusing
    features/velocity.py, behavioral.py, round_amount.py, and contextual.py
    completely unchanged - only peer_deviation.py's group-level computation is
    swapped for a precomputed lookup (see _peer_zscore), since that one needs
    a full peer group's rows rather than just this customer's own history.
    """
    df = _build_history_dataframe(history, new_txn, new_txn_id, customer_id)

    velocity = compute_velocity_features(df)
    behavioral = compute_behavioral_features(df)
    round_amount = compute_round_amount_features(df)
    contextual = compute_contextual_features(df)

    merged = velocity
    for other in (behavioral, round_amount, contextual):
        merged = merged.merge(other, on="transaction_id", how="inner", validate="one_to_one")

    new_row = merged[merged["transaction_id"] == new_txn_id][PEER_FREE_FEATURE_COLUMNS]
    # Same pandas->native-Python trick load_data.py uses (json round-trip) -
    # numpy scalar types (np.bool_, np.int64, ...) don't reliably satisfy
    # TransactionFeatures' plain bool/int/float field types otherwise.
    feature_dict = json.loads(new_row.to_json(orient="records"))[0]
    feature_dict["peer_zscore"] = _peer_zscore(new_txn.amount, peer_stats)

    return TransactionFeatures(**feature_dict)
