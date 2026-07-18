from __future__ import annotations

from datetime import UTC, datetime, timedelta

from api.live_features import compute_live_features
from api.schemas import RawTransactionRequest
from features.peer_deviation import MAD_TO_STD

# The individual feature formulas (velocity windows, amount_to_avg_ratio,
# new-counterparty/channel-switch, round-amount) already have hand-crafted
# coverage in tests/test_features.py, run unchanged here - these tests focus
# on what's new: assembling a customer's history + a raw new transaction into
# the right shape, and the precomputed-lookup peer_zscore path.

NEW_TXN_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _history_row(transaction_id, hours_ago, amount, channel, counterparty_id):
    return {
        "transaction_id": transaction_id,
        "customer_id": "CUST1",
        "timestamp": NEW_TXN_TIME - timedelta(hours=hours_ago),
        "amount": amount,
        "direction": "debit",
        "channel": channel,
        "counterparty_id": counterparty_id,
        "counterparty_country": "GB",
        "is_cross_border": False,
    }


def test_compute_live_features_uses_customer_history():
    history = [
        _history_row(
            "H1", hours_ago=40 * 24, amount=133.5, channel="online", counterparty_id="CPTY1"
        ),
        _history_row(
            "H2", hours_ago=5 * 24, amount=250.75, channel="card", counterparty_id="CPTY2"
        ),
    ]
    new_txn = RawTransactionRequest(
        customer_id="CUST1",
        timestamp=NEW_TXN_TIME,
        amount=10000.0,
        direction="debit",
        channel="branch",
        counterparty_id="CPTY3",
        is_cross_border=False,
    )
    peer_stats = {"peer_group": "retail_GB", "peer_median": 200.0, "peer_mad": 50.0}

    features = compute_live_features(history, peer_stats, new_txn, "LIVE001", "CUST1")

    assert features.is_round_amount is True
    assert features.is_new_counterparty is True  # CPTY3 never seen before for this customer
    assert features.is_channel_switch is True  # prior was "card", this is "branch"
    assert (
        features.velocity_count_7d == 2
    )  # H2 (5d ago) + itself; H1 (40d ago) is outside the window
    assert features.velocity_count_30d == 2  # same - H1 is outside 30D too
    assert features.round_amount_count_30d == 1  # only the new txn itself is round within 30D

    expected_peer_zscore = (10000.0 - 200.0) / (50.0 * MAD_TO_STD)
    assert features.peer_zscore == expected_peer_zscore


def test_compute_live_features_zero_history_new_customer():
    """A genuinely new customer_id with no prior transactions - every
    feature module has a documented "no prior data" fallback (see
    features/utils.py:safe_zscore), this confirms compute_live_features
    actually exercises that path without erroring.
    """
    new_txn = RawTransactionRequest(
        customer_id="CUST-NEW",
        timestamp=NEW_TXN_TIME,
        amount=500.0,
        direction="debit",
        channel="online",
        is_cross_border=False,
    )

    features = compute_live_features([], None, new_txn, "LIVE002", "CUST-NEW")

    assert features.velocity_count_1h == 1
    assert features.velocity_count_30d == 1
    assert features.amount_to_avg_ratio == 1.0  # neutral fill, no prior average
    assert features.personal_amount_zscore == 0.0  # safe_zscore's NaN->0.0 floor
    assert features.is_new_counterparty is True
    assert features.is_channel_switch is False  # no prior transaction to switch from
    assert features.peer_zscore == 0.0  # no peer_stats provided


def test_compute_live_features_peer_zscore_falls_back_to_zero_on_zero_mad():
    new_txn = RawTransactionRequest(
        customer_id="CUST1",
        timestamp=NEW_TXN_TIME,
        amount=500.0,
        direction="debit",
        channel="online",
        is_cross_border=False,
    )
    peer_stats = {"peer_group": "sme_GB", "peer_median": 500.0, "peer_mad": 0.0}

    features = compute_live_features([], peer_stats, new_txn, "LIVE003", "CUST1")

    assert features.peer_zscore == 0.0
