from __future__ import annotations

import numpy as np
import pandas as pd

from data_sim.config import HIGH_RISK_COUNTRIES, N_COUNTERPARTIES, SEGMENT_AMOUNT_PARAMS, SimConfig
from data_sim.transactions import TRANSACTION_COLUMNS


def _empty_like() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def _make_rows(
    customer_id: np.ndarray,
    timestamp,
    amount: np.ndarray,
    direction: np.ndarray,
    channel: np.ndarray,
    counterparty_id: np.ndarray,
    counterparty_country: np.ndarray,
    home_country: np.ndarray,
    typology: str,
) -> pd.DataFrame:
    is_cross_border = counterparty_country != home_country
    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "timestamp": pd.to_datetime(timestamp),
            "amount": np.round(np.asarray(amount, dtype=float), 2),
            "direction": direction,
            "channel": channel,
            "counterparty_id": counterparty_id,
            "counterparty_country": counterparty_country,
            "is_cross_border": is_cross_border,
            "is_anomalous": True,
            "typology": typology,
        }
    )[TRANSACTION_COLUMNS]


def _sample_customers(customers: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    n = min(max(n, 1), len(customers))
    seed = int(rng.integers(0, 2**31 - 1))
    return customers.sample(n=n, random_state=seed)


def _random_counterparty_ids(rng: np.random.Generator, k: int) -> np.ndarray:
    idx = rng.integers(0, N_COUNTERPARTIES, size=k)
    return np.array([f"CPTY{i:06d}" for i in idx])


def inject_structuring(
    customers: pd.DataFrame, rng: np.random.Generator, config: SimConfig, n_target: int
) -> pd.DataFrame:
    """Clusters of transactions just under the reporting threshold, in a short window."""
    threshold = config.reporting_threshold
    picked = _sample_customers(customers, n_target // 4, rng)
    start, end = pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
    rows = []
    for _, cust in picked.iterrows():
        k = int(rng.choice([3, 4, 5]))
        base_ts = start + (end - start) * float(rng.random())
        timestamps = base_ts + pd.to_timedelta(rng.integers(0, 48, size=k), unit="h")
        amounts = rng.uniform(threshold * 0.85, threshold * 0.995, size=k)
        rows.append(
            _make_rows(
                customer_id=np.full(k, cust["customer_id"]),
                timestamp=timestamps,
                amount=amounts,
                direction=np.full(k, "debit"),
                channel=np.full(k, "wire"),
                counterparty_id=np.full(k, f"CPTY_STRUCT_{cust['customer_id']}"),
                counterparty_country=np.full(k, cust["home_country"]),
                home_country=np.full(k, cust["home_country"]),
                typology="structuring",
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else _empty_like()


def inject_layering(
    customers: pd.DataFrame, rng: np.random.Generator, config: SimConfig, n_target: int
) -> pd.DataFrame:
    """A large inbound credit followed within hours by several outbound debits
    that together move most of the funds back out - rapid layering."""
    picked = _sample_customers(customers, n_target // 6, rng)
    start, end = pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
    rows = []
    for _, cust in picked.iterrows():
        inflow = rng.lognormal(mean=cust["amount_mu"] + 1.2, sigma=0.4)
        base_ts = start + (end - start) * float(rng.random())
        n_out = int(rng.integers(2, 5))
        out_fractions = rng.dirichlet(np.ones(n_out)) * rng.uniform(0.75, 0.98)
        out_amounts = inflow * out_fractions
        out_offsets = np.sort(rng.uniform(1, 36, size=n_out))

        k = n_out + 1
        timestamps = [base_ts] + [base_ts + pd.Timedelta(hours=float(h)) for h in out_offsets]
        amounts = np.concatenate([[inflow], out_amounts])
        direction = np.array(["credit"] + ["debit"] * n_out)

        rows.append(
            _make_rows(
                customer_id=np.full(k, cust["customer_id"]),
                timestamp=timestamps,
                amount=amounts,
                direction=direction,
                channel=np.full(k, "wire"),
                counterparty_id=_random_counterparty_ids(rng, k),
                counterparty_country=np.full(k, cust["home_country"]),
                home_country=np.full(k, cust["home_country"]),
                typology="layering",
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else _empty_like()


def inject_round_amount(
    customers: pd.DataFrame, rng: np.random.Generator, config: SimConfig, n_target: int
) -> pd.DataFrame:
    """A burst of suspiciously round-numbered transactions for a customer."""
    picked = _sample_customers(customers, n_target // 5, rng)
    start, end = pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
    round_steps = np.array([500, 1_000, 2_000, 5_000])
    rows = []
    for _, cust in picked.iterrows():
        k = int(rng.integers(3, 6))
        timestamps = start + (end - start) * rng.random(size=k)
        amounts = rng.choice(round_steps, size=k) * rng.integers(1, 6, size=k)
        rows.append(
            _make_rows(
                customer_id=np.full(k, cust["customer_id"]),
                timestamp=timestamps,
                amount=amounts,
                direction=np.full(k, "debit"),
                channel=rng.choice(["online", "branch", "wire"], size=k),
                counterparty_id=_random_counterparty_ids(rng, k),
                counterparty_country=np.full(k, cust["home_country"]),
                home_country=np.full(k, cust["home_country"]),
                typology="round_amount",
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else _empty_like()


def inject_velocity_spike(
    customers: pd.DataFrame, rng: np.random.Generator, config: SimConfig, n_target: int
) -> pd.DataFrame:
    """A burst of extra, otherwise-typical-looking transactions in a 72h window -
    volume far outside the customer's own baseline."""
    picked = _sample_customers(customers, n_target // 8, rng)
    start, end = pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
    k = 8
    rows = []
    for _, cust in picked.iterrows():
        window_start = start + (end - start - pd.Timedelta(days=3)) * float(rng.random())
        timestamps = window_start + pd.to_timedelta(rng.uniform(0, 72, size=k), unit="h")
        amounts = rng.lognormal(mean=cust["amount_mu"], sigma=cust["amount_sigma"] * 0.5, size=k)
        rows.append(
            _make_rows(
                customer_id=np.full(k, cust["customer_id"]),
                timestamp=timestamps,
                amount=amounts,
                direction=np.full(k, "debit"),
                channel=rng.choice(["online", "card", "wire"], size=k),
                counterparty_id=_random_counterparty_ids(rng, k),
                counterparty_country=np.full(k, cust["home_country"]),
                home_country=np.full(k, cust["home_country"]),
                typology="velocity_spike",
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else _empty_like()


def inject_peer_deviation(
    customers: pd.DataFrame, rng: np.random.Generator, config: SimConfig, n_target: int
) -> pd.DataFrame:
    """Transactions sized like a materially higher-scale segment than the
    customer's own declared segment - behaviour inconsistent with peer group."""
    picked = _sample_customers(customers, n_target // 3, rng)
    start, end = pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
    rows = []
    for _, cust in picked.iterrows():
        k = int(rng.integers(1, 3))
        timestamps = start + (end - start) * rng.random(size=k)
        off_segment = "private_banking" if cust["segment"] != "private_banking" else "sme"
        mu, sigma = SEGMENT_AMOUNT_PARAMS[off_segment]
        amounts = rng.lognormal(mean=mu, sigma=sigma * 0.6, size=k)
        rows.append(
            _make_rows(
                customer_id=np.full(k, cust["customer_id"]),
                timestamp=timestamps,
                amount=amounts,
                direction=np.full(k, "debit"),
                channel=rng.choice(["wire", "online"], size=k),
                counterparty_id=_random_counterparty_ids(rng, k),
                counterparty_country=np.full(k, cust["home_country"]),
                home_country=np.full(k, cust["home_country"]),
                typology="peer_deviation",
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else _empty_like()


def inject_geographic_risk(
    customers: pd.DataFrame, rng: np.random.Generator, config: SimConfig, n_target: int
) -> pd.DataFrame:
    """Transactions to/from a jurisdiction on the simulation's higher-risk list."""
    picked = _sample_customers(customers, n_target // 2, rng)
    start, end = pd.Timestamp(config.start_date), pd.Timestamp(config.end_date)
    rows = []
    for _, cust in picked.iterrows():
        k = int(rng.integers(1, 3))
        timestamps = start + (end - start) * rng.random(size=k)
        amounts = rng.lognormal(mean=cust["amount_mu"] + 0.5, sigma=cust["amount_sigma"], size=k)
        cpty_country = rng.choice(HIGH_RISK_COUNTRIES, size=k)
        cpty_id = np.array([f"CPTY_HR_{c}_{int(rng.integers(0, 99_999))}" for c in cpty_country])
        rows.append(
            _make_rows(
                customer_id=np.full(k, cust["customer_id"]),
                timestamp=timestamps,
                amount=amounts,
                direction=rng.choice(["debit", "credit"], size=k),
                channel=np.full(k, "wire"),
                counterparty_id=cpty_id,
                counterparty_country=cpty_country,
                home_country=np.full(k, cust["home_country"]),
                typology="geographic_risk",
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else _empty_like()
