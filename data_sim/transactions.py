from __future__ import annotations

import numpy as np
import pandas as pd

from data_sim.config import (
    CHANNEL_DEBIT_PROB,
    N_COUNTERPARTIES,
    REGULAR_COUNTERPARTIES_RANGE,
    REGULAR_COUNTERPARTY_USE_PROB,
    SEGMENT_CHANNEL_PROBS,
    SimConfig,
    sample_country,
)

TRANSACTION_COLUMNS = [
    "customer_id",
    "timestamp",
    "amount",
    "direction",
    "channel",
    "counterparty_id",
    "counterparty_country",
    "is_cross_border",
    "is_anomalous",
    "typology",
]


def _build_counterparty_pool(rng: np.random.Generator) -> pd.DataFrame:
    counterparty_id = np.array([f"CPTY{i:06d}" for i in range(N_COUNTERPARTIES)])
    country = sample_country(rng, N_COUNTERPARTIES)
    return pd.DataFrame({"counterparty_id": counterparty_id, "counterparty_country": country})


def _build_regular_counterparty_matrix(
    home_country: np.ndarray, pool: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Each customer's regular payees skew toward their own country (~80%) with
    a minority genuinely cross-border - real payee books look like this, and it's
    what makes an out-of-pattern cross-border payment a meaningful signal later."""
    lo, hi = REGULAR_COUNTERPARTIES_RANGE
    k_max = hi
    n_customers = len(home_country)
    pool_ids = pool["counterparty_id"].to_numpy()
    pool_country = pool["counterparty_country"].to_numpy()
    matrix = np.empty((n_customers, k_max), dtype=object)
    k_per_customer = rng.integers(lo, hi + 1, size=n_customers)

    ids_by_country: dict[str, np.ndarray] = {
        c: pool_ids[pool_country == c] for c in np.unique(pool_country)
    }

    for i in range(n_customers):
        k = int(k_per_customer[i])
        n_domestic = min(int(round(k * 0.8)), len(ids_by_country.get(home_country[i], [])))
        chosen = []
        if n_domestic > 0:
            chosen.append(
                rng.choice(ids_by_country[home_country[i]], size=n_domestic, replace=False)
            )
        n_remaining = k - n_domestic
        if n_remaining > 0:
            chosen.append(rng.choice(pool_ids, size=n_remaining, replace=False))
        combined = np.concatenate(chosen) if chosen else rng.choice(pool_ids, size=k)
        matrix[i] = np.resize(combined, k_max)  # cycles to pad shorter rows
    return matrix


def _sample_channels(segment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    channel = np.empty(len(segment), dtype=object)
    for seg, probs in SEGMENT_CHANNEL_PROBS.items():
        mask = segment == seg
        n = int(mask.sum())
        if n == 0:
            continue
        channel[mask] = rng.choice(list(probs.keys()), size=n, p=list(probs.values()))
    return channel


def _sample_direction(channel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    debit_prob = np.array([CHANNEL_DEBIT_PROB[c] for c in channel])
    return np.where(rng.random(len(channel)) < debit_prob, "debit", "credit")


def generate_base_transactions(
    customers: pd.DataFrame, config: SimConfig, rng: np.random.Generator
) -> pd.DataFrame:
    n_customers = len(customers)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    relative_rate = customers["relative_monthly_txn_rate"].to_numpy()
    lam = relative_rate * config.target_n_transactions / relative_rate.sum()
    counts = np.maximum(rng.poisson(lam), 1)

    cust_idx = np.repeat(np.arange(n_customers), counts)
    total_n = len(cust_idx)

    total_days = (end - start).days
    day_offset = rng.integers(0, total_days, size=total_n)
    hour = np.clip(rng.normal(loc=13.0, scale=4.0, size=total_n), 0, 23.99)
    timestamp = start + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(hour, unit="h")

    amount_mu = customers["amount_mu"].to_numpy()[cust_idx]
    amount_sigma = customers["amount_sigma"].to_numpy()[cust_idx]
    amount = rng.lognormal(mean=amount_mu, sigma=amount_sigma)

    segment = customers["segment"].to_numpy()[cust_idx]
    channel = _sample_channels(segment, rng)
    direction = _sample_direction(channel, rng)

    customer_home_country = customers["home_country"].to_numpy()

    pool = _build_counterparty_pool(rng)
    pool_ids = pool["counterparty_id"].to_numpy()
    regular_matrix = _build_regular_counterparty_matrix(customer_home_country, pool, rng)

    use_regular = rng.random(total_n) < REGULAR_COUNTERPARTY_USE_PROB
    counterparty_id = pool_ids[rng.integers(0, len(pool_ids), size=total_n)]
    reg_cols = rng.integers(0, regular_matrix.shape[1], size=total_n)
    reg_pick = regular_matrix[cust_idx, reg_cols]
    counterparty_id = np.where(use_regular, reg_pick, counterparty_id)

    country_by_id = dict(zip(pool["counterparty_id"], pool["counterparty_country"], strict=True))
    counterparty_country = pd.Series(counterparty_id).map(country_by_id).to_numpy()

    customer_id = customers["customer_id"].to_numpy()[cust_idx]
    home_country = customer_home_country[cust_idx]
    is_cross_border = counterparty_country != home_country

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "timestamp": timestamp,
            "amount": np.round(amount, 2),
            "direction": direction,
            "channel": channel,
            "counterparty_id": counterparty_id,
            "counterparty_country": counterparty_country,
            "is_cross_border": is_cross_border,
            "is_anomalous": False,
            "typology": None,
        }
    )
    return df[TRANSACTION_COLUMNS].sort_values("timestamp").reset_index(drop=True)
