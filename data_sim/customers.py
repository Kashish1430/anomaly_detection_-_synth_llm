from __future__ import annotations

import numpy as np
import pandas as pd

from data_sim.config import (
    RISK_RATING_WEIGHTS,
    RISK_RATINGS,
    SEGMENT_AMOUNT_PARAMS,
    SEGMENT_RELATIVE_FREQUENCY,
    SEGMENT_WEIGHTS,
    SEGMENTS,
    SimConfig,
    sample_country,
)


def generate_customers(config: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_customers
    customer_id = np.array([f"CUST{i:07d}" for i in range(n)])

    segment = rng.choice(SEGMENTS, size=n, p=SEGMENT_WEIGHTS)

    base_freq = np.array([SEGMENT_RELATIVE_FREQUENCY[s] for s in segment])
    freq_jitter = rng.lognormal(mean=0.0, sigma=0.4, size=n)
    relative_monthly_txn_rate = base_freq * freq_jitter

    amount_mu = np.array([SEGMENT_AMOUNT_PARAMS[s][0] for s in segment])
    amount_sigma = np.array([SEGMENT_AMOUNT_PARAMS[s][1] for s in segment])
    # idiosyncratic per-customer offset around the segment's typical amount -
    # this is what makes "peer-group deviation" a learnable signal later on
    amount_mu = amount_mu + rng.normal(loc=0.0, scale=0.25, size=n)

    home_country = sample_country(rng, n)
    risk_rating = rng.choice(RISK_RATINGS, size=n, p=RISK_RATING_WEIGHTS)

    signup_offset_days = rng.integers(0, 365 * 3, size=n)
    signup_date = pd.Timestamp(config.start_date) - pd.to_timedelta(signup_offset_days, unit="D")

    peer_group = np.array([f"{s}_{c}" for s, c in zip(segment, home_country, strict=True)])

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "segment": segment,
            "home_country": home_country,
            "declared_risk_rating": risk_rating,
            "signup_date": signup_date,
            "relative_monthly_txn_rate": relative_monthly_txn_rate,
            "amount_mu": amount_mu,
            "amount_sigma": amount_sigma,
            "peer_group": peer_group,
        }
    )
