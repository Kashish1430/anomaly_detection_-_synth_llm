from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEGMENTS = ("retail", "sme", "private_banking")
SEGMENT_WEIGHTS = (0.80, 0.15, 0.05)

# Relative monthly transaction-frequency weight per segment (business accounts
# transact far more often than personal ones). Scaled at generation time so the
# *realised* total lands near `target_n_transactions` rather than being hardcoded.
SEGMENT_RELATIVE_FREQUENCY = {"retail": 1.0, "sme": 6.0, "private_banking": 2.5}

# lognormal(mu, sigma) parameters for a single transaction amount, by segment
SEGMENT_AMOUNT_PARAMS = {
    "retail": (4.10, 0.90),  # median ~ GBP 60
    "sme": (6.68, 1.10),  # median ~ GBP 800
    "private_banking": (8.00, 1.00),  # median ~ GBP 3,000
}

CHANNELS = ("online", "card", "atm", "branch", "wire")

SEGMENT_CHANNEL_PROBS = {
    "retail": {"online": 0.45, "card": 0.30, "atm": 0.15, "branch": 0.07, "wire": 0.03},
    "sme": {"online": 0.35, "card": 0.10, "atm": 0.05, "branch": 0.20, "wire": 0.30},
    "private_banking": {"online": 0.25, "card": 0.05, "atm": 0.00, "branch": 0.25, "wire": 0.45},
}

CHANNEL_DEBIT_PROB = {"card": 0.98, "atm": 0.95, "online": 0.55, "branch": 0.50, "wire": 0.55}

DOMESTIC_COUNTRY = "GB"
OTHER_COUNTRIES = ("US", "FR", "DE", "IE", "ES", "IN", "AE", "SG", "NL", "IT", "CA", "AU")
# Illustrative "higher-risk jurisdiction" list for the simulation only -
# not a real FATF/HMT designation. Used purely to give the geographic_risk
# typology and cross-border features something to detect.
HIGH_RISK_COUNTRIES = ("KY", "PA", "MT", "CY")
DOMESTIC_WEIGHT = 0.82
HIGH_RISK_WEIGHT = 0.03  # share of the *non-domestic* pool that is high-risk

RISK_RATINGS = ("low", "medium", "high")
RISK_RATING_WEIGHTS = (0.75, 0.20, 0.05)

N_COUNTERPARTIES = 6_000
REGULAR_COUNTERPARTIES_RANGE = (3, 10)  # inclusive
REGULAR_COUNTERPARTY_USE_PROB = 0.85


def sample_country(rng: np.random.Generator, n: int) -> np.ndarray:
    is_domestic = rng.random(n) < DOMESTIC_WEIGHT
    is_high_risk = (~is_domestic) & (rng.random(n) < HIGH_RISK_WEIGHT)
    countries = np.full(n, DOMESTIC_COUNTRY, dtype=object)
    other_mask = (~is_domestic) & (~is_high_risk)
    countries[other_mask] = rng.choice(OTHER_COUNTRIES, size=int(other_mask.sum()))
    countries[is_high_risk] = rng.choice(HIGH_RISK_COUNTRIES, size=int(is_high_risk.sum()))
    return countries


@dataclass
class SimConfig:
    seed: int = 42
    n_customers: int = 12_000
    start_date: str = "2024-01-01"
    end_date: str = "2025-06-30"
    target_n_transactions: int = 1_200_000
    # share of *base* transactions matched by typology-injected anomalous rows,
    # split across the six typologies below
    anomaly_rate: float = 0.02
    reporting_threshold: float = 10_000.0
    output_dir: str = "data/simulated"

    typology_share: dict = field(
        default_factory=lambda: {
            "structuring": 1 / 6,
            "layering": 1 / 6,
            "round_amount": 1 / 6,
            "velocity_spike": 1 / 6,
            "peer_deviation": 1 / 6,
            "geographic_risk": 1 / 6,
        }
    )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
