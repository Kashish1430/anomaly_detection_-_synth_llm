from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

from data_sim.config import CHANNELS, RISK_RATINGS, SEGMENTS

CustomerSchema = DataFrameSchema(
    {
        "customer_id": Column(str, unique=True),
        "segment": Column(str, Check.isin(SEGMENTS)),
        "home_country": Column(str),
        "declared_risk_rating": Column(str, Check.isin(RISK_RATINGS)),
        "signup_date": Column("datetime64[ns]"),
        "relative_monthly_txn_rate": Column(float, Check.gt(0)),
        "amount_mu": Column(float),
        "amount_sigma": Column(float, Check.gt(0)),
        "peer_group": Column(str),
    },
    strict=True,
    coerce=True,
)

TransactionSchema = DataFrameSchema(
    {
        "transaction_id": Column(str, unique=True),
        "customer_id": Column(str),
        "timestamp": Column("datetime64[ns]"),
        "amount": Column(float, Check.gt(0)),
        "direction": Column(str, Check.isin(["debit", "credit"])),
        "channel": Column(str, Check.isin(CHANNELS)),
        "counterparty_id": Column(str),
        "counterparty_country": Column(str),
        "is_cross_border": Column(bool),
        "is_anomalous": Column(bool),
        "typology": Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
)
