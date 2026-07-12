from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

FeatureTableSchema = DataFrameSchema(
    {
        "transaction_id": Column(str, unique=True),
        "velocity_count_1h": Column(float, Check.ge(0)),
        "velocity_sum_1h": Column(float, Check.ge(0)),
        "velocity_count_24h": Column(float, Check.ge(0)),
        "velocity_sum_24h": Column(float, Check.ge(0)),
        "velocity_count_7d": Column(float, Check.ge(0)),
        "velocity_sum_7d": Column(float, Check.ge(0)),
        "velocity_count_30d": Column(float, Check.ge(0)),
        "velocity_sum_30d": Column(float, Check.ge(0)),
        "amount_to_avg_ratio": Column(float, Check.ge(0)),
        "personal_amount_zscore": Column(float),
        "is_new_counterparty": Column(bool),
        "is_channel_switch": Column(bool),
        "is_round_amount": Column(bool),
        "round_amount_count_30d": Column(float, Check.ge(0)),
        "peer_zscore": Column(float),
        "hour_of_day": Column(int, Check.in_range(0, 23)),
        "hour_zscore": Column(float),
        "is_cross_border": Column(bool),
    },
    strict=True,
    coerce=True,
)
