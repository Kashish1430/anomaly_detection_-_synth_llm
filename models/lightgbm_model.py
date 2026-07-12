from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    # anomalies are ~1.5% of the data - without this the model can get high
    # accuracy by just predicting "normal" for everything
    "class_weight": "balanced",
    "verbosity": -1,
}


def fit_lightgbm(
    X: pd.DataFrame, y: np.ndarray, params: dict | None = None, random_state: int = 42
) -> lgb.LGBMClassifier:
    merged_params = {**DEFAULT_PARAMS, **(params or {}), "random_state": random_state}
    model = lgb.LGBMClassifier(**merged_params)
    model.fit(X, y)
    return model


def predict_proba_anomaly(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
