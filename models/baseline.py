from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) == 0:
        return X
    return X.astype({c: "int8" for c in bool_cols})


def fit_isolation_forest(
    X: pd.DataFrame,
    n_estimators: int = 200,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> IsolationForest:
    """Fits an unsupervised IsolationForest. No labels are used - this is the
    "before" baseline in PLAN.md §05: an anomaly-scoring layer with zero
    hand-labelled input, evaluated only afterwards against ground truth.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(_to_numeric_matrix(X))
    return model


def score_anomaly(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous (the reverse of sklearn's own score_samples
    convention, where higher means more normal) - kept this way so every
    downstream consumer of `anomaly_score` can assume "bigger is worse".
    """
    return -model.score_samples(_to_numeric_matrix(X))
