from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from api.schemas import TransactionFeatures
from models.baseline import score_anomaly


@lru_cache(maxsize=1)
def load_bundle(bundle_path: str) -> dict:
    """Loads the joblib artifact `models/package_artifact.py` produces
    (IsolationForest + tuned LightGBM + capacity threshold + feature-column
    order). Cached so the file is only deserialized once per process, not on
    every request.
    """
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `python -m models.package_artifact` first."
        )
    return joblib.load(path)


def score_features(bundle: dict, features: TransactionFeatures) -> tuple[float, bool]:
    """Scores one transaction's already-engineered features the same way the
    training/evaluation pipeline does: the compared
    against the bundle's capacity threshold (see models/package_artifact.py
    and models/train_lightgbm.py, which build the `i IsolationForest score becomes an
    extra input feature, then LightGBM's flagging probability isf_anomaly_score` feature
    identically).
    """
    row = pd.DataFrame([features.model_dump()])[bundle["feature_columns"]]
    if_score = score_anomaly(bundle["isolation_forest"], row)[0]
    lgbm_row = row.copy()
    lgbm_row["if_anomaly_score"] = if_score
    proba = bundle["lightgbm_model"].predict_proba(lgbm_row[bundle["lgbm_feature_columns"]])[0, 1]
    is_flagged = bool(proba >= bundle["capacity_threshold"])
    return float(proba), is_flagged
