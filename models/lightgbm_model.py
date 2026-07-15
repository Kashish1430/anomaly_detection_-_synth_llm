from __future__ import annotations

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
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


def load_tuned_model_and_threshold(run_id: str) -> tuple[lgb.LGBMClassifier, float]:
    """Loads a model + its capacity_threshold back from an MLflow run logged by
    train_lightgbm.py, instead of re-running the 30-trial Optuna search that
    produced it - any script that needs the real tuned model (fairness checks,
    sensitivity analysis, ...) should go through this rather than refitting.
    """
    client = mlflow.MlflowClient()
    mlflow_run = client.get_run(run_id)
    threshold = float(mlflow_run.data.params["capacity_threshold"])
    logged_model = next(
        m
        for m in client.search_logged_models(experiment_ids=[mlflow_run.info.experiment_id])
        if m.source_run_id == run_id
    )
    model = mlflow.lightgbm.load_model(f"models:/{logged_model.model_id}")
    return model, threshold


def predict_shap_contributions(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP feature contributions (log-odds scale) via LightGBM's native
    `pred_contrib` - the same algorithm the `shap` package implements, with no extra
    dependency. Shape (n_rows, n_features + 1); the last column is the base value
    (the model's expected output with no feature information), so each row's
    contributions plus the base value sum to that row's raw-margin prediction.
    """
    return model.predict(X, pred_contrib=True)
