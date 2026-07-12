from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluation.splits import expanding_window_splits
from models.lightgbm_model import fit_lightgbm, predict_proba_anomaly

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_score(
    X: pd.DataFrame, y: np.ndarray, timestamps: pd.Series, params: dict, n_folds: int
) -> tuple[float, float]:
    """Mean and std of PR-AUC across expanding-window folds - the mean is what
    Optuna optimizes, the std is what tells us whether that mean is trustworthy
    (PLAN.md §07's "validating stability across time-based cross-validation
    folds"), not just a single lucky split.
    """
    fold_scores = []
    for train_idx, test_idx in expanding_window_splits(timestamps, n_folds=n_folds):
        model = fit_lightgbm(X.iloc[train_idx], y[train_idx], params=params)
        proba = predict_proba_anomaly(model, X.iloc[test_idx])
        fold_scores.append(average_precision_score(y[test_idx], proba))
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def tune_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    n_trials: int = 30,
    n_folds: int = 4,
    seed: int = 42,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "class_weight": "balanced",
            "verbosity": -1,
        }
        mean_score, std_score = _cv_score(X, y, timestamps, params, n_folds=n_folds)
        trial.set_user_attr("cv_std", std_score)
        return mean_score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
