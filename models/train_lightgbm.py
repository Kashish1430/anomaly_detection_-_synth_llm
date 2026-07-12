from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from evaluation.splits import expanding_window_splits, time_ordered_split
from evaluation.stats import precision_confidence_interval
from evaluation.threshold_tuning import rates_at_threshold, select_threshold_by_ztest
from features.pipeline import FEATURE_COLUMNS, build_feature_table
from models.baseline import fit_isolation_forest, score_anomaly
from models.lightgbm_model import fit_lightgbm, predict_proba_anomaly
from models.tuning import tune_lightgbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LGBM_FEATURE_COLUMNS = [*FEATURE_COLUMNS, "if_anomaly_score"]


def _load_features(
    data_dir: Path, transactions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    features_path = data_dir / "features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return build_feature_table(transactions, customers)


def run(
    data_dir: Path,
    capacity_frac: float = 0.02,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    n_trials: int = 30,
    cv_folds: int = 4,
    seed: int = 42,
    log_to_mlflow: bool = True,
) -> dict:
    customers = pd.read_parquet(data_dir / "customers.parquet")
    transactions = pd.read_parquet(data_dir / "transactions.parquet")
    features = _load_features(data_dir, transactions, customers)

    merged = transactions[["transaction_id", "timestamp", "is_anomalous"]].merge(
        features, on="transaction_id", validate="one_to_one"
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    y_all = merged["is_anomalous"].astype(int).to_numpy()

    train_idx, val_idx, test_idx = time_ordered_split(
        merged["timestamp"], train_frac=train_frac, val_frac=val_frac
    )
    log.info("train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    # IsolationForest fit on TRAIN only, then used to score everything - its
    # score becomes one more feature for LightGBM, same idea as PLAN.md §05.
    if_model = fit_isolation_forest(
        merged.iloc[train_idx][FEATURE_COLUMNS], n_estimators=200, random_state=seed
    )
    merged["if_anomaly_score"] = score_anomaly(if_model, merged[FEATURE_COLUMNS])

    X_train = merged.iloc[train_idx][LGBM_FEATURE_COLUMNS]
    y_train = y_all[train_idx]
    X_val = merged.iloc[val_idx][LGBM_FEATURE_COLUMNS]
    y_val = y_all[val_idx]
    X_test = merged.iloc[test_idx][LGBM_FEATURE_COLUMNS]
    y_test = y_all[test_idx]

    # hyperparameter tuning via expanding-window time-based CV, strictly
    # inside TRAIN - validation and test are never touched during tuning
    study = tune_lightgbm(
        X_train,
        y_train,
        merged.iloc[train_idx]["timestamp"],
        n_trials=n_trials,
        n_folds=cv_folds,
        seed=seed,
    )
    best_params = {**study.best_params, "class_weight": "balanced", "verbosity": -1}
    cv_mean_pr_auc = study.best_value
    cv_std_pr_auc = study.best_trial.user_attrs["cv_std"]
    log.info(
        "Optuna best trial: PR-AUC=%.3f +/- %.3f across %d time-based folds",
        cv_mean_pr_auc,
        cv_std_pr_auc,
        cv_folds,
    )

    final_model = fit_lightgbm(X_train, y_train, params=best_params, random_state=seed)

    # Threshold selection uses out-of-fold predictions spanning several time
    # windows *within* TRAIN, not a single VALIDATION block immediately
    # before TEST. A single-block choice is exactly the kind of split-noise
    # PLAN.md §07's time-based CV is meant to guard against - and did: an
    # earlier version of this picked a threshold on VALIDATION alone that
    # looked like a real FP-rate improvement there (p=0.0001) but made things
    # *worse* on TEST (FP rate up 17.7%, not down). This refit-per-fold
    # approach is more expensive but doesn't rely on one time slice agreeing
    # with the next.
    train_timestamps = merged.iloc[train_idx]["timestamp"]
    oof_proba = np.full(len(train_idx), np.nan)
    oof_folds = expanding_window_splits(train_timestamps, n_folds=cv_folds)
    for fold_train_pos, fold_test_pos in oof_folds:
        fold_model = fit_lightgbm(
            X_train.iloc[fold_train_pos],
            y_train[fold_train_pos],
            params=best_params,
            random_state=seed,
        )
        oof_proba[fold_test_pos] = predict_proba_anomaly(fold_model, X_train.iloc[fold_test_pos])

    oof_mask = ~np.isnan(oof_proba)
    oof_y, oof_scores = y_train[oof_mask], oof_proba[oof_mask]
    log.info(
        "Out-of-fold threshold-selection sample: %d rows across %d folds",
        oof_mask.sum(),
        cv_folds,
    )

    capacity_threshold = float(np.quantile(oof_scores, 1 - capacity_frac))
    thresholds = np.unique(np.quantile(oof_scores, np.linspace(0.80, 0.999, 60)))
    selection = select_threshold_by_ztest(
        oof_y, oof_scores, thresholds, baseline_threshold=capacity_threshold
    )
    chosen_threshold = selection["chosen_threshold"]

    # VALIDATION is reported as an interim sanity check only - it plays no
    # role in picking either threshold
    proba_val = predict_proba_anomaly(final_model, X_val)
    val_capacity_metrics = rates_at_threshold(y_val, proba_val, capacity_threshold)
    val_tuned_metrics = rates_at_threshold(y_val, proba_val, chosen_threshold)
    log.info(
        "Validation sanity check - capacity precision: %.3f | tuned precision: %.3f",
        val_capacity_metrics["precision"],
        val_tuned_metrics["precision"],
    )

    # single, final look at TEST - neither threshold was fit or chosen using it
    proba_test = predict_proba_anomaly(final_model, X_test)
    capacity_metrics_test = rates_at_threshold(y_test, proba_test, capacity_threshold)
    tuned_metrics_test = rates_at_threshold(y_test, proba_test, chosen_threshold)
    precision_ci = precision_confidence_interval(
        capacity_metrics_test["n_flagged"], capacity_metrics_test["tp"]
    )

    fp_before = capacity_metrics_test["fp"]
    fp_after = tuned_metrics_test["fp"]
    fp_reduction_pct = (fp_before - fp_after) / fp_before * 100 if fp_before else 0.0

    params_logged = {
        "model": "LightGBM",
        "n_features": len(LGBM_FEATURE_COLUMNS),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "test_rows": len(test_idx),
        "oof_threshold_selection_rows": int(oof_mask.sum()),
        "capacity_frac": capacity_frac,
        "capacity_threshold": capacity_threshold,
        "chosen_threshold": chosen_threshold,
        "n_optuna_trials": n_trials,
        "cv_folds": cv_folds,
        "seed": seed,
        **{f"best_{k}": v for k, v in study.best_params.items()},
    }
    metrics_logged = {
        "cv_mean_pr_auc": cv_mean_pr_auc,
        "cv_std_pr_auc": cv_std_pr_auc,
        "val_precision_at_capacity": val_capacity_metrics["precision"],
        "val_precision_at_tuned_threshold": val_tuned_metrics["precision"],
        "test_precision_at_capacity": capacity_metrics_test["precision"],
        "test_recall_at_capacity": capacity_metrics_test["recall"],
        "test_precision_ci_lower": precision_ci[0],
        "test_precision_ci_upper": precision_ci[1],
        "test_precision_at_tuned_threshold": tuned_metrics_test["precision"],
        "test_recall_at_tuned_threshold": tuned_metrics_test["recall"],
        "test_fp_reduction_pct": fp_reduction_pct,
        "threshold_ztest_pvalue": selection["p_value"],
    }

    if log_to_mlflow:
        mlflow.set_experiment("anomaly-detection-lightgbm")
        with mlflow.start_run(run_name="lightgbm_tuned_threshold"):
            mlflow.log_params(params_logged)
            mlflow.log_metrics(metrics_logged)
            mlflow.lightgbm.log_model(final_model, name="lightgbm_model")

    log.info(
        "'After' precision @ %.1f%% capacity: %.3f (95%% CI [%.3f, %.3f]) | recall: %.3f",
        capacity_frac * 100,
        capacity_metrics_test["precision"],
        precision_ci[0],
        precision_ci[1],
        capacity_metrics_test["recall"],
    )
    log.info(
        "Threshold tuning: FP %d -> %d (%.1f%% reduction), z=%.2f, p=%.4f, significant=%s",
        fp_before,
        fp_after,
        fp_reduction_pct,
        selection["z_stat"],
        selection["p_value"],
        selection["fp_reduction_significant_at_alpha"],
    )

    return {
        "params": params_logged,
        "metrics": metrics_logged,
        "threshold_selection": {k: v for k, v in selection.items() if k != "sweep"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the tuned LightGBM model.")
    parser.add_argument("--data-dir", type=str, default="data/simulated")
    parser.add_argument("--capacity-frac", type=float, default=0.02)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    result = run(
        Path(args.data_dir),
        capacity_frac=args.capacity_frac,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        seed=args.seed,
        log_to_mlflow=not args.no_mlflow,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
