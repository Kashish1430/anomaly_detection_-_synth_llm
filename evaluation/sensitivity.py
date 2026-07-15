from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def perturb_feature(X: pd.DataFrame, feature: str, delta_in_std: float) -> pd.DataFrame:
    """Shifts one feature by `delta_in_std` standard deviations of its own
    empirical distribution, holding every other feature fixed. The unit is
    relative to each feature's own scale, so a +/-1.0 perturbation is
    comparable in "how unusual is this shift" terms across features with very
    different raw ranges (PLAN.md §08 sensitivity analysis).
    """
    perturbed = X.copy()
    std = perturbed[feature].std()
    perturbed[feature] = perturbed[feature] + delta_in_std * std
    return perturbed


def feature_sensitivity(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    feature_columns: list[str],
    delta_in_std: float = 1.0,
) -> pd.DataFrame:
    """One-at-a-time sensitivity: for each feature, shifts it by +/-`delta_in_std`
    standard deviations (holding all other features fixed) and reports how much
    the model's predicted score moves, on average and at the extreme. Features
    the score reacts to most strongly are exactly the ones a data-quality issue
    or an adversarial nudge would move the flagging decision on.
    """
    base_scores = predict_fn(X)
    rows = []
    for feature in feature_columns:
        delta_up = predict_fn(perturb_feature(X, feature, delta_in_std)) - base_scores
        delta_down = predict_fn(perturb_feature(X, feature, -delta_in_std)) - base_scores
        rows.append(
            {
                "feature": feature,
                "mean_abs_delta_up": float(np.mean(np.abs(delta_up))),
                "mean_abs_delta_down": float(np.mean(np.abs(delta_down))),
                "max_abs_delta": float(np.max(np.abs(np.concatenate([delta_up, delta_down])))),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("mean_abs_delta_up", ascending=False).reset_index(drop=True)
    )


def decision_flip_rate(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    threshold: float,
    noise_std_frac: float = 0.01,
    seed: int = 42,
) -> float:
    """Robustness check: adds small Gaussian noise (a fraction of each
    feature's own std - default 1%, simulating routine data-quality noise) to
    every feature simultaneously, and reports what fraction of rows flip their
    flagged/not-flagged decision as a result. A high flip rate means the
    flagging decision is fragile to noise unrelated to genuine anomalous
    behaviour.
    """
    rng = np.random.default_rng(seed)
    noisy = X.copy()
    for col in X.columns:
        std = X[col].std()
        noisy[col] = X[col] + rng.normal(0.0, noise_std_frac * std, size=len(X))

    base_flagged = predict_fn(X) >= threshold
    noisy_flagged = predict_fn(noisy) >= threshold
    return float(np.mean(base_flagged != noisy_flagged))
