from __future__ import annotations

from statsmodels.stats.proportion import proportion_confint, proportions_ztest


def wilson_confidence_interval(count: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion count / n. Wilson rather than the
    plain normal-approximation interval because the proportions here are often
    close to the [0, 1] boundary with a modest sample size, where the normal
    approximation is known to misbehave.
    """
    if n == 0:
        return (0.0, 0.0)
    lower, upper = proportion_confint(count, n, alpha=alpha, method="wilson")
    return float(lower), float(upper)


def precision_confidence_interval(
    n_flagged: int, n_true_positive: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Wilson score interval for precision = n_true_positive / n_flagged."""
    return wilson_confidence_interval(n_true_positive, n_flagged, alpha=alpha)


def two_proportion_ztest(count1: int, nobs1: int, count2: int, nobs2: int) -> tuple[float, float]:
    """Two-proportion z-test, e.g. comparing false-positive rates between two
    candidate thresholds (PLAN.md §07). Returns (z_stat, p_value); a positive
    z_stat means proportion1 > proportion2.
    """
    stat, p_value = proportions_ztest([count1, count2], [nobs1, nobs2])
    return float(stat), float(p_value)
