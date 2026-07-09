"""Fit the Entropy-Compression Law and evaluate go/no-go criteria."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    """Result of fitting log(D_min) = intercept + slope * S1."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n: int
    c_constrained: float  # exp(mean(log(D_min) - S1)), slope=1 fit
    rmse_log: float  # RMSE of residuals in log space (slope=1 model)


def fit_entropy_law(s1_values: np.ndarray, dmin_values: np.ndarray) -> FitResult:
    """Fit log(D_min) = intercept + slope * S1 via OLS.

    Also computes the constrained slope=1 fit: D_min = c * exp(S1).

    Args:
        s1_values: array of von Neumann entropies.
        dmin_values: array of D_min values (integers).

    Returns:
        FitResult with regression statistics.

    Raises:
        ValueError: if fewer than 3 valid data points.
    """
    mask = dmin_values > 0
    s1 = s1_values[mask]
    dmin = dmin_values[mask].astype(float)

    if len(s1) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(s1)}")

    log_dmin = np.log(dmin)

    result = stats.linregress(s1, log_dmin)

    # Constrained slope=1 fit
    c_constrained = np.exp(np.mean(log_dmin - s1))

    # RMSE of slope=1 model
    predicted = s1 + np.log(c_constrained)
    residuals = log_dmin - predicted
    rmse_log = float(np.sqrt(np.mean(residuals ** 2)))

    return FitResult(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue ** 2),
        p_value=float(result.pvalue),
        n=len(s1),
        c_constrained=float(c_constrained),
        rmse_log=rmse_log,
    )


def evaluate_go_nogo(fit_results: dict) -> str:
    """Evaluate go/no-go based on fit results across epsilons.

    Args:
        fit_results: dict mapping epsilon label to FitResult.

    Returns:
        "GO", "MARGINAL", or "PIVOT" with justification.
    """
    r2_values = []
    slopes = []
    for label, fit in fit_results.items():
        r2_values.append(fit.r_squared)
        slopes.append(fit.slope)

    go_count = sum(1 for r2 in r2_values if r2 > 0.85)
    marginal_count = sum(1 for r2 in r2_values if 0.70 <= r2 <= 0.85)
    slope_ok = all(0.3 < s < 2.0 for s in slopes)

    if go_count >= 2 and slope_ok:
        best_r2 = max(r2_values)
        return f"GO (R2={best_r2:.3f}, {go_count}/{len(r2_values)} epsilons above 0.85)"
    elif go_count >= 1 or marginal_count >= 2:
        best_r2 = max(r2_values)
        return f"MARGINAL (R2={best_r2:.3f}, needs more models to confirm)"
    else:
        best_r2 = max(r2_values) if r2_values else 0
        return f"PIVOT (best R2={best_r2:.3f}, law does not hold at this scale)"
