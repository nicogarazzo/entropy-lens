"""
Core spectral computations: entropy, D_min, participation ratio, Hill estimator.

All functions operate on 1D numpy arrays of singular values (not squared).
Logarithms are natural (nats). Input need not be normalized.
"""

import numpy as np

_EPSILON_SV = 1e-15


def _to_probabilities(sv: np.ndarray) -> np.ndarray:
    """Convert singular values to Schmidt probabilities p_i = s_i^2 / sum(s_j^2)."""
    sv = sv[sv > _EPSILON_SV]
    if len(sv) == 0:
        raise ValueError("All singular values are below noise threshold")
    p = sv ** 2
    total = p.sum()
    if total == 0:
        raise ValueError("Sum of squared singular values is zero")
    return p / total


def compute_s1(sv: np.ndarray) -> float:
    """Von Neumann entropy S1 = -sum(p_i * log(p_i)) in nats.

    Args:
        sv: 1D array of singular values (not squared).

    Returns:
        Entropy in nats, >= 0.
    """
    p = _to_probabilities(sv)
    p_safe = np.clip(p, 1e-300, None)
    entropy = -np.sum(p * np.log(p_safe))
    return max(float(entropy), 0.0)


def compute_s2(sv: np.ndarray) -> float:
    """Renyi-2 entropy S2 = -log(sum(p_i^2)) in nats.

    Args:
        sv: 1D array of singular values (not squared).

    Returns:
        Renyi-2 entropy in nats, >= 0.
    """
    p = _to_probabilities(sv)
    purity = np.sum(p ** 2)
    if purity <= 0:
        return 0.0
    entropy = -np.log(purity)
    return max(float(entropy), 0.0)


def compute_dmin(sv: np.ndarray, epsilon: float) -> int:
    """Smallest rank D such that ||W - W_D||_F / ||W||_F <= epsilon.

    Uses Eckart-Young: cumulative energy of top-D singular values
    must reach (1 - epsilon^2) * total energy.

    Args:
        sv: 1D array of singular values, descending order.
        epsilon: relative Frobenius error threshold.

    Returns:
        Minimum rank D (1-indexed).
    """
    if epsilon >= 1.0:
        return 1
    sv2 = sv ** 2
    total = sv2.sum()
    if total == 0:
        return 1
    target = (1.0 - epsilon ** 2) * total
    cumulative = 0.0
    for d, s2 in enumerate(sv2, start=1):
        cumulative += s2
        if cumulative >= target:
            return d
    return len(sv)


def compute_participation_ratio(sv: np.ndarray) -> float:
    """Participation ratio PR = exp(S2) = effective number of active singular values.

    Args:
        sv: 1D array of singular values.

    Returns:
        PR >= 1.0.
    """
    return float(np.exp(compute_s2(sv)))


def compute_alpha_hill(sv: np.ndarray, k_min: int = 10) -> float:
    """Power-law exponent via Hill estimator on singular values.

    From AlphaPruning (2024). Used for comparison with S1 as predictor.

    Args:
        sv: 1D array of singular values, descending order.
        k_min: number of top singular values for the estimator.

    Returns:
        Estimated power-law alpha. Returns NaN if not enough values.
    """
    sv_sorted = np.sort(sv)[::-1]
    if len(sv_sorted) < k_min + 1:
        return float("nan")
    log_sv = np.log(sv_sorted[:k_min])
    log_sv_k = np.log(sv_sorted[k_min - 1])
    denom = np.sum(log_sv - log_sv_k)
    if denom == 0:
        return float("nan")
    alpha = 1.0 + k_min / denom
    return float(alpha)
