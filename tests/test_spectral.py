"""Unit tests for spectral computations."""

import math

import numpy as np
import pytest

from entropy_lens.spectral import (
    compute_alpha_hill,
    compute_dmin,
    compute_participation_ratio,
    compute_s1,
    compute_s2,
)


class TestComputeS1:
    """Von Neumann entropy tests."""

    def test_uniform_spectrum(self):
        """S1 of D equal singular values = log(D)."""
        D = 100
        sv = np.ones(D)
        s1 = compute_s1(sv)
        expected = math.log(D)
        assert abs(s1 - expected) < 1e-10, f"Expected {expected}, got {s1}"

    def test_single_value(self):
        """S1 of a single nonzero singular value = 0 (product state)."""
        sv = np.array([5.0])
        s1 = compute_s1(sv)
        assert abs(s1) < 1e-10, f"Expected 0, got {s1}"

    def test_two_equal_values(self):
        """S1 of two equal singular values = log(2)."""
        sv = np.array([3.0, 3.0])
        s1 = compute_s1(sv)
        expected = math.log(2)
        assert abs(s1 - expected) < 1e-10

    def test_nonnegative(self):
        """S1 is always >= 0."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            sv = rng.random(50) * 10
            assert compute_s1(sv) >= 0

    def test_scale_invariant(self):
        """S1 is scale-invariant: S1(c*sv) = S1(sv)."""
        rng = np.random.default_rng(42)
        sv = rng.random(50)
        s1_orig = compute_s1(sv)
        s1_scaled = compute_s1(sv * 1000.0)
        assert abs(s1_orig - s1_scaled) < 1e-10

    def test_known_spectrum(self):
        """S1 of a known decaying spectrum."""
        # Geometric decay: sv_i = r^i, probabilities p_i = r^{2i} / sum(r^{2j})
        r = 0.9
        n = 50
        sv = np.array([r**i for i in range(n)])
        s1 = compute_s1(sv)
        # S1 must be between 0 and log(n)
        assert 0 < s1 < math.log(n)


class TestComputeS2:
    """Renyi-2 entropy tests."""

    def test_uniform_spectrum(self):
        """S2 of D equal values = log(D)."""
        D = 100
        sv = np.ones(D)
        s2 = compute_s2(sv)
        expected = math.log(D)
        assert abs(s2 - expected) < 1e-10

    def test_single_value(self):
        """S2 of a single value = 0."""
        sv = np.array([1.0])
        s2 = compute_s2(sv)
        assert abs(s2) < 1e-10

    def test_s2_leq_s1(self):
        """S2 <= S1 for any non-uniform distribution."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            sv = rng.random(50) * 10
            s1 = compute_s1(sv)
            s2 = compute_s2(sv)
            assert s2 <= s1 + 1e-10, f"S2={s2} > S1={s1}"


class TestComputeDmin:
    """D_min (Eckart-Young threshold) tests."""

    def test_uniform_spectrum(self):
        """For uniform sv, D_min(eps) = ceil((1-eps^2) * D)."""
        D = 100
        sv = np.ones(D)
        for eps in [0.10, 0.20, 0.50]:
            dmin = compute_dmin(sv, eps)
            # Each sv contributes 1/D of total energy.
            # Need D_min such that D_min/D >= 1 - eps^2.
            expected = math.ceil((1.0 - eps**2) * D)
            assert dmin == expected, f"eps={eps}: expected {expected}, got {dmin}"

    def test_single_dominant(self):
        """If sv[0] >> rest, D_min should be 1 for large epsilon."""
        sv = np.array([100.0] + [0.001] * 99)
        assert compute_dmin(sv, 0.50) == 1

    def test_monotone_in_epsilon(self):
        """D_min decreases as epsilon increases."""
        rng = np.random.default_rng(42)
        sv = np.sort(rng.random(100))[::-1]
        prev = len(sv)
        for eps in [0.01, 0.05, 0.10, 0.20, 0.50]:
            d = compute_dmin(sv, eps)
            assert d <= prev, f"D_min({eps})={d} > D_min(prev)={prev}"
            prev = d

    def test_epsilon_one(self):
        """D_min(1.0) = 1 always (trivial case)."""
        sv = np.ones(100)
        assert compute_dmin(sv, 1.0) == 1

    def test_full_rank_for_tiny_epsilon(self):
        """D_min(0.0) = full rank (need all singular values)."""
        sv = np.ones(50)
        # eps=0 means target = total, so we need all
        assert compute_dmin(sv, 0.0) == 50


class TestParticipationRatio:
    """PR = exp(S2) tests."""

    def test_uniform(self):
        """PR of D equal values = D."""
        D = 100
        sv = np.ones(D)
        pr = compute_participation_ratio(sv)
        assert abs(pr - D) < 1e-6

    def test_single_value(self):
        """PR of a single value = 1."""
        sv = np.array([5.0])
        pr = compute_participation_ratio(sv)
        assert abs(pr - 1.0) < 1e-6


class TestAlphaHill:
    """Hill estimator tests."""

    def test_not_enough_values(self):
        """Returns NaN for too few values."""
        sv = np.array([1.0, 0.5])
        alpha = compute_alpha_hill(sv, k_min=10)
        assert np.isnan(alpha)

    def test_power_law_spectrum(self):
        """For sv_i ~ i^{-beta}, alpha should approximate 2*beta + 1."""
        # sv_i = i^{-1}, so density exponent ~ 2*1+1 = 3 (roughly)
        n = 1000
        sv = np.array([1.0 / (i + 1) for i in range(n)])
        alpha = compute_alpha_hill(sv, k_min=50)
        # Hill estimator on sorted data. For sv = i^{-1},
        # the tail index should be finite and positive.
        assert alpha > 0
        assert not np.isnan(alpha)
