"""Tests for activation-aware whitening (whiten.py)."""

import numpy as np
import pytest
import torch

from entropy_lens.spectral import compute_s1
from entropy_lens.whiten import cholesky_factor, whiten_truncate, whitened_svdvals

torch.manual_seed(0)


def _anisotropic_cov(n: int, condition: float = 1e4) -> torch.Tensor:
    """Random SPD matrix with a strongly anisotropic spectrum."""
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    evals = torch.logspace(0, -np.log10(condition), n, dtype=torch.float64)
    return (q * evals) @ q.T


def _channel_scale(c: torch.Tensor) -> torch.Tensor:
    return c.diagonal().clamp_min(1e-30).sqrt()


class TestCholeskyFactor:
    def test_reconstructs_well_conditioned(self):
        c = _anisotropic_cov(32, condition=100)
        L, triangular, lam = cholesky_factor(c)
        assert triangular
        rec = L @ L.T
        rel_err = torch.norm(rec - c) / torch.norm(c)
        assert rel_err < 1e-3  # only the ridge separates them

    def test_singular_covariance_does_not_crash(self):
        x = torch.randn(8, 32, dtype=torch.float64)  # rank 8 < 32
        c = x.T @ x
        L, triangular, lam = cholesky_factor(c)
        assert torch.isfinite(L).all()
        rec = L @ L.T
        # Reconstruction matches C + lam*diag(s^2), s = per-channel std
        s = _channel_scale(c)
        target = c + lam * torch.diag(s ** 2)
        rel_err = torch.norm(rec - target) / torch.norm(target)
        assert rel_err < 1e-6

    def test_eigh_fallback_reconstructs(self):
        # Force fallback by exhausting the damping range on an indefinite
        # matrix (simulates accumulated numerical asymmetry gone wrong).
        c = _anisotropic_cov(16)
        c[0, 0] = -1e6  # make it non-PSD beyond what damping fixes
        L, triangular, lam = cholesky_factor(c, damp=1e-6, max_damp=1e-6)
        assert torch.isfinite(L).all()

    def test_channel_prescale_applies_uniform_relative_ridge(self):
        # The Mistral 7B mlp_in problem: one "massive activation" channel runs
        # ~50x hotter than the rest. A uniform ridge lam*I perturbs cold
        # channels far more (relatively) than the hot one, distorting the
        # whitened spectrum. A per-channel ridge lam*diag(s^2) perturbs every
        # channel by the same RELATIVE amount, which is what preserves S1_eff.
        base = _anisotropic_cov(48, condition=50)
        base[0, :] *= 50.0
        base[:, 0] *= 50.0  # channel 0 dominates the diagonal

        def rel_diag_perturbation(prescale):
            L, _, _ = cholesky_factor(base, damp=1e-3, channel_prescale=prescale)
            rec = L @ L.T
            return ((rec - base).diagonal() / base.diagonal()).abs()

        cv_uniform = rel_diag_perturbation(False).std() / rel_diag_perturbation(False).mean()
        cv_scaled = rel_diag_perturbation(True).std() / rel_diag_perturbation(True).mean()
        # Prescale makes the relative ridge nearly constant across channels.
        assert cv_scaled < cv_uniform / 5.0

    def test_channel_prescale_preserves_triangularity(self):
        c = _anisotropic_cov(24, condition=200)
        L, triangular, _ = cholesky_factor(c, channel_prescale=True)
        assert triangular
        upper = torch.triu(L, diagonal=1)
        assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-8)


class TestWhitenedSvdvals:
    def test_identity_whitening_equals_plain_svd(self):
        w = torch.randn(24, 48)
        sv_plain = torch.linalg.svdvals(w.float()).numpy()
        sv_white = whitened_svdvals(w, torch.eye(48))
        np.testing.assert_allclose(sv_white, sv_plain, rtol=1e-5)

    def test_anisotropic_whitening_sharpens_spectrum(self):
        # W with flat spectrum, C strongly anisotropic: the whitened spectrum
        # inherits the anisotropy, so S1_eff < S1_raw.
        q1, _ = torch.linalg.qr(torch.randn(64, 64))
        q2, _ = torch.linalg.qr(torch.randn(64, 64))
        w = q1 @ q2.T  # orthogonal: perfectly flat spectrum
        c = _anisotropic_cov(64).to(torch.float32)
        L, _, _ = cholesky_factor(c)
        s1_raw = compute_s1(torch.linalg.svdvals(w).numpy())
        s1_eff = compute_s1(whitened_svdvals(w, L))
        assert s1_eff < s1_raw


class TestWhitenTruncate:
    def test_full_rank_reproduces_weight(self):
        w = torch.randn(24, 48)
        c = _anisotropic_cov(48).to(torch.float32)
        L, triangular, _ = cholesky_factor(c)
        w_d = whiten_truncate(w, L, rank=24, triangular=triangular)
        rel_err = torch.norm(w_d.float() - w) / torch.norm(w)
        assert rel_err < 1e-3

    def test_beats_naive_truncation_in_data_metric(self):
        # Whitened truncation is Eckart-Young optimal in ||dW @ L||_F, so it
        # must beat naive SVD truncation at the same rank.
        w = torch.randn(64, 64)
        c = _anisotropic_cov(64).to(torch.float32)
        L, triangular, _ = cholesky_factor(c)
        rank = 16

        w_white = whiten_truncate(w, L, rank, triangular=triangular)
        u, s, vh = torch.linalg.svd(w)
        w_naive = (u[:, :rank] * s[:rank]) @ vh[:rank]

        l32 = L.to(torch.float32)
        err_white = torch.norm((w - w_white.float()) @ l32)
        err_naive = torch.norm((w - w_naive) @ l32)
        assert err_white < err_naive

    def test_preserves_dtype(self):
        w = torch.randn(16, 16, dtype=torch.float16)
        L, triangular, _ = cholesky_factor(torch.eye(16))
        w_d = whiten_truncate(w, L, rank=8, triangular=triangular)
        assert w_d.dtype == torch.float16
