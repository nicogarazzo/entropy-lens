"""Tests for SVD model compression.

Uses GPT-2 Small (124M params, ~500MB) which fits comfortably in RAM.
Downloads from HuggingFace on first run, cached after that.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from entropy_lens.compress import _svd_truncate, _frobenius_error, compress_model


# ---------------------------------------------------------------------------
# Unit tests: SVD truncation core
# ---------------------------------------------------------------------------


class TestSVDTruncate:
    def test_full_rank_is_identity(self):
        """Truncation at full rank should reconstruct the original matrix exactly."""
        torch.manual_seed(42)
        W = torch.randn(64, 32, dtype=torch.float32)
        reconstructed = _svd_truncate(W, rank=32)
        # Full rank reconstruction should be near-perfect (float32 precision)
        err = _frobenius_error(W, reconstructed)
        assert err < 1e-5, f"Full-rank error too high: {err}"

    def test_rank_1_is_best_rank1_approx(self):
        """Rank-1 SVD should capture the top singular value's energy."""
        torch.manual_seed(42)
        W = torch.randn(64, 32, dtype=torch.float32)
        reconstructed = _svd_truncate(W, rank=1)
        assert reconstructed.shape == W.shape

        # Verify it's rank 1: second singular value should be ~0
        sv = torch.linalg.svdvals(reconstructed.float())
        assert sv[1].item() < 1e-5

    def test_lower_rank_has_higher_error(self):
        """Lower truncation rank should produce higher reconstruction error."""
        torch.manual_seed(42)
        W = torch.randn(128, 64, dtype=torch.float32)
        err_high = _frobenius_error(W, _svd_truncate(W, rank=10))
        err_low = _frobenius_error(W, _svd_truncate(W, rank=50))
        assert err_high > err_low, (
            f"rank=10 error ({err_high}) should be > rank=50 error ({err_low})"
        )

    def test_preserves_dtype_float16(self):
        """Output should match input dtype."""
        W = torch.randn(32, 16, dtype=torch.float16)
        reconstructed = _svd_truncate(W, rank=8)
        assert reconstructed.dtype == torch.float16

    def test_preserves_dtype_bfloat16(self):
        W = torch.randn(32, 16, dtype=torch.bfloat16)
        reconstructed = _svd_truncate(W, rank=8)
        assert reconstructed.dtype == torch.bfloat16

    def test_rank_clamped_to_min_dim(self):
        """Rank larger than min(m,n) should be clamped, not crash."""
        W = torch.randn(64, 32, dtype=torch.float32)
        reconstructed = _svd_truncate(W, rank=999)
        err = _frobenius_error(W, reconstructed)
        assert err < 1e-5

    def test_rank_clamped_to_1(self):
        """Rank 0 or negative should be clamped to 1."""
        W = torch.randn(32, 16, dtype=torch.float32)
        reconstructed = _svd_truncate(W, rank=0)
        assert reconstructed.shape == W.shape
        sv = torch.linalg.svdvals(reconstructed.float())
        # Only 1 non-zero singular value
        assert sv[1].item() < 1e-5

    def test_frobenius_error_eckart_young(self):
        """Verify Frobenius error matches Eckart-Young theorem prediction.

        ||W - W_D||_F^2 = sum_{i>D} sigma_i^2
        ||W||_F^2 = sum_i sigma_i^2
        relative error = sqrt(sum_{i>D} sigma_i^2 / sum_i sigma_i^2)
        """
        torch.manual_seed(123)
        W = torch.randn(64, 32, dtype=torch.float32)
        rank = 10

        # Compute expected error from singular values
        sv = torch.linalg.svdvals(W)
        sv_sq = sv ** 2
        total_energy = sv_sq.sum().item()
        tail_energy = sv_sq[rank:].sum().item()
        expected_error = math.sqrt(tail_energy / total_energy)

        # Compute actual error from reconstruction
        reconstructed = _svd_truncate(W, rank=rank)
        actual_error = _frobenius_error(W, reconstructed)

        assert abs(actual_error - expected_error) < 1e-5, (
            f"Actual error {actual_error:.6f} != expected {expected_error:.6f}"
        )


class TestFrobeniusError:
    def test_zero_error_for_identical(self):
        W = torch.randn(16, 8)
        assert _frobenius_error(W, W.clone()) == 0.0

    def test_zero_matrix(self):
        W = torch.zeros(16, 8)
        assert _frobenius_error(W, W) == 0.0

    def test_error_is_positive(self):
        W = torch.randn(32, 16)
        W2 = W + 0.1 * torch.randn_like(W)
        assert _frobenius_error(W, W2) > 0.0


# ---------------------------------------------------------------------------
# Integration test: compress GPT-2 Small
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_model_path():
    """Resolve GPT-2 model path (downloads if needed)."""
    from entropy_lens.extract import _resolve_model_path
    return _resolve_model_path("openai-community/gpt2")


class TestCompressGPT2:
    """End-to-end compression of GPT-2 Small.

    GPT-2 Small has 12 layers, each with:
      - c_attn (fused Q/K/V): (768, 2304) Conv1D
      - c_proj (O): (768, 768)
      - c_fc (up): (768, 3072)
      - c_proj (down): (3072, 768)

    This tests the fused weight handling path.
    """

    def test_compress_uniform_ranks(self, gpt2_model_path, tmp_path):
        """Compress all layers to rank 64 and verify forward pass works."""
        # Build uniform rank assignment for all 12 layers
        ranks = {}
        for i in range(12):
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]:
                ranks[f"layer_{i}.{proj}"] = 64

        output_dir = str(tmp_path / "gpt2-compressed")

        errors = compress_model(
            model_path=gpt2_model_path,
            ranks=ranks,
            output_path=output_dir,
            dtype="float32",
            verify=True,
        )

        # All layers should have been compressed
        assert len(errors) == 72, f"Expected 72 layers, got {len(errors)}"

        # Errors should be reasonable (not zero, not huge)
        for name, err in errors.items():
            assert 0.0 < err < 1.0, f"{name}: error {err} out of range"

        # Verify saved files exist
        out = Path(output_dir)
        assert (out / "config.json").exists()
        assert any(out.glob("*.safetensors")) or any(out.glob("*.bin"))
        assert (out / "compression_metadata.json").exists()

        # Load the compressed model and do a forward pass
        from transformers import AutoModelForCausalLM, AutoTokenizer

        compressed = AutoModelForCausalLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)

        inputs = tokenizer("The quick brown fox", return_tensors="pt")
        with torch.no_grad():
            outputs = compressed(**inputs)

        # Should produce valid logits
        assert outputs.logits.shape[-1] == tokenizer.vocab_size
        assert not torch.isnan(outputs.logits).any()
        assert not torch.isinf(outputs.logits).any()

    def test_compress_partial_layers(self, gpt2_model_path, tmp_path):
        """Compress only some layers; others should be untouched."""
        # Only compress layer 0 attention
        ranks = {
            "layer_0.q_proj": 32,
            "layer_0.k_proj": 32,
            "layer_0.v_proj": 32,
            "layer_0.o_proj": 32,
        }

        output_dir = str(tmp_path / "gpt2-partial")

        errors = compress_model(
            model_path=gpt2_model_path,
            ranks=ranks,
            output_path=output_dir,
            dtype="float32",
            verify=True,
        )

        assert len(errors) == 4
        assert all(name.startswith("layer_0.") for name in errors)

        # Forward pass should still work
        from transformers import AutoModelForCausalLM, AutoTokenizer

        compressed = AutoModelForCausalLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)

        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            outputs = compressed(**inputs)
        assert not torch.isnan(outputs.logits).any()

    def test_metadata_saved_correctly(self, gpt2_model_path, tmp_path):
        """Verify compression metadata JSON is complete and correct."""
        ranks = {"layer_0.q_proj": 16, "layer_0.k_proj": 16}
        output_dir = str(tmp_path / "gpt2-meta")

        compress_model(
            model_path=gpt2_model_path,
            ranks=ranks,
            output_path=output_dir,
            dtype="float32",
            verify=True,
        )

        meta_path = Path(output_dir) / "compression_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["n_compressed_layers"] == 2
        assert "layer_0.q_proj" in meta["ranks"]
        assert "layer_0.k_proj" in meta["ranks"]
        assert "errors" in meta
        assert len(meta["errors"]) == 2

    def test_higher_rank_lower_error(self, gpt2_model_path, tmp_path):
        """Compressing at higher rank should produce lower reconstruction error."""
        errors_low = compress_model(
            model_path=gpt2_model_path,
            ranks={"layer_0.q_proj": 16},
            output_path=str(tmp_path / "low"),
            dtype="float32",
        )
        errors_high = compress_model(
            model_path=gpt2_model_path,
            ranks={"layer_0.q_proj": 256},
            output_path=str(tmp_path / "high"),
            dtype="float32",
        )

        assert errors_low["layer_0.q_proj"] > errors_high["layer_0.q_proj"]
