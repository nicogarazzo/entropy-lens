"""Integration tests for weight extraction using GPT-2 Small.

These tests require downloading GPT-2 Small (~500 MB) from HuggingFace.
They verify that the extraction pipeline produces the correct number of
matrices with the expected names and shapes.
"""

import numpy as np
import pytest

from entropy_lens.extract import extract_svdvals_streaming
from entropy_lens.spectral import compute_s1


GPT2_MODEL = "openai-community/gpt2"
GPT2_NUM_LAYERS = 12
GPT2_PROJS_PER_LAYER = 6  # q, k, v, o, up, down
GPT2_TOTAL_MATRICES = GPT2_NUM_LAYERS * GPT2_PROJS_PER_LAYER  # 72

# Expected shapes for GPT-2 Small (hidden=768, ffn=3072):
# q/k/v/o_proj: (768, 768) -> 768 singular values
# up_proj: (768, 3072) -> 768 singular values (min dimension)
# down_proj: (3072, 768) -> 768 singular values
GPT2_SV_COUNT = 768


@pytest.fixture(scope="module")
def gpt2_results():
    """Extract all svdvals from GPT-2 Small (cached across tests in this module)."""
    results = list(extract_svdvals_streaming(GPT2_MODEL))
    return results


def test_correct_number_of_matrices(gpt2_results):
    """GPT-2 Small should yield exactly 72 matrices."""
    assert len(gpt2_results) == GPT2_TOTAL_MATRICES, (
        f"Expected {GPT2_TOTAL_MATRICES} matrices, got {len(gpt2_results)}"
    )


def test_canonical_names(gpt2_results):
    """All canonical names should follow the layer_{i}.{proj_type} pattern."""
    names = [name for name, _ in gpt2_results]

    expected_projs = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"}

    for name in names:
        parts = name.split(".")
        assert len(parts) == 2, f"Bad name format: {name}"
        layer_part, proj_part = parts
        assert layer_part.startswith("layer_"), f"Bad layer prefix: {name}"
        layer_idx = int(layer_part.split("_")[1])
        assert 0 <= layer_idx < GPT2_NUM_LAYERS, f"Bad layer index in {name}"
        assert proj_part in expected_projs, f"Unknown proj type in {name}: {proj_part}"

    # Check all layers and proj types are present
    for i in range(GPT2_NUM_LAYERS):
        for proj in expected_projs:
            expected_name = f"layer_{i}.{proj}"
            assert expected_name in names, f"Missing: {expected_name}"


def test_singular_value_shapes(gpt2_results):
    """All GPT-2 Small matrices yield 768 singular values."""
    for name, sv in gpt2_results:
        assert sv.ndim == 1, f"{name}: expected 1D, got {sv.ndim}D"
        assert len(sv) == GPT2_SV_COUNT, (
            f"{name}: expected {GPT2_SV_COUNT} svdvals, got {len(sv)}"
        )


def test_singular_values_positive_descending(gpt2_results):
    """Singular values should be positive and in descending order."""
    for name, sv in gpt2_results:
        assert np.all(sv > 0), f"{name}: has non-positive singular values"
        assert np.all(np.diff(sv) <= 1e-6), (
            f"{name}: singular values not in descending order"
        )


def test_mean_s1_gpt2(gpt2_results):
    """Mean S1 across all 72 matrices should be approximately 6.04 nats (paper value)."""
    s1_values = [compute_s1(sv) for _, sv in gpt2_results]
    mean_s1 = np.mean(s1_values)

    # Paper reports S1_bar ~ 6.04 for GPT-2 Small.
    # Allow 5% tolerance for numerical differences (fp32 vs fp64, etc.)
    assert 5.5 < mean_s1 < 6.5, (
        f"Mean S1 = {mean_s1:.4f}, expected ~6.04 (paper value)"
    )
