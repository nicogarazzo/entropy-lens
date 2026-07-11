"""SVD-based model compression using entropy-guided rank allocation.

Takes a HuggingFace model + per-layer rank assignments from the allocator,
applies truncated SVD to each weight matrix, reconstructs the dense matrix
at the lower rank, and saves the result as a standard HuggingFace model.

The compressed model can be loaded with AutoModelForCausalLM.from_pretrained()
and evaluated with lm-evaluation-harness or any other standard pipeline.

Memory strategy: we load the full model in the target dtype, then process
each weight matrix one at a time. For a 7B model in fp16 this needs ~14GB
plus ~2GB SVD workspace. If that doesn't fit, use the layer-by-layer mode
(not yet implemented — document what's needed and move on).
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .arch.auto import detect_extractor
from .extract import _resolve_model_path

logger = logging.getLogger(__name__)

# Projection types that we compress. Everything else (embeddings, norms,
# lm_head, biases) stays untouched.
_COMPRESSIBLE_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"}


def _svd_truncate(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """Apply truncated SVD to a 2D weight matrix and return the reconstruction.

    Args:
        weight: (m, n) tensor.
        rank: number of singular values to keep. Clamped to [1, min(m, n)].

    Returns:
        Reconstructed (m, n) tensor at the given rank.
    """
    m, n = weight.shape
    rank = max(1, min(rank, min(m, n)))

    # SVD in float32 for numerical stability, then cast back.
    original_dtype = weight.dtype
    w = weight.float()

    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    # Truncate to rank D
    U_d = U[:, :rank]
    S_d = S[:rank]
    Vh_d = Vh[:rank, :]

    # Reconstruct: U_d @ diag(S_d) @ Vh_d
    reconstructed = (U_d * S_d.unsqueeze(0)) @ Vh_d

    return reconstructed.to(original_dtype)


def _frobenius_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Relative Frobenius error: ||W - W_D||_F / ||W||_F."""
    diff_norm = torch.linalg.norm((original - reconstructed).float()).item()
    orig_norm = torch.linalg.norm(original.float()).item()
    if orig_norm == 0:
        return 0.0
    return diff_norm / orig_norm


def _build_canonical_to_statedict_map(
    extractor,
) -> dict[str, list[str]]:
    """Build mapping from canonical name to state_dict key(s).

    For most architectures this is 1:1. For GPT-2's fused c_attn,
    multiple canonical names (q/k/v_proj) map to the same state_dict key,
    so we handle that separately.

    Returns:
        dict mapping canonical name -> list of state_dict keys.
        (Usually a single-element list, except for fused weights.)
    """
    mapping = {}
    for canonical, st_key in extractor.iter_weight_names():
        mapping[canonical] = st_key
    return mapping


def _is_fused_architecture(extractor) -> bool:
    """Check if this architecture has fused weights (e.g., GPT-2 c_attn)."""
    from .arch.gpt2 import GPT2Extractor
    return isinstance(extractor, GPT2Extractor)


def compress_model(
    model_path: str,
    ranks: dict[str, int],
    output_path: str,
    dtype: str = "float16",
    verify: bool = True,
) -> dict[str, float]:
    """Compress a HuggingFace model using truncated SVD at specified ranks.

    Args:
        model_path: HuggingFace model ID or local path.
        ranks: dict mapping canonical layer names (e.g., "layer_0.q_proj")
               to truncation ranks. Layers not in this dict are left untouched.
        output_path: directory to save the compressed model.
        dtype: model loading dtype ("float16", "bfloat16", "float32").
        verify: if True, log per-layer Frobenius reconstruction errors.

    Returns:
        dict mapping layer names to their relative Frobenius errors.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model path and detect architecture
    model_dir = _resolve_model_path(model_path)
    extractor = detect_extractor(model_dir)

    # Build canonical -> state_dict key mapping
    canonical_map = _build_canonical_to_statedict_map(extractor)
    is_fused = _is_fused_architecture(extractor)

    # Choose torch dtype
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    logger.info("Loading model: %s (dtype=%s)", model_path, dtype)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    logger.info("Model loaded in %.1fs", time.time() - t0)

    state_dict = model.state_dict()
    errors = {}
    compressed_count = 0
    skipped_count = 0

    if is_fused:
        # GPT-2: handle fused c_attn weights.
        # Group canonical names by their shared state_dict key.
        fused_groups: dict[str, list[str]] = {}
        non_fused: dict[str, str] = {}

        for canonical, st_key in canonical_map.items():
            if extractor.needs_split(st_key):
                fused_groups.setdefault(st_key, []).append(canonical)
            else:
                non_fused[canonical] = st_key

        # Process fused weights
        for st_key, canonical_names in fused_groups.items():
            # Check if any of the canonicals have ranks assigned
            assigned = {c: ranks[c] for c in canonical_names if c in ranks}
            if not assigned:
                skipped_count += len(canonical_names)
                continue

            weight = state_dict[st_key]
            original_weight = weight.clone() if verify else None

            # Split into individual projections
            splits = extractor.split_fused(st_key, weight)
            compressed_parts = []

            for sub_name, sub_tensor in splits:
                if sub_name in ranks:
                    rank = ranks[sub_name]
                    original_sub = sub_tensor.clone() if verify else None
                    compressed_sub = _svd_truncate(sub_tensor, rank)

                    if verify and original_sub is not None:
                        err = _frobenius_error(original_sub, compressed_sub)
                        errors[sub_name] = err
                        logger.info(
                            "  %s: rank %d/%d, error=%.4f",
                            sub_name, rank, min(sub_tensor.shape), err,
                        )
                        del original_sub

                    compressed_parts.append(compressed_sub)
                    compressed_count += 1
                else:
                    compressed_parts.append(sub_tensor)
                    skipped_count += 1

            # Rejoin: concatenate along the split dimension (dim=1 for GPT-2 c_attn)
            rejoined = torch.cat(compressed_parts, dim=1)
            state_dict[st_key] = rejoined

            del weight, compressed_parts
            if original_weight is not None:
                del original_weight

        # Process non-fused weights
        for canonical, st_key in non_fused.items():
            if canonical not in ranks:
                skipped_count += 1
                continue

            rank = ranks[canonical]
            weight = state_dict[st_key]
            original_weight = weight.clone() if verify else None

            compressed = _svd_truncate(weight, rank)

            if verify and original_weight is not None:
                err = _frobenius_error(original_weight, compressed)
                errors[canonical] = err
                logger.info(
                    "  %s: rank %d/%d, error=%.4f",
                    canonical, rank, min(weight.shape), err,
                )
                del original_weight

            state_dict[st_key] = compressed
            compressed_count += 1
            del weight

    else:
        # Standard architecture (LLaMA, Mistral, Qwen, Phi): 1:1 mapping
        for canonical, st_key in canonical_map.items():
            if canonical not in ranks:
                skipped_count += 1
                continue

            rank = ranks[canonical]

            # The state_dict key might need "model." prefix adjustment
            actual_key = st_key
            if actual_key not in state_dict:
                # Try without "model." prefix
                if actual_key.startswith("model."):
                    alt = actual_key[len("model."):]
                else:
                    alt = "model." + actual_key
                if alt in state_dict:
                    actual_key = alt
                else:
                    logger.warning(
                        "Key not found in state_dict: %s (tried %s). Skipping.",
                        st_key, alt,
                    )
                    skipped_count += 1
                    continue

            weight = state_dict[actual_key]
            original_weight = weight.clone() if verify else None

            compressed = _svd_truncate(weight, rank)

            if verify and original_weight is not None:
                err = _frobenius_error(original_weight, compressed)
                errors[canonical] = err
                logger.info(
                    "  %s: rank %d/%d, error=%.4f",
                    canonical, rank, min(weight.shape), err,
                )
                del original_weight

            state_dict[actual_key] = compressed
            compressed_count += 1
            del weight

    logger.info(
        "Compression done: %d layers compressed, %d skipped",
        compressed_count, skipped_count,
    )

    # Load modified state_dict back into model
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()

    # Save model
    logger.info("Saving compressed model to %s", output_path)
    model.save_pretrained(out_dir)

    # Also save tokenizer if available
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("Tokenizer saved.")
    except Exception:
        logger.warning("Could not save tokenizer (non-fatal).")

    # Save compression metadata
    metadata = {
        "source_model": model_path,
        "dtype": dtype,
        "n_compressed_layers": compressed_count,
        "n_skipped_layers": skipped_count,
        "ranks": ranks,
        "errors": {k: round(v, 6) for k, v in errors.items()},
    }
    meta_path = out_dir / "compression_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return errors
