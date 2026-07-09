"""Extract singular values from model weights, layer by layer, via safetensors mmap.

Key design decision: we never load the full model into RAM. We use safetensors'
memory-mapped file access to load one tensor at a time, compute svdvals on it,
and discard it before loading the next. Peak RAM: ~2-3 GB for 7B models.
"""

import json
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .arch.auto import detect_extractor


def _resolve_model_path(model_id: str) -> str:
    """Resolve a HuggingFace model ID to a local directory path.

    If model_id is already a local directory, return it as-is.
    Otherwise, download via huggingface_hub.
    """
    local = Path(model_id)
    if local.is_dir() and (local / "config.json").exists():
        return str(local)

    # Download only safetensors + config (skip pytorch_model.bin, tokenizer, etc.)
    path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.onnx", "*.msgpack"],
    )
    return path


def _find_safetensor_files(model_dir: str) -> list:
    """Find all .safetensors files in a model directory."""
    p = Path(model_dir)
    files = sorted(p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}. "
            f"Model might use pytorch_model.bin format (not supported)."
        )
    return files


def _build_key_to_file_map(safetensor_files: list) -> dict:
    """Build a mapping from safetensors key -> file path.

    For sharded models, each shard contains a subset of tensors.
    We scan all shards and build the index.
    """
    key_map = {}
    for fpath in safetensor_files:
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for key in f.keys():
                key_map[key] = str(fpath)
    return key_map


def extract_svdvals_streaming(
    model_path: str,
    dtype: str = "float32",
) -> Iterator[Tuple[str, np.ndarray]]:
    """Extract singular values from all weight matrices, one at a time.

    Loads each tensor via safetensors mmap, computes svdvals in fp32,
    and yields the result. Peak RAM is bounded by the largest single tensor
    plus LAPACK workspace (~2-3 GB for 4096x14336).

    Args:
        model_path: HuggingFace model ID or local directory path.
        dtype: compute dtype for svdvals. "float32" recommended.

    Yields:
        (canonical_name, singular_values) where singular_values is a 1D
        numpy array sorted descending.
    """
    model_dir = _resolve_model_path(model_path)
    extractor = detect_extractor(model_dir)
    safetensor_files = _find_safetensor_files(model_dir)
    key_map = _build_key_to_file_map(safetensor_files)

    # For GPT-2, c_attn is fused Q/K/V. We need to load it once and split.
    # Track which fused keys we've already processed.
    processed_fused = set()

    # Collect unique (canonical, st_key) pairs respecting fused weights
    for canonical, st_key in extractor.iter_weight_names():
        if st_key not in key_map:
            # Try without "model." prefix (some models don't have it)
            alt_key = st_key
            if st_key.startswith("model."):
                alt_key = st_key[len("model."):]
            elif not st_key.startswith("model."):
                alt_key = "model." + st_key
            if alt_key in key_map:
                st_key = alt_key
            else:
                continue

        if extractor.needs_split(st_key):
            if st_key in processed_fused:
                continue
            processed_fused.add(st_key)

            # Load fused tensor, split, compute svdvals for each part
            fpath = key_map[st_key]
            with safe_open(fpath, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(st_key)

            for sub_name, sub_tensor in extractor.split_fused(st_key, tensor):
                sv = _compute_svdvals(sub_tensor, dtype)
                yield sub_name, sv
            del tensor
        else:
            fpath = key_map[st_key]
            with safe_open(fpath, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(st_key)
            sv = _compute_svdvals(tensor, dtype)
            yield canonical, sv
            del tensor


def _compute_svdvals(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    """Compute singular values of a 2D tensor.

    Args:
        tensor: 2D weight matrix.
        dtype: "float32" or "float64".

    Returns:
        1D numpy array of singular values, descending, noise-filtered.
    """
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.ndim}D shape {tensor.shape}")

    target_dtype = torch.float64 if dtype == "float64" else torch.float32
    w = tensor.to(dtype=target_dtype)
    sv = torch.linalg.svdvals(w)
    sv_np = sv.numpy()

    # Filter numerical noise
    if len(sv_np) > 0 and sv_np[0] > 0:
        threshold = sv_np[0] * 1e-12
        sv_np = sv_np[sv_np > threshold]

    return sv_np
