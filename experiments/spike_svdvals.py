#!/usr/bin/env python3
"""
Spike de viabilidad: verify that svdvals of large matrices works on this hardware.

Tests:
1. svdvals of a random 4096x14336 matrix (largest FFN shape in 7B models)
2. Peak RAM during computation
3. safetensors mmap loading of a single tensor (if a model is available)

Go/no-go: svdvals < 180s AND peak RAM < 8 GB.
"""

import resource
import sys
import time

import numpy as np
import torch


def get_peak_ram_mb() -> float:
    """Get peak RSS in MB (macOS/Linux)."""
    # ru_maxrss is in bytes on macOS, KB on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)  # bytes -> MB
    return peak / 1024  # KB -> MB


def test_svdvals_random():
    """Benchmark svdvals on a random matrix matching 7B FFN dimensions."""
    print("=" * 60)
    print("SPIKE 1: svdvals of random 4096 x 14336 matrix (fp32)")
    print("=" * 60)

    m, n = 4096, 14336
    print(f"  Generating random matrix ({m} x {n}) in fp32...")
    matrix = torch.randn(m, n, dtype=torch.float32)
    mem_matrix_mb = matrix.nelement() * 4 / (1024 * 1024)
    print(f"  Matrix memory: {mem_matrix_mb:.1f} MB")

    ram_before = get_peak_ram_mb()
    print(f"  Peak RAM before: {ram_before:.0f} MB")

    print(f"  Computing svdvals...")
    t0 = time.perf_counter()
    sv = torch.linalg.svdvals(matrix)
    elapsed = time.perf_counter() - t0

    ram_after = get_peak_ram_mb()
    print(f"  Peak RAM after: {ram_after:.0f} MB")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Singular values shape: {sv.shape}")
    print(f"  sv[0]={sv[0]:.4f}, sv[-1]={sv[-1]:.6f}")

    # Also test a smaller matrix (attention projection size)
    print(f"\n  --- Bonus: svdvals of 4096 x 4096 (attention proj) ---")
    mat2 = torch.randn(4096, 4096, dtype=torch.float32)
    t0 = time.perf_counter()
    sv2 = torch.linalg.svdvals(mat2)
    elapsed2 = time.perf_counter() - t0
    print(f"  Time: {elapsed2:.1f}s")

    print(f"\n  --- Bonus: svdvals of 4096 x 1024 (GQA K/V proj) ---")
    mat3 = torch.randn(4096, 1024, dtype=torch.float32)
    t0 = time.perf_counter()
    sv3 = torch.linalg.svdvals(mat3)
    elapsed3 = time.perf_counter() - t0
    print(f"  Time: {elapsed3:.1f}s")

    return elapsed, ram_after


def test_safetensors_mmap():
    """Try to load a single tensor via safetensors mmap (if available)."""
    print("\n" + "=" * 60)
    print("SPIKE 2: safetensors mmap tensor loading")
    print("=" * 60)

    try:
        from safetensors import safe_open
    except ImportError:
        print("  safetensors not installed. Skipping.")
        return False

    # Try to find a local HF model with safetensors
    from pathlib import Path
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists():
        print(f"  No HF cache at {hf_cache}. Skipping mmap test.")
        print("  (Run 'huggingface-cli download openai-community/gpt2' first)")
        return False

    # Find any .safetensors file
    safetensor_files = list(hf_cache.rglob("*.safetensors"))
    if not safetensor_files:
        print("  No .safetensors files in HF cache. Skipping.")
        return False

    fpath = safetensor_files[0]
    print(f"  Found: {fpath.name}")
    print(f"  File size: {fpath.stat().st_size / (1024*1024):.1f} MB")

    t0 = time.perf_counter()
    with safe_open(str(fpath), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"  Keys in file: {len(keys)}")
        if keys:
            first_key = keys[0]
            tensor = f.get_tensor(first_key)
            elapsed = time.perf_counter() - t0
            print(f"  Loaded '{first_key}': shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"  Load time: {elapsed*1000:.1f}ms")
            return True

    return False


def main():
    print("\nentropy-lens Spike de Viabilidad")
    print("Hardware: Apple Silicon M1 16GB target\n")

    # Spike 1: svdvals benchmark
    elapsed, peak_ram = test_svdvals_random()

    # Spike 2: safetensors mmap
    mmap_ok = test_safetensors_mmap()

    # Go/no-go
    print("\n" + "=" * 60)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 60)

    svd_ok = elapsed < 180
    ram_ok = peak_ram < 8192  # 8 GB

    print(f"  svdvals 4096x14336: {elapsed:.1f}s {'[GO]' if svd_ok else '[NO-GO: > 180s]'}")
    print(f"  Peak RAM: {peak_ram:.0f} MB {'[GO]' if ram_ok else '[NO-GO: > 8 GB]'}")
    print(f"  safetensors mmap: {'[GO]' if mmap_ok else '[SKIPPED]'}")

    if svd_ok and ram_ok:
        # Estimate total time for a 7B model
        # 7 projections * 32 layers = 224 matrices
        # Rough: 3 FFN (4096x14336) * 32 = 96 @ elapsed each
        #        4 attn (4096x4096 or 4096x1024) * 32 = 128 @ much less
        est_ffn = elapsed * 96
        est_attn = elapsed * 0.15 * 128  # attn is ~15% the time
        est_total = est_ffn + est_attn
        print(f"\n  Estimated time for 7B model: {est_total/3600:.1f} hours")
        print(f"  (96 FFN matrices @ {elapsed:.0f}s + 128 attn matrices @ {elapsed*0.15:.0f}s)")
        print("\n  VERDICT: GO. Proceed with implementation.")
    else:
        print("\n  VERDICT: NO-GO. Consider GPU cloud or optimization.")


if __name__ == "__main__":
    main()
