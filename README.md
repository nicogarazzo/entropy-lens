# entropy-lens

**Von Neumann entanglement entropy predicts how much each layer of an LLM can be compressed.**

Compute the entanglement entropy S₁ of every weight matrix in a transformer via SVD, then read off the minimum bond dimension from the Entropy-Compression Law:

```
D_min(ε) ≈ c(ε) · exp(α(ε) · S₁)
```

## Validation results

| Model | Params | Matrices | R²(ε=10%) | R²(ε=50%) | Slope(ε=50%) |
|-------|--------|----------|-----------|-----------|-------------|
| GPT-2 Small | 124M | 72 | 0.808 | 0.865 | 1.001 |
| **Mistral 7B** | **7.2B** | **224** | **0.916** | **0.997** | **1.041** |
| **Qwen2 7B** | **7.1B** | **196** | **0.949** | **0.993** | **1.020** |

S₁ outperforms AlphaPruning's PL_Alpha_Hill metric as a compressibility predictor: R²=0.997 vs R²=0.022 on Mistral 7B (224 matrices).

## Install

```bash
git clone https://github.com/nicogarazzo/entropy-lens.git
cd entropy-lens
pip install -e ".[dev]"
```

## Quick start

```bash
# Analyze any HuggingFace model (~5 min for GPT-2, ~45 min for 7B)
entropy-lens analyze openai-community/gpt2 --output results/gpt2/

# Python API
from entropy_lens.extract import extract_svdvals_streaming
from entropy_lens.spectral import compute_s1, compute_dmin

for name, sv in extract_svdvals_streaming("mistralai/Mistral-7B-v0.3"):
    print(f"{name}: S1={compute_s1(sv):.3f}, D_min(10%)={compute_dmin(sv, 0.10)}")
```

## Supported architectures

- LLaMA 2/3, Mistral, Qwen (GQA, SwiGLU)
- GPT-2 (fused QKV)
- Phi-2/3

Layer-by-layer loading via safetensors mmap. Runs on a single Apple M1 with 16 GB RAM.

## Run tests

```bash
pytest  # 23 tests
```

## Citation

```bibtex
@misc{calderon2026entropycompression,
  author = {Calderon Gonzalez, Nicolas},
  title  = {Entanglement Cartography of Large Language Models:
            An Entropy-Compression Law for Tensor Network Compression},
  year   = {2026},
  note   = {Preprint. Code: https://github.com/nicogarazzo/entropy-compression-law}
}
```

## License

MIT
