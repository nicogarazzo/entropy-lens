# entropy-lens

Validate the Entropy-Compression Law (D_min ~ c * e^S1) across LLM architectures.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# Spike: verify svdvals feasibility on your hardware
python experiments/spike_svdvals.py

# Full analysis on GPT-2 Small (sanity check, ~5 min)
entropy-lens analyze openai-community/gpt2 --output results/gpt2/

# Or use the Python API
from entropy_lens.extract import extract_svdvals_streaming
from entropy_lens.spectral import compute_s1, compute_dmin

for name, sv in extract_svdvals_streaming("openai-community/gpt2"):
    print(f"{name}: S1={compute_s1(sv):.3f}, D_min(10%)={compute_dmin(sv, 0.10)}")
```

## Run tests

```bash
pytest
```
