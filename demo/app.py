"""entropy-lens Gradio demo: visualize the entanglement entropy map of any LLM.

Deploy to Hugging Face Spaces:
  1. Create a new Space (Gradio SDK)
  2. Upload this file + requirements.txt
  3. Done — users can analyze any public HF model
"""

import json
import os
import tempfile
import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_model(model_id: str, progress=gr.Progress()):
    """Run entropy-lens analysis on a HuggingFace model."""
    try:
        from entropy_lens.extract import extract_svdvals_streaming
        from entropy_lens.spectral import compute_s1, compute_s2, compute_dmin
        from entropy_lens.law import fit_entropy_law
    except ImportError:
        return None, None, "Error: entropy_lens not installed. Run `pip install -e .`"

    progress(0, desc="Loading model metadata...")

    rows = []
    t0 = time.time()

    try:
        for i, (name, sv) in enumerate(extract_svdvals_streaming(model_id)):
            s1 = compute_s1(sv)
            s2 = compute_s2(sv)
            d10 = compute_dmin(sv, 0.10)
            d50 = compute_dmin(sv, 0.50)

            layer_idx = int(name.split(".")[0].split("_")[1]) if "layer_" in name else -1
            proj = name.split(".")[-1]

            rows.append({
                "layer": layer_idx, "projection": proj, "name": name,
                "S₁ (nats)": round(s1, 3), "S₂ (nats)": round(s2, 3),
                "D_min(10%)": d10, "D_min(50%)": d50,
                "rank": sv.shape[0],
            })

            if (i + 1) % 7 == 0:
                progress((i + 1) / max(i + 2, 1), desc=f"Processing {name}...")
    except Exception as e:
        return None, None, f"Error analyzing model: {str(e)}"

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    # ── Entropy map plot ──
    fig, ax = plt.subplots(figsize=(12, 5))

    COLORS = {
        "q_proj": "#2166ac", "k_proj": "#4393c3", "v_proj": "#92c5de", "o_proj": "#d1e5f0",
        "gate_proj": "#b2182b", "up_proj": "#d6604d", "down_proj": "#f4a582",
    }

    for proj, color in COLORS.items():
        mask = df["projection"] == proj
        if mask.any():
            sub = df[mask].sort_values("layer")
            ax.plot(sub["layer"], sub["S₁ (nats)"], "o-", color=color,
                    markersize=3, linewidth=1, alpha=0.8, label=proj)

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("S₁ (von Neumann entropy, nats)", fontsize=11)
    ax.set_title(f"{model_id}: Entanglement Entropy Map ({len(rows)} matrices, {elapsed:.0f}s)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ── Fit the law ──
    s1_arr = df["S₁ (nats)"].values
    d50_arr = df["D_min(50%)"].values.astype(float)
    d10_arr = df["D_min(10%)"].values.astype(float)

    from scipy import stats
    sl50, ic50, r50, _, _ = stats.linregress(s1_arr, np.log(d50_arr))
    sl10, ic10, r10, _, _ = stats.linregress(s1_arr, np.log(d10_arr))

    summary = f"""## Results for `{model_id}`

| Metric | Value |
|--------|-------|
| Matrices analyzed | {len(rows)} |
| Time | {elapsed:.0f}s |
| S₁ mean | {s1_arr.mean():.3f} nats |
| S₁ range | [{s1_arr.min():.3f}, {s1_arr.max():.3f}] |
| **R² (ε=50%)** | **{r50**2:.4f}** |
| Slope (ε=50%) | {sl50:.3f} |
| **R² (ε=10%)** | **{r10**2:.4f}** |
| Slope (ε=10%) | {sl10:.3f} |

### Interpretation

{"The Entropy-Compression Law holds well (R² > 0.85)." if r50**2 > 0.85 else "R² is moderate — the law may need calibration for this architecture." if r50**2 > 0.7 else "R² is low — the law does not fit this model well."}

Attention layers (S₁ ≈ {df[df['projection'].isin(['q_proj','k_proj','v_proj','o_proj'])]['S₁ (nats)'].mean():.2f}) are more compressible than FFN layers (S₁ ≈ {df[df['projection'].isin(['gate_proj','up_proj','down_proj'])]['S₁ (nats)'].mean():.2f}).
"""

    return fig, df[["name", "S₁ (nats)", "D_min(10%)", "D_min(50%)", "rank"]], summary


# ── Gradio interface ──
with gr.Blocks(title="entropy-lens", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# entropy-lens: Entanglement Entropy Map of LLMs

**Von Neumann entanglement entropy predicts how much each layer of an LLM can be compressed.**

Enter any public HuggingFace model ID below. The tool computes S₁ for every weight matrix
via SVD and fits the Entropy-Compression Law: D_min(ε) ≈ c(ε) · exp(α(ε) · S₁).

Small models (GPT-2, ~5 min) work in this Space. For 7B+ models, run locally:
`pip install entropy-lens && entropy-lens analyze <model>`
""")

    with gr.Row():
        model_input = gr.Textbox(
            value="openai-community/gpt2",
            label="HuggingFace Model ID",
            placeholder="e.g., openai-community/gpt2",
        )
        analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        plot_output = gr.Plot(label="Entropy Map")

    with gr.Row():
        summary_output = gr.Markdown(label="Summary")

    with gr.Row():
        table_output = gr.Dataframe(label="Per-layer results", interactive=False)

    analyze_btn.click(
        fn=analyze_model,
        inputs=[model_input],
        outputs=[plot_output, table_output, summary_output],
    )

if __name__ == "__main__":
    demo.launch()
