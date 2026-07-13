# Per-layer naive-SVD ablation — Feynman falsable loop part (2)

Date: 2026-07-12. Script: `experiments/runpod_ablation.py`. Model: Mistral-7B-v0.3.
Method: for each of 32 layers, truncate its 7 weight matrices to 50% of their
own full rank via naive (Frobenius) SVD, measure damage = Δlog PPL on the
first 25 chunks of WikiText-2 test (baseline PPL 5.738), restore, next layer.
Correlate per-layer damage against four candidate predictors.

## Verdict: part (2) CONFIRMED, with nuance

Feynman predicted the gap `S1_raw − S1_eff` (how anisotropic a layer's weights
are relative to the data) predicts how much naive SVD hurts it. It does, and
it is the best of the four predictors:

| predictor | Pearson r (p) | Spearman r (p) |
|-----------|---------------|----------------|
| **s1_gap** (S1_raw − S1_eff) | **+0.475** (6e-3) | **+0.438** (1.2e-2) |
| s1_raw (original law, metric-blind) | −0.277 (0.12) | **+0.008** (0.97) |
| s1_eff | −0.497 (3.8e-3) | −0.375 (3.5e-2) |
| s2_eff | −0.340 (5.7e-2) | −0.469 (6.8e-3) |

**Money finding:** raw entropy — the predictor the original Frobenius-metric law
uses — has essentially zero rank correlation with compression damage (Spearman
+0.008). The gap, which can only be computed in the data metric (needs the
whitened spectrum), is the one that sees the damage. This is direct empirical
support for the mesa's thesis that compression loss lives in the data metric,
not the Frobenius metric.

`s1_eff` and `s2_eff` correlate *negatively*: low effective entropy paired with
high raw entropy is exactly a high gap, i.e. the regime where naive truncation
wastes rank on Frobenius-important but data-irrelevant directions. Coherent.

## Robustness (not in the raw JSON — computed from the CSV)

- **Controlling for depth**, the gap→damage link survives:
  partial Spearman(gap, damage | depth) = +0.359 (p=0.044);
  partial Spearman(gap, damage | edge-ness `|layer−15.5|`) = +0.419 (p=0.017).
  So it is not merely a depth artifact.
- **Excluding the layer-1 outlier** (n=31), the linear fit is very strong:
  Pearson +0.835 (p=5e-9); Spearman +0.383 (p=0.033).
- **Depth alone** is a weaker predictor than the gap: Pearson −0.226 (ns),
  Spearman +0.320 (p=0.075).

## Honest caveats (must be disclosed in the paper)

1. **Tail-dominated signal.** Within the bulk (layers 3–28) damage is nearly
   flat (Δlog ≈ 0.01–0.04) and no predictor discriminates (gap Spearman +0.198,
   ns). The relationship lives in the extremes: layers 0–1 and 29–31.
2. **One catastrophic layer.** Layer 1 alone jumps to PPL 25.3 (Δlog 1.48),
   ~30× any other layer. Four of the top-6 most-damaged layers (1, 31, 30, 29)
   are also top-5 by gap, but layer 1 dominates the raw Pearson.
3. **Single-layer robustness ≠ whole-model.** Almost every layer individually
   tolerates a 50%-rank naive truncation (Δlog < 0.05); the model routes around
   isolated damage. The v3/v4 whole-model collapse came from compressing all
   layers at once at an aggressive budget, not from any single layer.

## Takeaway for the north star

Compression sensitivity is sparse and localized, and where it exists it tracks
the data-metric gap, not raw entropy. This supports joint rank+bits allocation
that spends its budget defending the few critical layers (identified by high
gap / low effective entropy) rather than treating all layers uniformly.
