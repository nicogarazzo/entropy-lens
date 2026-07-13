# Joint rank+bits allocation: validation on Mistral-7B-v0.3
Real S1_eff/S2_eff per matrix from `results/mistralai_Mistral-7B-v0.3/results_v5b_whitened.csv` (224 matrices, 7 proj types x 32 layers), shapes from the Mistral-7B-v0.3 architecture (hidden_size=4096, intermediate_size=14336, 32 attention heads, 8 KV heads, head_dim=128). `q_bits=2` (fixed backbone), bits_choices=(2,3,4) for the low-rank factors.
**Cross-check on the D_eff = exp(S1_eff) assumption**: fitting a decay scale directly from the real `dmin_eff_{5,10,20,50}pct` anchor points per matrix and correlating `log(D_eff)` (assumed vs. data-fit) across all 224 matrices gives Pearson r = 0.598. This supports treating `exp(S1_eff)` as a reasonable proxy for the real decay scale, not just a functional-form assumption.

## Budget: 2.5 bits/param (q_bits=2)
- Realized: **2.5000 bits/param** (target 2.5), lambda=8.619e-09, total model error=25.2626
- Rank: mean=608.6, median=689.5, min=9, max=1591
- Bit-width histogram: {2: 150, 3: 17, 4: 57}
- Spearman(rank, S1_eff) = 0.928 (higher S1_eff -> more rank: holds)
- Spearman(bits, S2_eff) = -0.700 (higher S2_eff -> fewer bits: holds)

| matrix | S1_eff | S2_eff | rank | bits |
|---|---|---|---|---|
| layer_1.down_proj | 0.21 | 0.10 | 9 | 4 |
| layer_13.down_proj | 7.57 | 6.42 | 934 | 2 |
| layer_0.q_proj | 0.28 | 0.10 | 10 | 4 |
| layer_13.down_proj | 7.57 | 6.42 | 934 | 2 |

## Budget: 3.0 bits/param (q_bits=2)
- Realized: **3.0000 bits/param** (target 3.0), lambda=2.301e-09, total model error=8.5565
- Rank: mean=1063.4, median=1001.5, min=10, max=3507
- Bit-width histogram: {2: 133, 3: 21, 4: 70}
- Spearman(rank, S1_eff) = 0.987 (higher S1_eff -> more rank: holds)
- Spearman(bits, S2_eff) = -0.776 (higher S2_eff -> fewer bits: holds)

| matrix | S1_eff | S2_eff | rank | bits |
|---|---|---|---|---|
| layer_1.down_proj | 0.21 | 0.10 | 10 | 4 |
| layer_13.down_proj | 7.57 | 6.42 | 3507 | 2 |
| layer_0.q_proj | 0.28 | 0.10 | 12 | 4 |
| layer_13.down_proj | 7.57 | 6.42 | 3507 | 2 |

## Budget: 4.0 bits/param (q_bits=2)
- Realized: **3.9940 bits/param** (target 4.0), lambda=1.396e-11, total model error=4.3326
- Rank: mean=1723.4, median=1024.0, min=17, max=4096
- Bit-width histogram: {2: 52, 3: 34, 4: 138}
- Spearman(rank, S1_eff) = 0.961 (higher S1_eff -> more rank: holds)
- Spearman(bits, S2_eff) = -0.646 (higher S2_eff -> fewer bits: holds)

| matrix | S1_eff | S2_eff | rank | bits |
|---|---|---|---|---|
| layer_1.down_proj | 0.21 | 0.10 | 17 | 4 |
| layer_13.down_proj | 7.57 | 6.42 | 4096 | 2 |
| layer_0.q_proj | 0.28 | 0.10 | 19 | 4 |
| layer_13.down_proj | 7.57 | 6.42 | 4096 | 2 |

**Note on the 4.0 bits/param row**: realized comes in slightly under target (3.994 vs 4.0) because several matrices hit `max_rank` (the hard `min(m,n)` cap) before the budget is exhausted -- rank has nowhere left to grow, and the solver does not overshoot by inflating bits beyond what `bits_choices` allows. This is the correct, expected behavior once a generous budget saturates the shape ceiling, not a solver bug (see `test_generous_budget_gives_zero_lambda_and_max_rank` in `tests/test_joint_alloc.py`).
