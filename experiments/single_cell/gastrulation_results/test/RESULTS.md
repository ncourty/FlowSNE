# Gastrulation: which representation preserves direction? (erythroid, n = 1500)

## Information checks

Share of the (log-space) velocity's norm inside the first 30 / 50 / 100 PCs (median over cells): 0.09 / 0.10 / 0.13. Cosine between scVelo's transition-matrix projection and the direct projection (median): +0.27.

Fraction of cells whose velocity points toward later-stage neighbours:

| velocity | forward |
|---|---|
| scvelo_T_projection_30pc | 32.7% |
| direct_30pc | 32.1% |
| direct_100pc | 33.9% |

Winnable cell types (direct velocity forward for ≥ 50% of their cells) cover 18.6% of cells.

## Days gained relative to each representation's own direction-blind control

Same chain rule for every representation (k nearest cells in it, weights exp(10 cos)); `_native` = FlowSNE's own Randers kernel. Mean ± std over seeds.

### k = 30

| representation | s=1 | s=5 | s=25 | s=1, winnable | s=5, winnable | s=25, winnable | fwd s=25 |
|---|---|---|---|---|---|---|---|
| input30 | +0.005 | +0.002 | -0.099 | +0.019 | +0.045 | -0.071 | +0.510 |
| pca2 | +0.002 | -0.012 | -0.346 | +0.025 | +0.049 | -0.276 | +0.182 |
| pca10 | +0.002 | +0.009 | -0.033 | +0.025 | +0.111 | +0.006 | +0.510 |
| sym2 | +0.000 | +0.002 | +0.006 | -0.007 | -0.009 | +0.006 | +0.459 |
| flowsne2 | -0.005 | -0.012 | -0.001 | +0.006 | +0.011 | +0.105 | +0.261 |
| flowsne2_native | -0.003 | -0.010 | -0.014 | -0.004 | -0.010 | +0.003 | +0.235 |
| sym10 | +0.010 | +0.018 | -0.011 | +0.019 | +0.043 | +0.011 | +0.533 |
| flowsne10 | -0.001 | +0.002 | +0.018 | +0.004 | +0.026 | +0.038 | +0.323 |
| flowsne10_native | +0.002 | +0.002 | +0.021 | +0.013 | +0.029 | +0.048 | +0.286 |
| input_randers | +0.001 | -0.002 | -0.048 | +0.011 | +0.025 | -0.022 | +0.321 |
| flowsne2_proj | +0.004 | -0.008 | -0.113 | +0.004 | -0.011 | -0.024 | +0.241 |
| flowsne10_proj | +0.001 | +0.013 | +0.047 | +0.022 | +0.091 | +0.094 | +0.546 |

