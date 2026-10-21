# Gastrulation: which representation preserves direction? (full, n = 30000)

## Information checks

Share of the (log-space) velocity's norm inside the first 30 / 50 / 100 PCs (median over cells): 0.16 / 0.20 / 0.23. Cosine between scVelo's transition-matrix projection and the direct projection (median): +0.42.

Fraction of cells whose velocity points toward later-stage neighbours:

| velocity | forward |
|---|---|
| scvelo_T_projection_30pc | 45.2% |
| direct_30pc | 43.4% |
| direct_100pc | 45.6% |

Winnable cell types (direct velocity forward for ≥ 50% of their cells) cover 32.8% of cells.

## Days gained relative to each representation's own direction-blind control

Same chain rule for every representation (k nearest cells in it, weights exp(10 cos)); `_native` = FlowSNE's own Randers kernel. Mean ± std over seeds.

### k = 30

| representation | s=1 | s=5 | s=25 | s=1, winnable | s=5, winnable | s=25, winnable | fwd s=25 |
|---|---|---|---|---|---|---|---|
| input30 | +0.022 | +0.109 | +0.362 | +0.037 | +0.190 | +0.424 | +0.768 |
| pca2 | +0.004 | +0.018 | +0.114 | +0.004 | +0.020 | +0.146 | +0.555 |
| pca10 | +0.040 | +0.196 | +0.583 | +0.060 | +0.282 | +0.492 | +0.791 |
| sym2 | +0.007 | +0.036 | +0.127 | +0.015 | +0.081 | +0.286 | +0.560 |
| flowsne2 | +0.009 ± 0.001 | +0.033 ± 0.006 | +0.085 ± 0.031 | +0.016 ± 0.002 | +0.059 ± 0.009 | +0.110 ± 0.022 | +0.511 ± 0.035 |
| flowsne2_native | +0.003 ± 0.000 | +0.010 ± 0.000 | +0.031 ± 0.002 | +0.004 ± 0.000 | +0.017 ± 0.000 | +0.052 ± 0.003 | +0.475 ± 0.008 |
| sym10 | +0.013 | +0.043 | +0.103 | +0.024 | +0.094 | +0.223 | +0.525 |
| flowsne10 | +0.007 ± 0.000 | +0.028 ± 0.001 | +0.044 ± 0.007 | +0.010 ± 0.000 | +0.032 ± 0.002 | +0.057 ± 0.013 | +0.448 ± 0.012 |
| flowsne10_native | +0.009 ± 0.000 | +0.029 ± 0.001 | +0.059 ± 0.003 | +0.012 ± 0.000 | +0.043 ± 0.001 | +0.089 ± 0.002 | +0.516 ± 0.022 |

### k = 100

| representation | s=1 | s=5 | s=25 | s=1, winnable | s=5, winnable | s=25, winnable | fwd s=25 |
|---|---|---|---|---|---|---|---|
| input30 | +0.036 | +0.176 | +0.473 | +0.063 | +0.285 | +0.469 | +0.792 |
| pca2 | +0.007 | +0.029 | +0.284 | +0.006 | +0.031 | +0.264 | +0.687 |
| pca10 | +0.059 | +0.281 | +0.620 | +0.092 | +0.367 | +0.513 | +0.805 |
| sym2 | +0.014 | +0.061 | +0.229 | +0.035 | +0.156 | +0.366 | +0.641 |
| flowsne2 | +0.015 ± 0.001 | +0.055 ± 0.005 | +0.097 ± 0.012 | +0.030 ± 0.003 | +0.114 ± 0.012 | +0.138 ± 0.018 | +0.521 ± 0.018 |
| flowsne2_native | +0.005 ± 0.000 | +0.016 ± 0.001 | +0.042 ± 0.002 | +0.009 ± 0.000 | +0.030 ± 0.001 | +0.073 ± 0.004 | +0.521 ± 0.010 |
| sym10 | +0.022 | +0.071 | +0.193 | +0.044 | +0.162 | +0.302 | +0.618 |
| flowsne10 | +0.011 ± 0.001 | +0.043 ± 0.003 | +0.058 ± 0.006 | +0.017 ± 0.002 | +0.058 ± 0.003 | +0.075 ± 0.002 | +0.483 ± 0.013 |
| flowsne10_native | +0.011 ± 0.000 | +0.037 ± 0.001 | +0.073 ± 0.003 | +0.018 ± 0.001 | +0.062 ± 0.001 | +0.110 ± 0.004 | +0.562 ± 0.003 |

