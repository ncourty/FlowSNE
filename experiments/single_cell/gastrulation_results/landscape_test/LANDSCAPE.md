# Gastrulation: landscape (gradient) FlowSNE (erythroid, n = 1500)

Input potential ψ = Hodge projection of the direct velocity (30-PC kNN graph), scaled by α = 8.34 so that max |Δψ|/|Δx| over the affinity's candidate pairs = 0.95; that ratio's median / q90 / q99: 0.107 / 0.272 / 0.498. Input cost power 1. Inadmissible pairs among those kept: 0.0e+00.

## Pseudotime: Spearman correlation with embryonic stage

| landscape | ρ |
|---|---|
| input_psi (ceiling) | -0.695 |
| pca2 (post hoc) | -0.816 |
| pca10 (post hoc) | -0.728 |
| sym2 (post hoc) | -0.762 |
| vec2 (post hoc) | -0.640 |
| grad2 (post hoc) | -0.142 |
| grad2 (-h, fitted) | -0.142 |
| sym10 (post hoc) | -0.507 |
| vec10 (post hoc) | -0.535 |
| grad10 (post hoc) | +0.033 |
| grad10 (-h, fitted) | +0.034 |

Post hoc = the same Hodge projection applied to the representation's own positions and directions.

## Walk: days gained over each representation's own direction-blind control

`_native` = the fitted Randers kernel's own walk vs the same rule with the field off. `input_vec`/`input_grad` = the input affinities (truncated to k) vs their symmetric counterpart. Mean ± std over seeds.

### k = 30

| representation | s=1 | s=5 | s=25 | fwd s=25 |
|---|---|---|---|---|
| input30 | +0.005 | +0.002 | -0.099 | +0.510 |
| input30_grad | +0.004 | -0.012 | -0.171 | +0.233 |
| pca2 | +0.002 | -0.012 | -0.346 | +0.182 |
| pca10 | +0.002 | +0.009 | -0.033 | +0.510 |
| input_vec | +0.001 | -0.002 | -0.048 | +0.321 |
| input_grad | +0.006 | +0.006 | -0.115 | +0.252 |
| sym2 | +0.006 | +0.008 | -0.048 | +0.373 |
| vec2 | -0.002 | -0.012 | -0.064 | +0.163 |
| vec2_native | -0.002 | -0.008 | -0.018 | +0.320 |
| grad2 | +0.014 | +0.085 | +0.130 | +0.510 |
| grad2_native | +0.003 | +0.010 | +0.029 | +0.336 |
| sym10 | +0.010 | +0.022 | +0.007 | +0.442 |
| vec10 | -0.000 | +0.002 | +0.017 | +0.323 |
| vec10_native | +0.001 | +0.002 | +0.023 | +0.315 |
| grad10 | +0.009 | +0.032 | +0.028 | +0.337 |
| grad10_native | +0.011 | +0.025 | +0.016 | +0.374 |

## Fit checks

| fit | loss | loss, field off | gain | field norm q10/50/90/max | loss curve (every 500 its) |
|---|---|---|---|---|---|
| sym2_s0 | 12.92851 | 12.92851 | +0.00000 | 0.00/0.00/0.00/0.00 | 13.9542 |
| vec2_s0 | 12.93610 | 12.94782 | +0.01172 | 0.05/0.15/0.29/0.39 | 13.9596 |
| grad2_s0 | 13.05737 | 13.11665 | +0.05928 | 0.19/0.33/0.44/0.50 | 14.0236 |
| sym10_s0 | 12.41928 | 12.41928 | +0.00000 | 0.00/0.00/0.00/0.00 | 13.8077 |
| vec10_s0 | 12.28093 | 12.59988 | +0.31895 | 0.87/0.90/0.91/0.92 | 13.8221 |
| grad10_s0 | 12.41863 | 13.02630 | +0.60767 | 0.81/0.90/0.95/0.95 | 13.9228 |
