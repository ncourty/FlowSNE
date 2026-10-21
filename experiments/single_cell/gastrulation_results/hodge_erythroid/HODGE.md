# Hodge decomposition of the velocity (erythroid, n = 9815, 219886 edges, k = 30)

Edge flow f_ij = ½(v_i + v_j)·(x_j − x_i) on the symmetrised kNN graph in 30-PC space, split as f = grad φ + rest (curl + harmonic). Mean = share of the velocity carried by the vector common to all cells (a constant field is an exact gradient); `_centred` = that mean removed. Null = centred velocities permuted among cells.

| velocity | mean | gradient share | null | φ vs stage (Spearman) | time corr: full / gradient / rest | coherence: raw / full / gradient / rest |
|---|---|---|---|---|---|---|
| direct | 0.58 | 0.918 | 0.317 | -0.688 | +0.041 / +0.042 / +0.000 | 0.91 / 0.86 / 0.89 / 0.34 |
| scvelo_T | 0.07 | 0.654 | 0.246 | -0.382 | +0.027 / +0.034 / +0.000 | 0.72 / 0.54 / 0.70 / 0.35 |
| direct_centred | 0.00 | 0.767 | 0.316 | +0.317 | +0.054 / +0.061 / +0.000 | 0.73 / 0.63 / 0.66 / 0.34 |
| scvelo_T_centred | 0.00 | 0.623 | 0.241 | -0.330 | +0.032 / +0.040 / +0.000 | 0.70 / 0.53 / 0.68 / 0.35 |

## Walk: days gained over the direction-blind walk (30-PC space, cosine rule)

| velocity | part | s=1 | s=5 | s=25 |
|---|---|---|---|---|
| direct | raw velocity | +0.003 | +0.016 | +0.054 |
| direct | full | +0.003 | +0.014 | +0.038 |
| direct | gradient | +0.003 | +0.015 | +0.016 |
| direct | rest | +0.000 | +0.001 | +0.014 |
| scvelo_T | raw velocity | +0.004 | +0.016 | +0.005 |
| scvelo_T | full | +0.005 | +0.023 | +0.024 |
| scvelo_T | gradient | +0.006 | +0.010 | -0.013 |
| scvelo_T | rest | +0.002 | +0.021 | +0.066 |
| direct_centred | raw velocity | +0.010 | +0.045 | +0.147 |
| direct_centred | full | +0.008 | +0.040 | +0.139 |
| direct_centred | gradient | +0.009 | +0.040 | +0.096 |
| direct_centred | rest | +0.000 | +0.001 | +0.014 |
| scvelo_T_centred | raw velocity | +0.005 | +0.022 | +0.007 |
| scvelo_T_centred | full | +0.006 | +0.028 | +0.025 |
| scvelo_T_centred | gradient | +0.008 | +0.028 | +0.028 |
| scvelo_T_centred | rest | +0.002 | +0.021 | +0.066 |

## Gradient share on edges inside each cell type

| cell type | direct | scvelo_T | direct_centred | scvelo_T_centred |
|---|---|---|---|---|
| Blood progenitors 2 | 0.94 | 0.73 | 0.81 | 0.74 |
| Erythroid3 | 0.93 | 0.83 | 0.75 | 0.76 |
| Blood progenitors 1 | 0.91 | 0.64 | 0.82 | 0.66 |
| Erythroid1 | 0.91 | 0.35 | 0.73 | 0.36 |
| Erythroid2 | 0.85 | 0.57 | 0.63 | 0.50 |
