# Hodge decomposition of the velocity (full, n = 30000, 638333 edges, k = 30)

Edge flow f_ij = ½(v_i + v_j)·(x_j − x_i) on the symmetrised kNN graph in 30-PC space, split as f = grad φ + rest (curl + harmonic). Mean = share of the velocity carried by the vector common to all cells (a constant field is an exact gradient); `_centred` = that mean removed. Null = centred velocities permuted among cells.

| velocity | mean | gradient share | null | φ vs stage (Spearman) | time corr: full / gradient / rest | coherence: raw / full / gradient / rest |
|---|---|---|---|---|---|---|
| direct | 0.57 | 0.928 | 0.329 | +0.625 | +0.097 / +0.100 / +0.001 | 0.94 / 0.86 / 0.89 / 0.33 |
| scvelo_T | 0.10 | 0.784 | 0.306 | +0.662 | +0.135 / +0.152 / +0.001 | 0.80 / 0.55 / 0.65 / 0.34 |
| direct_centred | 0.00 | 0.825 | 0.329 | +0.635 | +0.057 / +0.062 / +0.001 | 0.79 / 0.65 / 0.69 / 0.33 |
| scvelo_T_centred | 0.00 | 0.746 | 0.306 | +0.666 | +0.127 / +0.146 / +0.001 | 0.77 / 0.52 / 0.61 / 0.34 |

## Walk: days gained over the direction-blind walk (30-PC space, cosine rule)

| velocity | part | s=1 | s=5 | s=25 |
|---|---|---|---|---|
| direct | raw velocity | +0.022 | +0.109 | +0.362 |
| direct | full | +0.019 | +0.097 | +0.343 |
| direct | gradient | +0.021 | +0.100 | +0.329 |
| direct | rest | -0.002 | -0.011 | +0.001 |
| scvelo_T | raw velocity | +0.029 | +0.137 | +0.475 |
| scvelo_T | full | +0.023 | +0.120 | +0.478 |
| scvelo_T | gradient | +0.027 | +0.135 | +0.473 |
| scvelo_T | rest | +0.000 | +0.012 | +0.111 |
| direct_centred | raw velocity | +0.015 | +0.088 | +0.328 |
| direct_centred | full | +0.012 | +0.072 | +0.292 |
| direct_centred | gradient | +0.014 | +0.081 | +0.269 |
| direct_centred | rest | -0.002 | -0.011 | +0.001 |
| scvelo_T_centred | raw velocity | +0.027 | +0.125 | +0.417 |
| scvelo_T_centred | full | +0.021 | +0.111 | +0.424 |
| scvelo_T_centred | gradient | +0.025 | +0.122 | +0.377 |
| scvelo_T_centred | rest | +0.000 | +0.012 | +0.111 |

## Gradient share on edges inside each cell type

| cell type | direct | scvelo_T | direct_centred | scvelo_T_centred |
|---|---|---|---|---|
| Blood progenitors 1 | 1.23 | 0.76 | 1.09 | 0.72 |
| Rostral neurectoderm | 1.15 | 1.43 | 1.29 | 1.46 |
| PGC | 1.13 | 0.92 | 0.99 | 0.85 |
| Surface ectoderm | 1.09 | 0.87 | 0.98 | 0.82 |
| Caudal neurectoderm | 1.03 | 1.26 | 0.96 | 1.21 |
| Forebrain/Midbrain/Hindbrain | 1.03 | 1.03 | 0.85 | 0.93 |
| Neural crest | 1.03 | 0.99 | 1.07 | 0.97 |
| Epiblast | 1.01 | 0.81 | 0.87 | 0.79 |
| Spinal cord | 1.00 | 0.97 | 0.84 | 0.89 |
| Anterior Primitive Streak | 0.98 | 0.59 | 0.89 | 0.55 |
| ExE mesoderm | 0.97 | 0.78 | 0.75 | 0.69 |
| Notochord | 0.96 | 0.94 | 0.99 | 0.94 |
| Endothelium | 0.95 | 0.75 | 0.92 | 0.71 |
| Erythroid3 | 0.95 | 0.73 | 0.94 | 0.71 |
| Allantois | 0.94 | 0.84 | 0.80 | 0.79 |
| Mesenchyme | 0.94 | 0.79 | 0.84 | 0.73 |
| Erythroid2 | 0.94 | 0.66 | 0.92 | 0.64 |
| Paraxial mesoderm | 0.93 | 0.66 | 0.76 | 0.56 |
| NMP | 0.93 | 0.53 | 0.73 | 0.47 |
| Blood progenitors 2 | 0.92 | 0.66 | 0.96 | 0.59 |
| Haematoendothelial progenitors | 0.92 | 0.87 | 0.86 | 0.83 |
| Erythroid1 | 0.92 | 0.72 | 0.95 | 0.67 |
| Gut | 0.92 | 0.61 | 0.75 | 0.51 |
| Caudal epiblast | 0.90 | 0.62 | 0.74 | 0.57 |
| Primitive Streak | 0.90 | 0.58 | 0.69 | 0.54 |
| Cardiomyocytes | 0.89 | 0.85 | 0.83 | 0.85 |
| Intermediate mesoderm | 0.89 | 0.62 | 0.66 | 0.56 |
| Def. endoderm | 0.88 | 0.80 | 0.79 | 0.78 |
| Pharyngeal mesoderm | 0.84 | 0.69 | 0.67 | 0.65 |
| Nascent mesoderm | 0.83 | 0.64 | 0.72 | 0.60 |
| Mixed mesoderm | 0.83 | 0.64 | 0.66 | 0.58 |
| Somitic mesoderm | 0.81 | 0.34 | 0.64 | 0.31 |
| Visceral endoderm | 0.80 | 0.39 | 0.69 | 0.36 |
| Caudal Mesoderm | 0.79 | 0.54 | 0.62 | 0.50 |
