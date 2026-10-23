# RPE1-FUCCI: does the velocity circulate around the cycle?

300 closed loops per graph, each through one random cell per FUCCI phase bin (12 bins), joined by shortest kNN-graph paths. net/abs = sum of the edge flow along the loop / sum of its absolute values; a gradient gives 0 on every loop. Null: centred velocities permuted among cells (200 times, same loops).

| graph | flow | mean net/abs | loops with net > 0 | null | z | p (one-sided) |
|---|---|---|---|---|---|---|
| k10 | ideal_phase_flow (reference) | +0.358 | 95% | – | – | – |
| k10 | V_pca_direct | +0.178 | 90% | +0.001 ± 0.035 | +5.1 | 0.005 |
| k10 | V_pca_kernel | +0.500 | 99% | -0.001 ± 0.029 | +17.2 | 0.005 |
| k30 | ideal_phase_flow (reference) | +0.411 | 91% | – | – | – |
| k30 | V_pca_direct | +0.181 | 85% | +0.001 ± 0.029 | +6.3 | 0.005 |
| k30 | V_pca_kernel | +0.490 | 97% | -0.001 ± 0.027 | +18.1 | 0.005 |

## Per segment: net / abs flow from phase bin b to b+1 (summed over loops)

Bin b covers FUCCI positions [b/12, (b+1)/12); the last segment closes the loop across the M → G1 seam.

| graph | flow | 0→1 | 1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 6→7 | 7→8 | 8→9 | 9→10 | 10→11 | 11→0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k10 | ideal_phase_flow | +0.28 | +0.48 | +0.34 | +0.36 | +0.36 | +0.41 | +0.43 | +0.39 | +0.34 | +0.28 | +0.29 | +0.23 |
| k10 | V_pca_direct | +0.66 | +0.55 | +0.39 | -0.01 | -0.09 | -0.05 | +0.06 | +0.07 | -0.11 | +0.14 | +0.20 | -0.10 |
| k10 | V_pca_kernel | +0.79 | +0.82 | +0.63 | +0.10 | +0.03 | +0.26 | +0.32 | +0.17 | +0.23 | +0.59 | +0.08 | +0.64 |
| k30 | ideal_phase_flow | +0.24 | +0.41 | +0.39 | +0.44 | +0.40 | +0.52 | +0.56 | +0.51 | +0.44 | +0.35 | +0.35 | +0.15 |
| k30 | V_pca_direct | +0.68 | +0.71 | +0.46 | -0.12 | -0.08 | -0.05 | +0.02 | +0.14 | +0.04 | +0.09 | +0.18 | -0.31 |
| k30 | V_pca_kernel | +0.68 | +0.89 | +0.70 | +0.15 | +0.05 | +0.21 | +0.29 | +0.29 | +0.35 | +0.52 | +0.03 | +0.57 |
