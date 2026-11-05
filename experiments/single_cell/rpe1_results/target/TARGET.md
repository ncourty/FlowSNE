# RPE1-FUCCI, d = 10: which asymmetry does the flow fit?

Dense row-stochastic input affinities; early exaggeration + cosine decay, 2000 iterations, seed 0. Target asymmetry = |log P_ij / P_ji| over each cell's 30 nearest pairs (median / q90 / q99); with no velocity: 1.56 / 5.19 / 11.74.

- direct: standard target 2.25 / 7.34 / 18.09; velocity-only target 1.59 / 5.20 / 11.05; D-J affinity on its support, eps = median edge 0.19 / 0.68 / 1.50, eps = their default 4.36 / 15.67 / 34.55
- kernel: standard target 2.51 / 7.71 / 16.43; velocity-only target 1.67 / 4.94 / 9.82; D-J affinity on its support, eps = median edge 0.19 / 0.60 / 1.26, eps = their default 4.68 / 14.53 / 30.41

Phase progress (fraction of a cycle) over the direction-blind walk, k = 30; seam = first and last twelfth of the cycle; circulation of the direction field on the map's own graph.

| velocity | fit / field | s=1 | s=5 | s=10 | s=1 seam | s=5 seam | s=10 seam | circulation | |ω| q10/50/90 | loss |
|---|---|---|---|---|---|---|---|---|---|---|
| direct | no velocity, learned field | +0.0041 | +0.0470 | +0.1019 | +0.0029 | -0.0496 | -0.1320 | +0.020 | 0.92/0.92/0.93 | 13.5438 |
| direct | no velocity, projected velocity | +0.0075 | +0.0310 | +0.0542 | +0.0220 | +0.0479 | +0.0694 | +0.374 | – | – |
| direct | standard target, learned field | +0.0063 | +0.0783 | +0.1525 | +0.0080 | -0.0042 | -0.0482 | +0.128 | 0.92/0.93/0.93 | 13.4772 |
| direct | standard target, projected velocity | +0.0078 | +0.0477 | +0.1070 | +0.0183 | +0.0384 | +0.0649 | +0.247 | – | – |
| direct | velocity-only target, learned field | -0.0023 | +0.0040 | +0.0109 | +0.0361 | +0.0509 | +0.0669 | +0.003 | 0.57/0.88/0.91 | 7.7637 |
| direct | velocity-only target, projected velocity | +0.0025 | +0.0084 | +0.0161 | +0.0091 | +0.0253 | +0.0375 | +0.031 | – | – |
| direct | D-J affinity target (eps = std), learned field | +0.0008 | +0.0090 | +0.0249 | -0.0152 | -0.0168 | -0.0069 | +0.043 | 0.00/0.01/0.02 | 7.6598 |
| direct | D-J affinity target (eps = std), projected velocity | +0.0089 | +0.0334 | +0.0671 | +0.0045 | +0.0222 | +0.0545 | +0.015 | – | – |
| direct | D-J full method, eps = std (spectral layout, V = L^a Psi) | +0.0047 | +0.0116 | +0.0154 | +0.0160 | +0.0105 | +0.0039 | +0.039 | – | – |
| direct | D-J affinity target (eps = median), learned field | +0.0045 | +0.0128 | +0.0194 | +0.0102 | +0.0153 | +0.0183 | +0.154 | 0.79/0.85/0.88 | 13.4434 |
| direct | D-J affinity target (eps = median), projected velocity | +0.0051 | +0.0220 | +0.0466 | +0.0200 | +0.0325 | +0.0507 | +0.340 | – | – |
| direct | D-J full method, eps = median (spectral layout, V = L^a Psi) | +0.0048 | +0.0278 | +0.0663 | +0.0171 | +0.0491 | +0.0830 | +0.345 | – | – |
| kernel | no velocity, learned field | +0.0041 | +0.0470 | +0.1019 | +0.0029 | -0.0496 | -0.1320 | +0.016 | 0.92/0.92/0.93 | 13.5438 |
| kernel | no velocity, projected velocity | +0.0229 | +0.0972 | +0.1758 | +0.0236 | +0.0724 | +0.1485 | +0.606 | – | – |
| kernel | standard target, learned field | +0.0206 | +0.1270 | +0.2712 | +0.0208 | +0.0735 | +0.1973 | +0.482 | 0.90/0.92/0.93 | 13.536 |
| kernel | standard target, projected velocity | +0.0239 | +0.1040 | +0.1898 | +0.0243 | +0.0559 | +0.1396 | +0.622 | – | – |
| kernel | velocity-only target, learned field | -0.0010 | +0.0033 | +0.0065 | +0.0209 | +0.0289 | +0.0352 | -0.002 | 0.65/0.89/0.91 | 7.7527 |
| kernel | velocity-only target, projected velocity | +0.0122 | +0.0773 | +0.1551 | +0.0206 | +0.0888 | +0.1687 | +0.041 | – | – |
| kernel | D-J affinity target (eps = std), learned field | +0.0019 | -0.0072 | -0.0289 | -0.0460 | -0.0591 | -0.0761 | -0.003 | 0.01/0.01/0.03 | 7.6556 |
| kernel | D-J affinity target (eps = std), projected velocity | +0.0136 | +0.0819 | +0.1594 | -0.0003 | +0.0478 | +0.1275 | +0.044 | – | – |
| kernel | D-J full method, eps = std (spectral layout, V = L^a Psi) | +0.0068 | +0.0282 | +0.0479 | +0.0218 | +0.0424 | +0.0877 | +0.159 | – | – |
| kernel | D-J affinity target (eps = median), learned field | +0.0126 | +0.0335 | +0.0553 | +0.0142 | +0.0339 | +0.0507 | +0.511 | 0.66/0.76/0.83 | 13.4455 |
| kernel | D-J affinity target (eps = median), projected velocity | +0.0171 | +0.0578 | +0.1053 | +0.0243 | +0.0585 | +0.0957 | +0.517 | – | – |
| kernel | D-J full method, eps = median (spectral layout, V = L^a Psi) | +0.0200 | +0.0888 | +0.1694 | +0.0222 | +0.0769 | +0.1548 | +0.583 | – | – |
