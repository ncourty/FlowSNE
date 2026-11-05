# Balanced vs row-stochastic Randers affinities on real velocity fields

See `balanced_vs_rs.py`. `cos` = mean cosine between each affinity's shift in mean outgoing displacement (drifted minus symmetric, same normalisation and cost power) and the velocity / its graph-Hodge gradient part / its remainder. `progress` = ground-truth progress of the 30-truncated walk minus the symmetric walk's (pancreas: pseudotime; RPE1: fraction of a cycle).

## pancreas (n = 3696, gradient share of the edge flow 0.832)

| affinity | cos velocity | cos gradient | cos rest | |shift| | progress s=1 | progress s=5 | progress s=10 | marginal err |
|---|---|---|---|---|---|---|---|---|
| DS_q1 | +0.796 | +0.437 | +0.240 | 2.03 | +0.0054 | +0.0137 | +0.0156 | 9.7e-10 |
| RS_q1 | +0.926 | +0.553 | +0.207 | 2.49 | +0.0181 | +0.0841 | +0.1480 | nan |
| DS_q2 | +0.729 | +0.381 | +0.262 | 1.61 | +0.0000 | -0.0052 | -0.0146 | 9.9e-10 |
| RS_q2 | +0.930 | +0.555 | +0.207 | 2.5 | +0.0184 | +0.0853 | +0.1504 | nan |

## rpe1_direct (n = 2793, gradient share of the edge flow 0.668)

| affinity | cos velocity | cos gradient | cos rest | |shift| | progress s=1 | progress s=5 | progress s=10 | marginal err |
|---|---|---|---|---|---|---|---|---|
| DS_q1 | +0.530 | -0.009 | +0.323 | 1.15 | +0.0079 | +0.0396 | +0.0800 | 8.7e-10 |
| RS_q1 | +0.578 | +0.553 | +0.135 | 1.69 | +0.0084 | +0.0434 | +0.0661 | nan |
| DS_q2 | +0.557 | +0.014 | +0.332 | 1.18 | +0.0077 | +0.0380 | +0.0768 | 8.3e-10 |
| RS_q2 | +0.586 | +0.564 | +0.135 | 1.66 | +0.0084 | +0.0431 | +0.0660 | nan |

## rpe1_kernel (n = 2793, gradient share of the edge flow 0.294)

| affinity | cos velocity | cos gradient | cos rest | |shift| | progress s=1 | progress s=5 | progress s=10 | marginal err |
|---|---|---|---|---|---|---|---|---|
| DS_q1 | +0.689 | +0.121 | +0.364 | 1.3 | +0.0247 | +0.1252 | +0.2517 | 8.6e-10 |
| RS_q1 | +0.754 | +0.335 | +0.353 | 1.42 | +0.0405 | +0.1769 | +0.3034 | nan |
| DS_q2 | +0.691 | +0.118 | +0.366 | 1.29 | +0.0228 | +0.1142 | +0.2275 | 8.2e-10 |
| RS_q2 | +0.768 | +0.342 | +0.357 | 1.4 | +0.0407 | +0.1782 | +0.3059 | nan |
