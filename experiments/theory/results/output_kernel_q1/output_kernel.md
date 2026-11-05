# Output kernel: what the target drift captures

Subsample n = 2000 per dataset, 2-D embeddings, input affinity with cost power q = 1 (perplexity sqrt(n)), averaged drift on both sides. See `output_kernel.py`.

## Test 1a: first-order loss sensitivity at the symmetric fit (omega = 0)

Median |dL/dt| over random fields, gradient grad h vs rotated J grad h of the same rms norm. `net div` = sum_i |(N 1)_i| / sum |N_ij|, N = P - P^T.

| input | net div | Student: grad | rot | ratio | Laplace: grad | rot | ratio | Laplace, exact h_j - h_i |
|---|---|---|---|---|---|---|---|---|
| pancreas/DS | 0.000 | 2.10e-02 | 9.38e-02 | 0.224 | 2.55e-02 | 5.73e-02 | 0.446 | 4.6e-14 |
| pancreas/RS | 0.365 | 5.49e-02 | 1.06e-01 | 0.520 | 3.28e-02 | 8.23e-02 | 0.399 | 5.6e-05 |

## Test 1b: at the fitted FlowSNE model, L(omega* + t f) - L(omega*) = a t + b t^2

| input | grad: median abs a | median b | rot: median abs a | median b |
|---|---|---|---|---|
| pancreas/DS | 1.06e-02 | 1.34e+00 | 1.62e-02 | 1.53e+00 |
| pancreas/RS | 2.35e-02 | 2.54e+00 | 3.07e-02 | 3.29e+00 |

## Test 2: Hodge decomposition of the fitted flow direction on the input kNN graph

Means over 3 seeds; correlations are over edges (cyclic parts, full flows) or cells (potentials).

`mean` = share of the field's mean squared norm carried by its mean vector (a constant field is an exact gradient of c.z); `centred` = the same decomposition after removing it.

| input | mean | gradient share | centred | potential vs input potential | centred | potential vs pseudotime | cyclic vs input cyclic | full vs input full | potential across seeds | cyclic across seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| pancreas/DS (input share 0.81) | 0.01 | 0.79 | 0.80 | +0.99 | +0.99 | +0.99 | +0.66 | +0.70 | +1.00 | +0.86 |
| pancreas/RS (input share 0.81) | 0.05 | 0.89 | 0.89 | +1.00 | +1.00 | +0.99 | +0.51 | +0.70 | +1.00 | +0.82 |

## Test 3: target kernels

Loss = cross-entropy (mean over 3 seeds; symmetric: seed 0). Progress = walk on Q minus the symmetric fit's, s = 10 steps (pancreas: pseudotime; RPE1: fraction of a cycle).

| input | sym | flow | stream | weight | stream_weight |
|---|---|---|---|---|---|
| pancreas/DS loss | 12.5999 | 12.3990 | 12.4126 | 12.5691 | 12.3742 |
| pancreas/DS progress | +0.0000 | +0.0405 ± 0.0065 | +0.0461 ± 0.0009 | +0.0018 ± 0.0004 | +0.0322 ± 0.0040 |
| pancreas/RS loss | 12.7095 | 12.3809 | 12.4802 | 12.4947 | 12.2763 |
| pancreas/RS progress | +0.0000 | +0.1004 ± 0.0024 | +0.0877 ± 0.0032 | +0.0067 ± 0.0002 | +0.1029 ± 0.0032 |
