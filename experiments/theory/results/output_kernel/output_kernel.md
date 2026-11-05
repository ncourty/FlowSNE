# Output kernel: what the target drift captures

Subsample n = 2000 per dataset, 2-D embeddings, squared-cost input affinity (perplexity sqrt(n)), averaged drift on both sides. See `output_kernel.py`.

## Test 1a: first-order loss sensitivity at the symmetric fit (omega = 0)

Median |dL/dt| over random fields, gradient grad h vs rotated J grad h of the same rms norm. `net div` = sum_i |(N 1)_i| / sum |N_ij|, N = P - P^T.

| input | net div | Student: grad | rot | ratio | Laplace: grad | rot | ratio | Laplace, exact h_j - h_i |
|---|---|---|---|---|---|---|---|---|
| pancreas/DS | 0.000 | 1.85e-02 | 7.89e-02 | 0.235 | 1.13e-02 | 5.53e-02 | 0.204 | 6.1e-14 |
| pancreas/RS | 0.364 | 4.93e-02 | 1.07e-01 | 0.461 | 3.41e-02 | 8.20e-02 | 0.416 | 6.5e-05 |
| rpe1/DS | 0.000 | 1.00e-02 | 3.36e-02 | 0.298 | 4.18e-03 | 2.12e-02 | 0.197 | 6.8e-15 |
| rpe1/RS | 0.551 | 4.12e-02 | 2.35e-02 | 1.755 | 1.92e-02 | 2.26e-02 | 0.850 | 6.2e-05 |

## Test 1b: at the fitted FlowSNE model, L(omega* + t f) - L(omega*) = a t + b t^2

| input | grad: median abs a | median b | rot: median abs a | median b |
|---|---|---|---|---|
| pancreas/DS | 1.08e-02 | 9.29e-01 | 1.61e-02 | 1.08e+00 |
| pancreas/RS | 2.24e-02 | 2.61e+00 | 3.19e-02 | 3.30e+00 |
| rpe1/DS | 7.02e-03 | 6.54e-01 | 8.30e-03 | 7.72e-01 |
| rpe1/RS | 6.91e-02 | 2.59e+00 | 7.60e-02 | 3.37e+00 |

## Test 2: Hodge decomposition of the fitted flow direction on the input kNN graph

Means over 3 seeds; correlations are over edges (cyclic parts, full flows) or cells (potentials).

`mean` = share of the field's mean squared norm carried by its mean vector (a constant field is an exact gradient of c.z); `centred` = the same decomposition after removing it.

| input | mean | gradient share | centred | potential vs input potential | centred | potential vs pseudotime | cyclic vs input cyclic | full vs input full | potential across seeds | cyclic across seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| pancreas/DS (input share 0.81) | 0.01 | 0.82 | 0.82 | +0.98 | +0.98 | +0.99 | +0.70 | +0.54 | +1.00 | +0.90 |
| pancreas/RS (input share 0.81) | 0.06 | 0.89 | 0.88 | +1.00 | +0.99 | +0.99 | +0.51 | +0.70 | +1.00 | +0.86 |
| rpe1/DS (input share 0.67) | 0.02 | 0.33 | 0.33 | -0.28 | -0.10 | +nan | +0.27 | +0.08 | +0.81 | +0.95 |
| rpe1/RS (input share 0.67) | 0.09 | 0.85 | 0.79 | +0.55 | +0.41 | +nan | +0.33 | +0.30 | +0.86 | +0.69 |

## Test 3: target kernels

Loss = cross-entropy (mean over 3 seeds; symmetric: seed 0). Progress = walk on Q minus the symmetric fit's, s = 10 steps (pancreas: pseudotime; RPE1: fraction of a cycle).

| input | sym | flow | stream | weight | stream_weight |
|---|---|---|---|---|---|
| pancreas/DS loss | 12.4109 | 12.2733 | 12.2807 | 12.3815 | 12.2445 |
| pancreas/DS progress | +0.0000 | +0.0337 ± 0.0020 | +0.0368 ± 0.0049 | +0.0015 ± 0.0002 | +0.0274 ± 0.0018 |
| pancreas/RS loss | 12.7390 | 12.3819 | 12.4963 | 12.5267 | 12.2993 |
| pancreas/RS progress | +0.0000 | +0.0999 ± 0.0006 | +0.0867 ± 0.0013 | +0.0067 ± 0.0000 | +0.1038 ± 0.0004 |
| rpe1/DS loss | 13.4061 | 13.3732 | 13.3704 | 13.3713 | 13.3326 |
| rpe1/DS progress | +0.0000 | +0.0284 ± 0.0042 | +0.0292 ± 0.0011 | +0.0032 ± 0.0002 | +0.0344 ± 0.0007 |
| rpe1/RS loss | 13.6786 | 13.4178 | 13.5684 | 13.0156 | 12.9724 |
| rpe1/RS progress | +0.0000 | +0.1212 ± 0.0062 | +0.0584 ± 0.0016 | +0.0215 ± 0.0005 | +0.0931 ± 0.0072 |
