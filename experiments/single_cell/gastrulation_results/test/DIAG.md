# FlowSNE fit diagnostics (erythroid, n = 1500)

Loss = cross-entropy of the input Randers affinity against the output kernel (lower is better). Walk = days gained after 25 steps over the direction-blind walk (k = 30, cosine rule). cos = median cosine between the fitted field and the raw velocity projected onto the same layout.

| run | loss | loss, ω = 0 | loss, sym layout | |ω| q10/50/90/max | cos | walk |
|---|---|---|---|---|---|---|
| flowsne2_s0 | 13.17946 | 13.18610 | 13.18149 | 0.04/0.13/0.25/0.31 | +0.30 | -0.001 |
| flowsne10_s0 | 12.45385 | 12.72135 | 12.63672 | 0.85/0.88/0.90/0.91 | +0.58 | +0.018 |

## Continuing the joint fit (loss every 100 iterations)

- flowsne2_s0: 12.68518 → 12.46641; walk afterwards +0.009
- flowsne10_s0: 12.02213 → 11.89893; walk afterwards +0.023

## Flow refitted alone on the frozen layout (200 iterations)

| run | network | loss | |ω| q10/50/90/max | cos | walk |
|---|---|---|---|---|---|
| flowsne2_s0 | 1x32 | 13.17696 | 0.04/0.15/0.30/0.45 | +0.51 | -0.062 |
| flowsne2_s0 | 2x128 | 13.17547 | 0.06/0.18/0.36/0.49 | +0.47 | -0.051 |
| flowsne10_s0 | 1x32 | 12.44945 | 0.87/0.90/0.91/0.91 | +0.58 | +0.017 |
| flowsne10_s0 | 2x128 | 12.43288 | 0.91/0.92/0.93/0.93 | +0.58 | +0.019 |
