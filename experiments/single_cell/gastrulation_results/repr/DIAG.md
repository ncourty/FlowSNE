# FlowSNE fit diagnostics (full, n = 30000)

Loss = cross-entropy of the input Randers affinity against the output kernel (lower is better). Walk = days gained after 25 steps over the direction-blind walk (k = 30, cosine rule). cos = median cosine between the fitted field and the raw velocity projected onto the same layout.

| run | loss | loss, ω = 0 | loss, sym layout | |ω| q10/50/90/max | cos | walk |
|---|---|---|---|---|---|---|
| flowsne2_s0 | 17.07403 | 17.10101 | 17.10181 | 0.08/0.19/0.35/0.55 | +0.61 | +0.093 |
| flowsne10_s0 | 16.63963 | 16.96465 | 16.84850 | 0.91/0.93/0.94/0.94 | +0.39 | +0.034 |

## Continuing the joint fit (loss every 250 iterations)

- flowsne2_s0: 17.04731 → 17.03054 → 17.01831 → 17.00846 → 16.99803 → 16.99176 → 16.98621 → 16.98247 → 16.97822 → 16.97579 → 16.97316 → 16.97117; walk afterwards +0.147
- flowsne10_s0: 16.62903 → 16.61893 → 16.61287 → 16.60620 → 16.60155 → 16.59682 → 16.59369 → 16.59005 → 16.58754 → 16.58477 → 16.58268 → 16.58071; walk afterwards +0.038

## Flow refitted alone on the frozen layout (3000 iterations)

| run | network | loss | |ω| q10/50/90/max | cos | walk |
|---|---|---|---|---|---|
| flowsne2_s0 | 1x32 | 17.07252 | 0.08/0.19/0.36/0.60 | +0.64 | +0.114 |
| flowsne2_s0 | 2x128 | 17.06902 | 0.11/0.25/0.44/0.71 | +0.56 | +0.050 |
| flowsne10_s0 | 1x32 | 16.69144 | 0.90/0.92/0.93/0.94 | +0.36 | +0.030 |
| flowsne10_s0 | 2x128 | 16.63428 | 0.92/0.93/0.94/0.94 | +0.39 | +0.038 |
