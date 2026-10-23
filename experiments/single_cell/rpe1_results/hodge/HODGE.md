# RPE1-FUCCI cell cycle: Hodge decomposition of the velocity (n = 2793, 66017 edges, 273838 triangles, k = 30)

Velocity: dynamo's labelling-based velocity of total RNA, projected linearly on dynamo's 30-PC PCA (median share of its norm kept: 0.67); `dynamo_kernel` = dynamo's own kernel-based projection. Median phase difference between a cell and its 30-PC neighbours: 0.067 cycle.

| velocity | common mean | gradient | curl | harmonic | φ ~ phase R² | phase corr: full / grad / curl / harm | coherence: raw / grad / curl / harm |
|---|---|---|---|---|---|---|---|
| direct | 0.38 | 0.668 (null 0.214) | 0.330 (null 0.782) | 0.002 (null 0.004) | 0.02 | +0.056 / +0.011 / +0.081 / +0.021 | 0.79 / 0.86 / 0.29 / 0.41 |
| dynamo_kernel | 0.09 | 0.294 (null 0.240) | 0.698 (null 0.753) | 0.008 (null 0.007) | 0.20 | +0.256 / +0.178 / +0.186 / +0.042 | 0.58 / 0.73 / 0.39 / 0.47 |

Shares of ||f||²; null = centred velocities permuted among cells. Phase corr = correlation over edges between the part's edge values and the wrapped FUCCI phase difference.

## Walk: phase advance over the direction-blind walk (fraction of a cycle)

| velocity | part | s=1 | s=5 | s=25 |
|---|---|---|---|---|
| direct | raw velocity | +0.0099 | +0.0442 | +0.1854 |
| direct | full | +0.0053 | +0.0312 | +0.1287 |
| direct | gradient | -0.0014 | -0.0166 | -0.0575 |
| direct | curl | +0.0091 | +0.0544 | +0.2636 |
| direct | harmonic | +0.0091 | +0.0393 | +0.0502 |
| dynamo_kernel | raw velocity | +0.0338 | +0.1397 | +0.5263 |
| dynamo_kernel | full | +0.0328 | +0.1733 | +0.7927 |
| dynamo_kernel | gradient | +0.0290 | +0.1308 | +0.1946 |
| dynamo_kernel | curl | +0.0234 | +0.1416 | +0.7419 |
| dynamo_kernel | harmonic | +0.0240 | +0.0909 | +0.0661 |
