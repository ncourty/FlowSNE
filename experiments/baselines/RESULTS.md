# Baseline: FlowSNE vs Finsler directed embedding (Debaussart-Joniec et al. 2026)

See `finsler_embedding_comparison.py` docstring. `cos_ref` = cosine between the estimated drift and the push-forward of the true input drift into that same embedding (points above the 20th percentile of reference speed). `speed ρ` = Spearman(|estimated|, |reference|). `knn` = 10-NN preservation of the layout, Euclidean in both spaces (direction-blind). `knn_dir` = overlap of the 10 nearest targets under the input Randers cost and under the method's own output cost (fitted / constant drift; Euclidean for the spectral estimator and symmetric fits). Mean ± std over seeds.

## sphere (n=1500, 3 seed(s))

| method | cos_ref | cos_ref_median | speed_spearman | knn | knn_dir | secs |
|---|---|---|---|---|---|---|
| theirs_isomap | +0.981 ± 0.002 | +0.996 ± 0.000 | +0.610 ± 0.066 | +0.480 ± 0.003 | +0.313 ± 0.005 | 0 |
| theirs_eig | +0.927 ± 0.004 | +0.989 ± 0.002 | +0.775 ± 0.031 | +0.545 ± 0.011 | +0.380 ± 0.004 | 1 |
| theirs_on_flowsne | +0.848 ± 0.011 | +0.959 ± 0.006 | +0.440 ± 0.028 | +0.680 ± 0.008 | +0.579 ± 0.001 | 0 |
| flowsne_ds_joint | +0.953 ± 0.005 | +0.996 ± 0.001 | +0.853 ± 0.024 | +0.809 ± 0.014 | +0.822 ± 0.016 | 629 |
| flowsne_ds_driftonly | +0.849 ± 0.004 | +0.983 ± 0.001 | +0.402 ± 0.022 | +0.680 ± 0.008 | +0.711 ± 0.008 | 357 |
| flowsne_ds_on_isomap | +0.778 ± 0.029 | +0.971 ± 0.002 | -0.043 ± 0.038 | +0.480 ± 0.003 | +0.345 ± 0.002 | 367 |
| flowsne_rs_joint | +0.950 ± 0.014 | +0.993 ± 0.001 | +0.758 ± 0.044 | +0.788 ± 0.007 | +0.801 ± 0.019 | 598 |
| flowsne_rs_driftonly | +0.824 ± 0.022 | +0.961 ± 0.006 | +0.375 ± 0.049 | +0.681 ± 0.002 | +0.715 ± 0.004 | 253 |
| flowsne_rs_on_isomap | +0.768 ± 0.032 | +0.973 ± 0.005 | -0.070 ± 0.022 | +0.480 ± 0.003 | +0.344 ± 0.002 | 270 |
| sym_ds | +0.000 ± 0.000 | +0.000 ± 0.000 | +nan ± nan | +0.680 ± 0.008 | +0.579 ± 0.001 | 278 |
| sym_rs | +0.000 ± 0.000 | +0.000 ± 0.000 | +nan ± nan | +0.681 ± 0.002 | +0.581 ± 0.002 | 308 |
| finsler_mds | +0.000 ± 0.000 | +0.000 ± 0.000 | +nan ± nan | +0.452 ± 0.006 | +0.379 ± 0.005 | 247 |
| finsler_tsne | +0.800 ± 0.012 | +0.997 ± 0.001 | +nan ± nan | +0.779 ± 0.017 | +0.794 ± 0.025 | 3183 |

## swiss (n=1500, 3 seed(s))

| method | cos_ref | cos_ref_median | speed_spearman | knn | knn_dir | secs |
|---|---|---|---|---|---|---|
| theirs_isomap | +0.974 ± 0.001 | +0.991 ± 0.001 | +0.139 ± 0.040 | +0.860 ± 0.014 | +0.363 ± 0.007 | 0 |
| theirs_eig | +0.999 ± 0.001 | +1.000 ± 0.000 | +0.661 ± 0.015 | +0.315 ± 0.031 | +0.125 ± 0.012 | 1 |
| theirs_on_flowsne | +0.871 ± 0.019 | +0.968 ± 0.003 | +0.368 ± 0.090 | +0.664 ± 0.014 | +0.461 ± 0.004 | 0 |
| flowsne_ds_joint | +0.981 ± 0.002 | +0.996 ± 0.001 | +0.411 ± 0.053 | +0.754 ± 0.011 | +0.698 ± 0.007 | 631 |
| flowsne_ds_driftonly | +0.813 ± 0.028 | +0.967 ± 0.002 | +0.044 ± 0.115 | +0.664 ± 0.014 | +0.627 ± 0.022 | 358 |
| flowsne_ds_on_isomap | +0.919 ± 0.012 | +0.995 ± 0.001 | +0.039 ± 0.023 | +0.860 ± 0.014 | +0.632 ± 0.014 | 365 |
| flowsne_rs_joint | +0.994 ± 0.001 | +0.998 ± 0.000 | +0.441 ± 0.142 | +0.797 ± 0.008 | +0.820 ± 0.008 | 599 |
| flowsne_rs_driftonly | +0.879 ± 0.037 | +0.976 ± 0.006 | +0.151 ± 0.063 | +0.633 ± 0.011 | +0.673 ± 0.017 | 254 |
| flowsne_rs_on_isomap | +0.983 ± 0.001 | +0.994 ± 0.001 | -0.013 ± 0.020 | +0.860 ± 0.014 | +0.698 ± 0.010 | 269 |
| sym_ds | +0.000 ± 0.000 | +0.000 ± 0.000 | +nan ± nan | +0.664 ± 0.014 | +0.461 ± 0.004 | 275 |
| sym_rs | +0.000 ± 0.000 | +0.000 ± 0.000 | +nan ± nan | +0.633 ± 0.011 | +0.507 ± 0.005 | 309 |
| finsler_mds | +0.997 ± 0.000 | +0.999 ± 0.000 | +nan ± nan | +0.937 ± 0.003 | +0.855 ± 0.000 | 162 |
| finsler_tsne | +0.912 ± 0.009 | +0.998 ± 0.001 | +nan ± nan | +0.830 ± 0.024 | +0.791 ± 0.028 | 3230 |

## cycle (n=1500, 3 seed(s))

| method | cos_ref | cos_ref_median | speed_spearman | knn | knn_dir | secs |
|---|---|---|---|---|---|---|
| flowsne_ds_joint | +0.951 ± 0.007 | +0.991 ± 0.001 | +0.626 ± 0.028 | +0.671 ± 0.002 | +0.530 ± 0.004 | 560 |
| flowsne_ds_driftonly | +0.915 ± 0.017 | +0.976 ± 0.004 | -0.249 ± 0.047 | +0.680 ± 0.002 | +0.501 ± 0.003 | 226 |
| flowsne_ds_on_isomap | +0.763 ± 0.027 | +0.960 ± 0.011 | +0.124 ± 0.051 | +0.512 ± 0.010 | +0.289 ± 0.009 | 230 |
| flowsne_rs_joint | +0.978 ± 0.001 | +0.992 ± 0.000 | +0.812 ± 0.025 | +0.668 ± 0.001 | +0.613 ± 0.005 | 569 |
| flowsne_rs_driftonly | +0.883 ± 0.005 | +0.932 ± 0.007 | -0.057 ± 0.046 | +0.622 ± 0.006 | +0.566 ± 0.006 | 238 |
| flowsne_rs_on_isomap | +0.812 ± 0.009 | +0.935 ± 0.009 | +0.300 ± 0.019 | +0.512 ± 0.010 | +0.290 ± 0.011 | 231 |
| sym_ds | +0.000 ± 0.000 | +0.000 ± 0.000 | +nan ± nan | +0.680 ± 0.002 | +0.341 ± 0.005 | 138 |
| sym_rs | +0.000 ± 0.000 | +0.000 ± 0.000 | +nan ± nan | +0.622 ± 0.006 | +0.412 ± 0.004 | 139 |
| finsler_mds | +0.251 ± 0.010 | +0.504 ± 0.020 | +nan ± nan | +0.559 ± 0.011 | +0.271 ± 0.013 | 79 |
| finsler_tsne | +0.785 ± 0.009 | +0.935 ± 0.005 | +nan ± nan | +0.661 ± 0.002 | +0.523 ± 0.003 | 1643 |
| theirs_isomap | +0.933 ± 0.016 | +0.989 ± 0.002 | +0.413 ± 0.019 | +0.512 ± 0.010 | +0.240 ± 0.011 | 0 |
| theirs_eig | +0.833 ± 0.012 | +1.000 ± 0.000 | +0.527 ± 0.036 | +0.396 ± 0.007 | +0.258 ± 0.010 | 0 |
| theirs_on_flowsne | +0.905 ± 0.000 | +0.968 ± 0.004 | +0.404 ± 0.038 | +0.680 ± 0.002 | +0.341 ± 0.005 | 0 |
| theirs_on_fmds | +0.912 ± 0.011 | +0.976 ± 0.003 | +0.304 ± 0.010 | +0.559 ± 0.011 | +0.262 ± 0.014 | 0 |
| theirs_on_ftsne | +0.865 ± 0.015 | +0.977 ± 0.002 | +0.405 ± 0.018 | +0.661 ± 0.002 | +0.328 ± 0.005 | 0 |

## pancreas (n=1112, 3 seed(s))

| method | cos_ref | cos_ref_median | speed_spearman | knn | knn_dir | cos_scvelo | pt_align | scvelo_vs_ref | secs |
|---|---|---|---|---|---|---|---|---|---|
| theirs_isomap | +0.808 ± 0.000 | +0.963 ± 0.000 | +0.629 ± 0.000 | +0.292 ± 0.000 | +0.214 ± 0.000 | +0.689 ± 0.000 | +0.318 ± 0.000 | +0.960 ± 0.000 | 0 |
| theirs_eig | +0.653 ± 0.000 | +0.963 ± 0.000 | +0.797 ± 0.000 | +0.260 ± 0.000 | +0.210 ± 0.000 | +0.570 ± 0.000 | +0.178 ± 0.000 | +0.847 ± 0.000 | 0 |
| theirs_on_flowsne | +0.883 ± 0.000 | +0.977 ± 0.000 | +0.500 ± 0.000 | +0.532 ± 0.000 | +0.389 ± 0.000 | +0.784 ± 0.000 | +0.344 ± 0.000 | +0.959 ± 0.000 | 0 |
| flowsne_ds_joint | +0.692 ± 0.011 | +0.958 ± 0.003 | +0.635 ± 0.010 | +0.522 ± 0.002 | +0.425 ± 0.000 | +0.535 ± 0.017 | +0.317 ± 0.025 | +0.964 ± 0.001 | 297 |
| flowsne_ds_driftonly | +0.721 ± 0.014 | +0.938 ± 0.011 | +0.538 ± 0.014 | +0.532 ± 0.000 | +0.421 ± 0.001 | +0.576 ± 0.021 | +0.350 ± 0.004 | +0.959 ± 0.000 | 128 |
| flowsne_ds_on_isomap | +0.630 ± 0.004 | +0.943 ± 0.002 | +0.754 ± 0.005 | +0.292 ± 0.000 | +0.247 ± 0.001 | +0.517 ± 0.004 | +0.369 ± 0.013 | +0.960 ± 0.000 | 128 |
| flowsne_rs_joint | +0.798 ± 0.003 | +0.949 ± 0.004 | +0.788 ± 0.005 | +0.504 ± 0.001 | +0.517 ± 0.002 | +0.758 ± 0.005 | +0.337 ± 0.005 | +0.948 ± 0.001 | 309 |
| flowsne_rs_driftonly | +0.710 ± 0.008 | +0.889 ± 0.010 | +0.499 ± 0.007 | +0.513 ± 0.000 | +0.491 ± 0.001 | +0.701 ± 0.004 | +0.384 ± 0.007 | +0.945 ± 0.000 | 135 |
| flowsne_rs_on_isomap | +0.786 ± 0.003 | +0.969 ± 0.002 | +0.769 ± 0.003 | +0.292 ± 0.000 | +0.275 ± 0.000 | +0.655 ± 0.008 | +0.333 ± 0.008 | +0.960 ± 0.000 | 144 |

![embeddings](results/comparison.png)

## Overnight 2026-11-17: warm start and perplexity (`report-extra`, full table in `results/logs/overnight/REPORT.txt`)

Warm start (`run-ws`: joint fit started from the saved symmetric layout, 3 seeds) is NOT a general improvement: it lowers the loss and rounds the ring on the noisy cycle only (11.32 vs 11.40 ds, 11.34 vs 11.42 rs; speed 0.71/0.83 vs 0.63/0.81); on the sphere it finds a worse optimum (loss 11.62 vs 11.58 ds; cos 0.91 vs 0.95, speed 0.71 vs 0.85) and on the balanced Swiss roll too (11.56 vs 11.52, speed 0.26 vs 0.41); RS Swiss roll is a wash (11.45 vs 11.47, speed 0.53 vs 0.44). Table 2 keeps the PCA init; the warm start stays a lever in the fold appendix.

Perplexity (seed 0, default sqrt(n) = 38.7): balanced Swiss roll gets WORSE at 60/100/150 (speed 0.26/0.30/0.33 vs 0.41) and at 10/20; row-stochastic Swiss roll peaks at 60 (speed 0.55, kNN 0.82, kNN-> 0.82 vs 0.44/0.80/0.82) and falls back at 100/150 (0.38/0.35). Sphere and cycle are flat in perplexity (sphere speed 0.84-0.86 ds across 20-100; cycle 0.56-0.61 ds, 0.76-0.81 rs). Nothing to change; sqrt(n) stays.
