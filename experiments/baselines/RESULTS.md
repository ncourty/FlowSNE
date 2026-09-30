# Baseline: FlowSNE vs Finsler directed embedding (Debaussart-Joniec et al. 2026)

See `finsler_embedding_comparison.py` docstring. `cos_ref` = cosine between the estimated drift and the push-forward of the true input drift into that same embedding (points above the 20th percentile of reference speed). `speed ρ` = Spearman(|estimated|, |reference|). `knn` = 10-NN preservation of the layout. Mean ± std over seeds.

## sphere (n=1500, 3 seed(s))

| method | cos_ref | cos_ref_median | speed_spearman | knn | secs |
|---|---|---|---|---|---|
| theirs_isomap | +0.981 ± 0.002 | +0.996 ± 0.000 | +0.610 ± 0.066 | +0.480 ± 0.003 | 0 |
| theirs_eig | +0.927 ± 0.004 | +0.989 ± 0.002 | +0.775 ± 0.031 | +0.545 ± 0.011 | 1 |
| theirs_on_flowsne | +0.848 ± 0.011 | +0.959 ± 0.006 | +0.440 ± 0.028 | +0.680 ± 0.008 | 0 |
| flowsne_ds_joint | +0.953 ± 0.005 | +0.996 ± 0.001 | +0.853 ± 0.024 | +0.809 ± 0.014 | 629 |
| flowsne_ds_driftonly | +0.849 ± 0.004 | +0.983 ± 0.001 | +0.402 ± 0.022 | +0.680 ± 0.008 | 357 |
| flowsne_ds_on_isomap | +0.778 ± 0.029 | +0.971 ± 0.002 | -0.043 ± 0.038 | +0.480 ± 0.003 | 367 |
| flowsne_rs_joint | +0.950 ± 0.014 | +0.993 ± 0.001 | +0.758 ± 0.044 | +0.788 ± 0.007 | 598 |
| flowsne_rs_driftonly | +0.824 ± 0.022 | +0.961 ± 0.006 | +0.375 ± 0.049 | +0.681 ± 0.002 | 253 |
| flowsne_rs_on_isomap | +0.768 ± 0.032 | +0.973 ± 0.005 | -0.070 ± 0.022 | +0.480 ± 0.003 | 270 |

## swiss (n=1500, 3 seed(s))

| method | cos_ref | cos_ref_median | speed_spearman | knn | secs |
|---|---|---|---|---|---|
| theirs_isomap | +0.974 ± 0.001 | +0.991 ± 0.001 | +0.139 ± 0.040 | +0.860 ± 0.014 | 0 |
| theirs_eig | +0.999 ± 0.001 | +1.000 ± 0.000 | +0.661 ± 0.015 | +0.315 ± 0.031 | 1 |
| theirs_on_flowsne | +0.871 ± 0.019 | +0.968 ± 0.003 | +0.368 ± 0.090 | +0.664 ± 0.014 | 0 |
| flowsne_ds_joint | +0.981 ± 0.002 | +0.996 ± 0.001 | +0.411 ± 0.053 | +0.754 ± 0.011 | 631 |
| flowsne_ds_driftonly | +0.813 ± 0.028 | +0.967 ± 0.002 | +0.044 ± 0.115 | +0.664 ± 0.014 | 358 |
| flowsne_ds_on_isomap | +0.919 ± 0.012 | +0.995 ± 0.001 | +0.039 ± 0.023 | +0.860 ± 0.014 | 365 |
| flowsne_rs_joint | +0.994 ± 0.001 | +0.998 ± 0.000 | +0.441 ± 0.142 | +0.797 ± 0.008 | 599 |
| flowsne_rs_driftonly | +0.879 ± 0.037 | +0.976 ± 0.006 | +0.151 ± 0.063 | +0.633 ± 0.011 | 254 |
| flowsne_rs_on_isomap | +0.983 ± 0.001 | +0.994 ± 0.001 | -0.013 ± 0.020 | +0.860 ± 0.014 | 269 |

## pancreas (n=1112, 3 seed(s))

| method | cos_ref | cos_ref_median | speed_spearman | knn | cos_scvelo | pt_align | scvelo_vs_ref | secs |
|---|---|---|---|---|---|---|---|---|
| theirs_isomap | +0.808 ± 0.000 | +0.963 ± 0.000 | +0.629 ± 0.000 | +0.292 ± 0.000 | +0.689 ± 0.000 | +0.318 ± 0.000 | +0.960 ± 0.000 | 0 |
| theirs_eig | +0.653 ± 0.000 | +0.963 ± 0.000 | +0.797 ± 0.000 | +0.260 ± 0.000 | +0.570 ± 0.000 | +0.178 ± 0.000 | +0.847 ± 0.000 | 0 |
| theirs_on_flowsne | +0.869 ± 0.000 | +0.973 ± 0.000 | +0.496 ± 0.000 | +0.532 ± 0.000 | +0.771 ± 0.000 | +0.351 ± 0.000 | +0.940 ± 0.000 | 0 |
| flowsne_ds_joint | +0.694 ± 0.009 | +0.959 ± 0.002 | +0.703 ± 0.011 | +0.521 ± 0.001 | +0.537 ± 0.017 | +0.306 ± 0.030 | +0.965 ± 0.001 | 319 |
| flowsne_ds_driftonly | +0.709 ± 0.032 | +0.944 ± 0.009 | +0.577 ± 0.025 | +0.532 ± 0.000 | +0.590 ± 0.044 | +0.350 ± 0.021 | +0.940 ± 0.000 | 149 |
| flowsne_ds_on_isomap | +0.670 ± 0.003 | +0.951 ± 0.001 | +0.761 ± 0.002 | +0.292 ± 0.000 | +0.550 ± 0.014 | +0.379 ± 0.021 | +0.960 ± 0.000 | 148 |
| flowsne_rs_joint | +0.797 ± 0.003 | +0.949 ± 0.004 | +0.788 ± 0.004 | +0.504 ± 0.002 | +0.758 ± 0.006 | +0.336 ± 0.004 | +0.948 ± 0.001 | 336 |
| flowsne_rs_driftonly | +0.704 ± 0.006 | +0.887 ± 0.003 | +0.489 ± 0.028 | +0.513 ± 0.000 | +0.699 ± 0.003 | +0.380 ± 0.001 | +0.945 ± 0.000 | 142 |
| flowsne_rs_on_isomap | +0.784 ± 0.010 | +0.969 ± 0.004 | +0.768 ± 0.003 | +0.292 ± 0.000 | +0.648 ± 0.012 | +0.327 ± 0.003 | +0.960 ± 0.000 | 144 |

![embeddings](results/comparison.png)
