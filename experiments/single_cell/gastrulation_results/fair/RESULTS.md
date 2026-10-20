# Gastrulation Markov chains vs real stage — equal-support comparison (full, n = 30000)

Input velocity points forward in time for **67.0%** of cells; winnable cell types (velocity forward for ≥ 50% of their cells) cover 89.6% of cells. scVelo kernel's native row sizes: median 109, 90th percentile 151, max 237.

Every chain is truncated to the same k neighbours per row. Mean ± std over seeds; seeds only change FlowSNE's flow initialisation (layouts start from PCA), so scVelo, the input affinity and the symmetric control are deterministic. Δ sym = days gained minus the direction-blind symmetric control's; days/jump = days gained in one step per unit of mean step length (PCA space).

## k = 30 neighbours per row

| kernel | days/jump | Δ sym s=1 | Δ sym s=5 | Δ sym s=25 | Δ sym s=1, winnable | Δ sym s=5, winnable | Δ sym s=25, winnable | fwd s=25 |
|---|---|---|---|---|---|---|---|---|
| scvelo | +0.007 | +0.041 | +0.154 | +0.386 | +0.049 | +0.184 | +0.465 | +0.757 |
| input_randers | +0.006 | +0.031 | +0.138 | +0.368 | +0.036 | +0.164 | +0.439 | +0.757 |
| symmetric_d2 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | -0.000 ± 0.000 | +0.464 ± 0.000 |
| flowsne_d2_nu1 | +0.002 ± 0.000 | +0.009 ± 0.000 | +0.036 ± 0.002 | +0.131 ± 0.010 | +0.010 ± 0.000 | +0.042 ± 0.002 | +0.159 ± 0.012 | +0.617 ± 0.005 |
| flowsne_d10_nu1 | +0.005 ± 0.000 | +0.024 ± 0.000 | +0.105 ± 0.001 | +0.301 ± 0.014 | +0.029 ± 0.000 | +0.127 ± 0.001 | +0.371 ± 0.015 | +0.721 ± 0.009 |
| flowsne_d10_nu9 | +0.005 ± 0.000 | +0.025 ± 0.000 | +0.105 ± 0.002 | +0.290 ± 0.002 | +0.029 ± 0.001 | +0.125 ± 0.002 | +0.352 ± 0.003 | +0.719 ± 0.002 |

## k = 100 neighbours per row

| kernel | days/jump | Δ sym s=1 | Δ sym s=5 | Δ sym s=25 | Δ sym s=1, winnable | Δ sym s=5, winnable | Δ sym s=25, winnable | fwd s=25 |
|---|---|---|---|---|---|---|---|---|
| scvelo | +0.005 | +0.025 | +0.087 | +0.248 | +0.029 | +0.104 | +0.300 | +0.693 |
| input_randers | +0.006 | +0.030 | +0.134 | +0.346 | +0.035 | +0.159 | +0.413 | +0.740 |
| symmetric_d2 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.477 ± 0.000 |
| flowsne_d2_nu1 | +0.002 ± 0.000 | +0.013 ± 0.000 | +0.052 ± 0.002 | +0.166 ± 0.008 | +0.016 ± 0.000 | +0.063 ± 0.003 | +0.207 ± 0.009 | +0.631 ± 0.004 |
| flowsne_d10_nu1 | +0.005 ± 0.000 | +0.028 ± 0.000 | +0.122 ± 0.001 | +0.314 ± 0.011 | +0.034 ± 0.000 | +0.148 ± 0.000 | +0.385 ± 0.012 | +0.733 ± 0.002 |
| flowsne_d10_nu9 | +0.005 ± 0.000 | +0.027 ± 0.001 | +0.114 ± 0.002 | +0.298 ± 0.002 | +0.032 ± 0.001 | +0.135 ± 0.002 | +0.359 ± 0.002 | +0.714 ± 0.003 |

## k = 300 neighbours per row

| kernel | days/jump | Δ sym s=1 | Δ sym s=5 | Δ sym s=25 | Δ sym s=1, winnable | Δ sym s=5, winnable | Δ sym s=25, winnable | fwd s=25 |
|---|---|---|---|---|---|---|---|---|
| scvelo | +0.004 | +0.021 | +0.078 | +0.220 | +0.025 | +0.093 | +0.268 | +0.687 |
| input_randers | +0.005 | +0.029 | +0.128 | +0.324 | +0.035 | +0.152 | +0.388 | +0.721 |
| symmetric_d2 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | -0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.492 ± 0.000 |
| flowsne_d2_nu1 | +0.003 ± 0.000 | +0.019 ± 0.000 | +0.070 ± 0.003 | +0.186 ± 0.009 | +0.022 ± 0.000 | +0.086 ± 0.003 | +0.233 ± 0.010 | +0.649 ± 0.003 |
| flowsne_d10_nu1 | +0.005 ± 0.000 | +0.032 ± 0.000 | +0.131 ± 0.001 | +0.322 ± 0.006 | +0.039 ± 0.000 | +0.158 ± 0.000 | +0.391 ± 0.008 | +0.739 ± 0.005 |
| flowsne_d10_nu9 | +0.005 ± 0.000 | +0.029 ± 0.001 | +0.120 ± 0.002 | +0.306 ± 0.002 | +0.034 ± 0.001 | +0.142 ± 0.002 | +0.367 ± 0.003 | +0.706 ± 0.001 |

## Per cell type, k = 30, 5 steps: days gained relative to the control

| cell type | velocity forward | scvelo | input_randers | flowsne_d2_nu1 | flowsne_d10_nu1 | flowsne_d10_nu9 |
|---|---|---|---|---|---|---|
| Erythroid3 | 4% | -0.123 | -0.113 | -0.032 | -0.120 | -0.095 |
| Erythroid2 | 14% | -0.147 | -0.138 | -0.073 | -0.150 | -0.135 |
| Blood progenitors 2 | 30% | -0.074 | -0.048 | +0.010 | -0.069 | -0.042 |
| Erythroid1 | 33% | -0.093 | -0.060 | -0.012 | -0.065 | -0.053 |
| Primitive Streak | 58% | +0.180 | +0.201 | +0.035 | +0.110 | +0.155 |
| Epiblast | 61% | +0.189 | +0.170 | +0.026 | +0.131 | +0.142 |
| Forebrain/Midbrain/Hindbrain | 66% | +0.009 | +0.000 | +0.013 | -0.010 | -0.015 |
| Spinal cord | 67% | +0.072 | +0.061 | +0.056 | +0.082 | +0.053 |
| Cardiomyocytes | 67% | +0.077 | +0.093 | +0.025 | +0.070 | +0.073 |
| Nascent mesoderm | 68% | +0.364 | +0.316 | +0.065 | +0.246 | +0.233 |
| PGC | 69% | +0.206 | +0.202 | -0.035 | +0.009 | +0.166 |
| Caudal neurectoderm | 70% | +0.324 | +0.262 | +0.044 | +0.285 | +0.221 |
| Pharyngeal mesoderm | 72% | +0.042 | +0.042 | -0.035 | -0.007 | +0.006 |
| Allantois | 73% | +0.119 | +0.168 | +0.043 | +0.153 | +0.111 |
| Notochord | 73% | +0.105 | +0.083 | +0.056 | +0.080 | +0.057 |
| Anterior Primitive Streak | 73% | +0.389 | +0.454 | +0.067 | +0.341 | +0.348 |
| Mesenchyme | 75% | +0.163 | +0.140 | +0.029 | +0.088 | +0.070 |
| Somitic mesoderm | 76% | +0.108 | +0.084 | +0.020 | +0.080 | +0.068 |
| Endothelium | 76% | +0.064 | +0.058 | +0.003 | +0.021 | +0.016 |
| ExE mesoderm | 76% | +0.108 | +0.086 | +0.018 | +0.082 | +0.051 |
| Haematoendothelial progenitors | 78% | +0.241 | +0.226 | +0.068 | +0.177 | +0.178 |
| Caudal Mesoderm | 78% | +0.218 | +0.169 | +0.010 | +0.137 | +0.143 |
| Surface ectoderm | 78% | +0.089 | +0.077 | +0.055 | +0.030 | +0.057 |
| Intermediate mesoderm | 78% | +0.198 | +0.148 | +0.039 | +0.157 | +0.123 |
| Visceral endoderm | 79% | +0.284 | +0.295 | +0.130 | +0.236 | +0.228 |
| Neural crest | 79% | -0.041 | -0.087 | +0.009 | +0.011 | -0.088 |
| NMP | 80% | +0.025 | +0.026 | +0.011 | +0.027 | +0.018 |
| Caudal epiblast | 81% | +0.339 | +0.299 | +0.061 | +0.292 | +0.260 |
| Blood progenitors 1 | 81% | +0.175 | +0.171 | +0.039 | +0.121 | +0.127 |
| Gut | 82% | +0.103 | +0.042 | +0.048 | -0.017 | -0.008 |
| Paraxial mesoderm | 85% | +0.050 | +0.014 | -0.009 | -0.014 | -0.024 |
| Def. endoderm | 87% | +0.438 | +0.382 | +0.113 | +0.289 | +0.340 |
| Rostral neurectoderm | 87% | +0.364 | +0.301 | +0.122 | +0.312 | +0.264 |
| Mixed mesoderm | 92% | +0.621 | +0.550 | +0.248 | +0.427 | +0.500 |

