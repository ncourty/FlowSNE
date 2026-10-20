# Gastrulation Markov chains vs real stage (full, n = 30000)

Input velocity points forward in time for **67.0%** of cells (ceiling for any velocity-based kernel). Per cell type: Erythroid3 4%, Erythroid2 14%, Blood progenitors 2 30%, Erythroid1 33%, Primitive Streak 58%, Epiblast 61%, Forebrain/Midbrain/Hindbrain 66%, Spinal cord 67%, Cardiomyocytes 67%, Nascent mesoderm 68%, PGC 69%, Caudal neurectoderm 70%, Pharyngeal mesoderm 72%, Allantois 73%, Notochord 73%, Anterior Primitive Streak 73%, Mesenchyme 75%, Somitic mesoderm 76%, Endothelium 76%, ExE mesoderm 76%, Haematoendothelial progenitors 78%, Caudal Mesoderm 78%, Surface ectoderm 78%, Intermediate mesoderm 78%, Visceral endoderm 79%, Neural crest 79%, NMP 80%, Caudal epiblast 81%, Blood progenitors 1 81%, Gut 82%, Paraxial mesoderm 85%, Def. endoderm 87%, Rostral neurectoderm 87%, Mixed mesoderm 92%

Days gained after k steps (mean over cells), the same relative to the direction-blind symmetric control (Δ sym), and the fraction of cells moving forward:

| kernel | days k=1 | Δ sym k=1 | fwd k=1 | days k=5 | Δ sym k=5 | fwd k=5 | days k=25 | Δ sym k=25 | fwd k=25 |
|---|---|---|---|---|---|---|---|---|---|
| scvelo | +0.024 | +0.021 | 0.51 | +0.083 | +0.077 | 0.59 | +0.234 | +0.222 | 0.69 |
| input_randers | +0.031 | +0.029 | 0.52 | +0.132 | +0.125 | 0.62 | +0.333 | +0.321 | 0.71 |
| symmetric_d2 | +0.003 | +0.000 | 0.47 | +0.006 | +0.000 | 0.47 | +0.012 | +0.000 | 0.46 |
| flowsne_d2_nu1 | +0.012 | +0.009 | 0.48 | +0.044 | +0.038 | 0.52 | +0.155 | +0.143 | 0.62 |
| flowsne_d10_nu1 | +0.026 | +0.023 | 0.51 | +0.111 | +0.104 | 0.61 | +0.301 | +0.289 | 0.71 |
| flowsne_d10_nu9 | +0.027 | +0.024 | 0.52 | +0.112 | +0.105 | 0.62 | +0.304 | +0.292 | 0.72 |

Per-stage days gained at k = 5:

| kernel | E6.5 | E6.75 | E7 | E7.25 | E7.5 | E7.75 | E8 | E8.25 | E8.5 |
|---|---|---|---|---|---|---|---|---|---|
| scvelo | +0.648 | +0.428 | +0.216 | +0.059 | +0.237 | +0.278 | +0.094 | -0.038 | -0.212 |
| input_randers | +0.711 | +0.520 | +0.322 | +0.193 | +0.356 | +0.332 | +0.122 | -0.040 | -0.239 |
| symmetric_d2 | +0.484 | +0.289 | +0.090 | -0.105 | +0.036 | +0.159 | +0.019 | -0.038 | -0.166 |
| flowsne_d2_nu1 | +0.534 | +0.328 | +0.142 | -0.050 | +0.162 | +0.230 | +0.051 | -0.032 | -0.178 |
| flowsne_d10_nu1 | +0.662 | +0.438 | +0.242 | +0.097 | +0.292 | +0.317 | +0.123 | -0.018 | -0.201 |
| flowsne_d10_nu9 | +0.677 | +0.452 | +0.243 | +0.126 | +0.316 | +0.305 | +0.107 | -0.024 | -0.204 |
