# FlowSNE in more than two dimensions

See `higher_dim.py`. Row-stochastic Randers input affinity; FlowSNE = flow fitted jointly, symmetric = same input affinity with zero flow. `knn` = 10-NN preservation; `cos_ref` / `speed ρ` = recovery of the true velocity pushed into the embedding.

## pancreas (n = 1112)

| d | ν | method | loss | knn | cos_ref | speed_spearman | cos_scvelo | pt_align | celltype_knn_acc | secs |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | symmetric | 11.576 | 0.515 | – | – | – | – | 0.880 | 141 |
| 2 | 1 | flowsne | 11.274 | 0.495 | 0.815 | 0.767 | 0.781 | 0.332 | 0.889 | 146 |
| 5 | 1 | symmetric | 11.468 | 0.604 | – | – | – | – | 0.877 | 151 |
| 5 | 1 | flowsne | 10.992 | 0.585 | 0.793 | 0.542 | 0.710 | 0.219 | 0.893 | 153 |
| 5 | 4 | symmetric | 11.129 | 0.651 | – | – | – | – | 0.897 | 152 |
| 5 | 4 | flowsne | 10.778 | 0.668 | 0.689 | 0.240 | 0.614 | 0.292 | 0.897 | 154 |
| 10 | 1 | symmetric | 11.452 | 0.612 | – | – | – | – | 0.879 | 156 |
| 10 | 1 | flowsne | 10.936 | 0.600 | 0.809 | 0.645 | 0.719 | 0.073 | 0.892 | 159 |
| 10 | 9 | symmetric | 10.960 | 0.719 | – | – | – | – | 0.897 | 160 |
| 10 | 9 | flowsne | 10.605 | 0.778 | 0.499 | -0.474 | 0.454 | 0.184 | 0.904 | 167 |

## sphere (n = 1500)

| d | ν | method | loss | knn | cos_ref | speed_spearman | secs |
|---|---|---|---|---|---|---|---|
| 2 | 1 | symmetric | 12.151 | 0.684 | – | – | 247 |
| 2 | 1 | flowsne | 11.605 | 0.793 | 0.940 | 0.742 | 262 |
| 5 | 1 | symmetric | 12.038 | 0.732 | – | – | 267 |
| 5 | 1 | flowsne | 11.278 | 0.868 | 0.975 | 0.846 | 272 |
| 5 | 4 | symmetric | 11.598 | 0.742 | – | – | 274 |
| 5 | 4 | flowsne | 11.017 | 0.936 | 0.913 | 0.937 | 255 |
| 10 | 1 | symmetric | 12.033 | 0.731 | – | – | 173 |
| 10 | 1 | flowsne | 11.256 | 0.862 | 0.972 | 0.839 | 187 |
| 10 | 9 | symmetric | 11.526 | 0.765 | – | – | 188 |
| 10 | 9 | flowsne | 10.982 | 0.938 | 0.841 | 0.886 | 186 |
