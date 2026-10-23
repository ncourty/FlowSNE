# RPE1-FUCCI cell cycle: which map keeps the direction around the cycle? (n = 2793)

Advance = mean FUCCI phase progress (unwrapped: summed per-step wrapped increments) after s steps of the cosine walk (fraction of a cycle), minus the same neighbours with uniform weights; seam = the 203 cells in the first and last twelfth of the cycle (M → G1). Circulation = mean net/abs of the map's own direction field around 200 loops through the 12 phase bins, on the map's own 30-NN graph (0 for any gradient). `_native` = FlowSNE's Randers walk vs the field switched off. Mean ± std over seeds.

| representation | s=1 | s=5 | s=10 | s=1, seam | s=5, seam | s=10, seam | circulation |
|---|---|---|---|---|---|---|---|
| direct/input30 | +0.0099 | +0.0442 | +0.0878 | +0.0353 | +0.0986 | +0.1626 | +0.1903 |
| direct/pca10 | +0.0095 | +0.0453 | +0.0911 | +0.0288 | +0.0839 | +0.1367 | +0.1867 |
| direct/sym10 | +0.0063 | +0.0289 | +0.0556 | +0.0169 | +0.0372 | +0.0571 | +0.3560 |
| direct/flowsne10 | +0.0060 ± 0.0003 | +0.0811 ± 0.0024 | +0.1658 ± 0.0212 | +0.0075 ± 0.0011 | -0.0015 ± 0.0033 | -0.0307 ± 0.0092 | +0.1325 ± 0.0079 |
| direct/flowsne10_native | +0.0135 ± 0.0003 | +0.0851 ± 0.0076 | +0.1376 ± 0.0168 | +0.0072 ± 0.0005 | -0.0071 ± 0.0090 | -0.0163 ± 0.0195 | – |
| kernel/input30 | +0.0338 | +0.1397 | +0.2359 | +0.0461 | +0.1540 | +0.3126 | +0.4988 |
| kernel/pca10 | +0.0327 | +0.1437 | +0.2555 | +0.0397 | +0.1573 | +0.3628 | +0.5555 |
| kernel/sym10 | +0.0176 | +0.0718 | +0.1378 | +0.0186 | +0.0551 | +0.1112 | +0.5854 |
| kernel/flowsne10 | +0.0207 ± 0.0002 | +0.1270 ± 0.0029 | +0.2605 ± 0.0167 | +0.0216 ± 0.0009 | +0.0587 ± 0.0091 | +0.1253 ± 0.0486 | +0.4697 ± 0.0089 |
| kernel/flowsne10_native | +0.0479 ± 0.0003 | +0.2223 ± 0.0016 | +0.3410 ± 0.0010 | +0.0284 ± 0.0005 | +0.0721 ± 0.0010 | +0.1481 ± 0.0039 | – |
