# RPE1-FUCCI cell cycle: which map keeps the direction around the cycle? (n = 2793)

Advance = mean FUCCI phase progress (unwrapped: summed per-step wrapped increments) after s steps of the cosine walk (fraction of a cycle), minus the same neighbours with uniform weights; seam = the 203 cells in the first and last twelfth of the cycle (M → G1). Circulation = mean net/abs of the map's own direction field around 200 loops through the 12 phase bins, on the map's own 30-NN graph (0 for any gradient). `_native` = FlowSNE's Randers walk vs the field switched off. Mean ± std over seeds.

| representation | s=1 | s=5 | s=10 | s=1, seam | s=5, seam | s=10, seam | circulation |
|---|---|---|---|---|---|---|---|
| direct/input30 | +0.0099 | +0.0442 | +0.0878 | +0.0353 | +0.0986 | +0.1626 | +0.1903 |
| direct/pca2 | +0.0045 | +0.0236 | +0.0430 | +0.0129 | +0.0418 | +0.0695 | +0.1786 |
| direct/sym2 | +0.0026 | +0.0146 | +0.0286 | +0.0152 | +0.0276 | +0.0408 | +0.1831 |
| direct/flowsne2 | +0.0020 ± 0.0004 | +0.0147 ± 0.0009 | +0.0366 ± 0.0031 | +0.0035 ± 0.0011 | +0.0015 ± 0.0076 | -0.0256 ± 0.0161 | +0.1690 ± 0.0178 |
| direct/flowsne2_native | +0.0035 ± 0.0002 | +0.0335 ± 0.0001 | +0.0780 ± 0.0005 | -0.0002 ± 0.0002 | -0.0265 ± 0.0118 | -0.0637 ± 0.0198 | – |
| direct/pca10 | +0.0095 | +0.0453 | +0.0911 | +0.0288 | +0.0839 | +0.1367 | +0.2074 |
| direct/sym10 | +0.0058 | +0.0284 | +0.0584 | +0.0181 | +0.0387 | +0.0573 | +0.0631 |
| direct/flowsne10 | +0.0047 ± 0.0004 | +0.0692 ± 0.0028 | +0.1670 ± 0.0021 | +0.0060 ± 0.0011 | +0.0098 ± 0.0049 | +0.0073 ± 0.0081 | +0.0306 ± 0.0533 |
| direct/flowsne10_native | +0.0147 ± 0.0002 | +0.0943 ± 0.0020 | +0.1578 ± 0.0050 | +0.0093 ± 0.0006 | +0.0084 ± 0.0022 | +0.0191 ± 0.0045 | – |
| kernel/input30 | +0.0338 | +0.1397 | +0.2359 | +0.0461 | +0.1540 | +0.3126 | +0.5033 |
| kernel/pca2 | +0.0086 | +0.0489 | +0.1004 | +0.0150 | +0.0558 | +0.1268 | +0.3755 |
| kernel/sym2 | +0.0081 | +0.0373 | +0.0713 | +0.0140 | +0.0412 | +0.0634 | +0.2064 |
| kernel/flowsne2 | +0.0088 ± 0.0005 | +0.0327 ± 0.0007 | +0.0609 ± 0.0089 | +0.0168 ± 0.0019 | +0.0295 ± 0.0059 | +0.0051 ± 0.0364 | +0.3782 ± 0.0180 |
| kernel/flowsne2_native | +0.0162 ± 0.0004 | +0.0803 ± 0.0013 | +0.1512 ± 0.0082 | +0.0164 ± 0.0020 | +0.0362 ± 0.0065 | +0.0486 ± 0.0101 | – |
| kernel/pca10 | +0.0327 | +0.1437 | +0.2555 | +0.0397 | +0.1573 | +0.3628 | +0.5615 |
| kernel/sym10 | +0.0172 | +0.0663 | +0.1205 | +0.0171 | +0.0383 | +0.0588 | +0.1655 |
| kernel/flowsne10 | +0.0205 ± 0.0002 | +0.1151 ± 0.0024 | +0.2319 ± 0.0047 | +0.0214 ± 0.0008 | +0.0588 ± 0.0008 | +0.0864 ± 0.0009 | +0.3473 ± 0.1911 |
| kernel/flowsne10_native | +0.0471 ± 0.0002 | +0.2160 ± 0.0010 | +0.3565 ± 0.0021 | +0.0306 ± 0.0003 | +0.0880 ± 0.0004 | +0.1973 ± 0.0007 | – |
