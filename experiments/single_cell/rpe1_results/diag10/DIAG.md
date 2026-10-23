# RPE1-FUCCI, d = 10: field or layout at the M → G1 seam?

Advance = FUCCI phase progress (unwrapped) (fraction of a cycle) over the direction-blind walk, k = 30; seam = first and last twelfth of the cycle. Circulation of the direction field on the map's own graph. Seam closure = share of the seam cells' 15 nearest cells lying on the other side of the seam (30-PC space: 0.010). Step 1: saved 2000-iteration fits; step 2: refits with early exaggeration (12×, 250 its) + cosine decay; step 3: flow alone refitted on the frozen FlowSNE-10 layout (seed 0), cosine decay.

| velocity | step | map / field | s=1 | s=5 | s=10 | s=1 seam | s=5 seam | s=10 seam | circulation | seam closure | |ω| q10/50/90 | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | 1 | input30 (velocity) | +0.0099 | +0.0442 | +0.0878 | +0.0353 | +0.0986 | +0.1626 | +0.190 | 0.010 | – | – |
| direct | 1 | sym10, projected velocity | +0.0058 | +0.0284 | +0.0584 | +0.0181 | +0.0387 | +0.0573 | +0.053 | 0.006 | – | – |
| direct | 1 | flowsne10 s0, learned field | +0.0043 | +0.0704 | +0.1699 | +0.0049 | +0.0030 | -0.0042 | -0.001 | 0.004 | 0.94/0.94/0.94 | 13.2107 |
| direct | 1 | flowsne10 s0, projected velocity | +0.0092 | +0.0455 | +0.0931 | +0.0226 | +0.0420 | +0.0583 | -0.031 | 0.004 | – | – |
| direct | 1 | flowsne10 s1, learned field | +0.0053 | +0.0718 | +0.1663 | +0.0075 | +0.0137 | +0.0123 | -0.011 | 0.005 | 0.94/0.94/0.94 | 13.2145 |
| direct | 1 | flowsne10 s1, projected velocity | +0.0085 | +0.0441 | +0.0951 | +0.0206 | +0.0425 | +0.0580 | -0.085 | 0.005 | – | – |
| direct | 1 | flowsne10 s2, learned field | +0.0044 | +0.0653 | +0.1649 | +0.0057 | +0.0127 | +0.0137 | +0.111 | 0.004 | 0.94/0.94/0.94 | 13.2102 |
| direct | 1 | flowsne10 s2, projected velocity | +0.0087 | +0.0419 | +0.0864 | +0.0212 | +0.0409 | +0.0563 | +0.196 | 0.004 | – | – |
| kernel | 1 | input30 (velocity) | +0.0338 | +0.1397 | +0.2359 | +0.0461 | +0.1540 | +0.3126 | +0.534 | 0.010 | – | – |
| kernel | 1 | sym10, projected velocity | +0.0172 | +0.0663 | +0.1205 | +0.0171 | +0.0383 | +0.0588 | +0.164 | 0.006 | – | – |
| kernel | 1 | flowsne10 s0, learned field | +0.0202 | +0.1121 | +0.2268 | +0.0220 | +0.0583 | +0.0856 | +0.521 | 0.005 | 0.93/0.94/0.94 | 13.259 |
| kernel | 1 | flowsne10 s0, projected velocity | +0.0235 | +0.0972 | +0.1634 | +0.0233 | +0.0530 | +0.0816 | +0.661 | 0.005 | – | – |
| kernel | 1 | flowsne10 s1, learned field | +0.0208 | +0.1180 | +0.2381 | +0.0203 | +0.0598 | +0.0877 | +0.094 | 0.005 | 0.93/0.94/0.94 | 13.2579 |
| kernel | 1 | flowsne10 s1, projected velocity | +0.0234 | +0.0980 | +0.1696 | +0.0220 | +0.0499 | +0.0798 | +0.119 | 0.005 | – | – |
| kernel | 1 | flowsne10 s2, learned field | +0.0206 | +0.1152 | +0.2308 | +0.0220 | +0.0582 | +0.0858 | +0.418 | 0.005 | 0.93/0.94/0.94 | 13.2561 |
| kernel | 1 | flowsne10 s2, projected velocity | +0.0243 | +0.1016 | +0.1811 | +0.0239 | +0.0509 | +0.0809 | +0.521 | 0.005 | – | – |
| direct | 2 | sym10 EE+cosine, projected velocity | +0.0063 | +0.0284 | +0.0546 | +0.0169 | +0.0369 | +0.0565 | +0.342 | 0.010 | – | – |
| kernel | 2 | sym10 EE+cosine, projected velocity | +0.0175 | +0.0715 | +0.1373 | +0.0184 | +0.0548 | +0.1108 | +0.602 | 0.010 | – | – |
| direct | 2 | flowsne10 EE+cosine, flow lr = lr, learned field | +0.0064 | +0.0794 | +0.1567 | +0.0084 | -0.0012 | -0.0435 | +0.128 | 0.006 | 0.92/0.93/0.93 | 13.4921 |
| direct | 2 | flowsne10 EE+cosine, flow lr = lr, projected velocity | +0.0080 | +0.0471 | +0.1063 | +0.0175 | +0.0382 | +0.0646 | +0.217 | 0.006 | – | – |
| direct | 2 | flowsne10 EE+cosine, flow lr = lr/10, learned field | +0.0073 | +0.0834 | +0.1568 | +0.0090 | -0.0001 | +0.0145 | +0.128 | 0.003 | 0.87/0.89/0.91 | 13.5807 |
| direct | 2 | flowsne10 EE+cosine, flow lr = lr/10, projected velocity | +0.0089 | +0.0437 | +0.0822 | +0.0232 | +0.0424 | +0.0665 | +0.325 | 0.003 | – | – |
| direct | 3 | frozen layout s0, std net, lr 0.03 | +0.0046 | +0.0704 | +0.1713 | +0.0056 | +0.0009 | -0.0039 | -0.003 | 0.004 | 0.93/0.94/0.94 | 13.2435 |
| direct | 3 | frozen layout s0, std net, lr 0.003 | +0.0051 | +0.0800 | +0.1761 | +0.0068 | -0.0008 | -0.0117 | +0.004 | 0.004 | 0.86/0.90/0.91 | 13.3558 |
| direct | 3 | frozen layout s0, RFF c = 0.1 | +0.0043 | +0.0721 | +0.1728 | +0.0056 | +0.0048 | +0.0009 | +0.002 | 0.004 | 0.93/0.93/0.94 | 13.2419 |
| direct | 3 | frozen layout s0, RFF c = 0.3 | +0.0046 | +0.0681 | +0.1631 | +0.0060 | +0.0024 | +0.0007 | +0.002 | 0.004 | 0.92/0.93/0.94 | 13.2898 |
| direct | 3 | frozen layout s0, RFF c = 1 | +0.0045 | +0.0666 | +0.1683 | +0.0069 | +0.0101 | +0.0149 | -0.008 | 0.004 | 0.91/0.93/0.94 | 13.3305 |
| kernel | 2 | flowsne10 EE+cosine, flow lr = lr, learned field | +0.0204 | +0.1272 | +0.2785 | +0.0205 | +0.0726 | +0.1950 | +0.472 | 0.007 | 0.90/0.92/0.93 | 13.5498 |
| kernel | 2 | flowsne10 EE+cosine, flow lr = lr, projected velocity | +0.0237 | +0.1042 | +0.1897 | +0.0243 | +0.0545 | +0.1396 | +0.601 | 0.007 | – | – |
| kernel | 2 | flowsne10 EE+cosine, flow lr = lr/10, learned field | +0.0236 | +0.1804 | +0.3470 | +0.0239 | +0.0731 | +0.1072 | +0.424 | 0.008 | 0.86/0.89/0.91 | 13.6772 |
| kernel | 2 | flowsne10 EE+cosine, flow lr = lr/10, projected velocity | +0.0238 | +0.1078 | +0.1869 | +0.0207 | +0.0492 | +0.0753 | +0.621 | 0.008 | – | – |
| kernel | 3 | frozen layout s0, std net, lr 0.03 | +0.0205 | +0.1152 | +0.2357 | +0.0224 | +0.0596 | +0.0866 | +0.527 | 0.005 | 0.92/0.93/0.94 | 13.3067 |
| kernel | 3 | frozen layout s0, std net, lr 0.003 | +0.0200 | +0.1280 | +0.2673 | +0.0219 | +0.0585 | +0.0845 | +0.520 | 0.005 | 0.84/0.89/0.90 | 13.4541 |
| kernel | 3 | frozen layout s0, RFF c = 0.1 | +0.0202 | +0.1143 | +0.2312 | +0.0227 | +0.0583 | +0.0855 | +0.519 | 0.005 | 0.93/0.93/0.94 | 13.2921 |
| kernel | 3 | frozen layout s0, RFF c = 0.3 | +0.0207 | +0.1176 | +0.2371 | +0.0213 | +0.0572 | +0.0829 | +0.545 | 0.005 | 0.92/0.93/0.94 | 13.3437 |
| kernel | 3 | frozen layout s0, RFF c = 1 | +0.0219 | +0.1166 | +0.2327 | +0.0218 | +0.0531 | +0.0745 | +0.520 | 0.005 | 0.91/0.93/0.94 | 13.3845 |
