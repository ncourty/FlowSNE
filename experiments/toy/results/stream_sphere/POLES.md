# Sphere vortex: where the vortex centres land

See `stream_sphere.py poles`. The true drift e_z x x has exactly two centres, the poles (|z| = 1), and its speed sin(colatitude) vanishes there. Critical points of the fitted field -omega (`fit`) and of the push-forward of the true drift into the same layout (`ref`) are found by the Poincare index on a kernel-smoothed grid (`critical_points`). `centres`: count; `|z|`: kernel-smoothed sphere height at each centre (slowest first; smoothing caps it near 0.95 even exactly on a pole); `speed`: field speed there / median speed. Joint fits.

| input | seed | method | ref centres | ref |z| | fit centres | fit |z| | fit speed | fit saddles+nodes |
|---|---|---|---|---|---|---|---|---|
| ds | 0 | free flow | 2 | 0.94, 0.94 | 2 | 0.94, 0.94 | 0.12, 0.13 | 1 |
| ds | 0 | free flow, 64x2 | 2 | 0.96, 0.95 | 2 | 0.95, 0.95 | 0.11, 0.11 | 1 |
| ds | 0 | stream, Lipschitz h | 1 | 0.97 | 0 | - | - | 0 |
| ds | 0 | stream, global rescale | 3 | 0.90, 0.71, 0.45 | 4 | 0.86, 0.31, 0.35, 0.67 | 0.12, 0.20, 0.21, 0.22 | 3 |
| ds | 0 | free flow, RFF c=0.3 | 2 | 0.89, 0.27 | 2 | 0.88, 0.19 | 0.14, 0.18 | 3 |
| ds | 0 | free flow, RFF c=1 | 3 | 0.96, 0.84, 0.46 | 3 | 0.95, 0.45, 0.78 | 0.10, 0.12, 0.17 | 2 |
| ds | 0 | stream, RFF c=0.3 | 4 | 0.17, 0.93, 0.85, 0.39 | 5 | 0.23, 0.36, 0.49, 0.04, 0.54 | 1.82, 2.03, 4.78, 5.11, 9.01 | 9 |
| ds | 0 | stream, RFF c=1 | 1 | 0.30 | 2 | 0.42, 0.01 | 0.19, 0.20 | 3 |
| ds | 0 | stream 32x1 | 4 | 0.96, 0.96, 0.52, 0.59 | 4 | 0.95, 0.96, 0.32, 0.59 | 0.09, 0.14, 0.22, 0.34 | 1 |
| ds | 0 | stream 32x1, RFF c=0.3 | 4 | 0.93, 0.76, 0.71, 0.11 | 7 | 0.02, 0.56, 0.92, 0.31, 0.55, 0.18, 0.81 | 0.40, 0.44, 1.22, 1.97, 2.73, 3.08, 3.29 | 8 |
| ds | 0 | stream 32x1, RFF c=1 | 1 | 0.01 | 5 | 0.71, 0.73, 0.00, 0.49, 0.00 | 0.05, 0.10, 0.12, 0.12, 0.17 | 4 |
| ds | 1 | free flow | 2 | 0.94, 0.95 | 2 | 0.95, 0.94 | 0.12, 0.13 | 1 |
| ds | 1 | free flow, 64x2 | 4 | 0.96, 0.96, 0.17, 0.04 | 4 | 0.07, 0.96, 0.96, 0.05 | 0.09, 0.10, 0.13, 0.22 | 3 |
| ds | 1 | stream, Lipschitz h | 1 | 0.97 | 0 | - | - | 0 |
| ds | 1 | stream, global rescale | 3 | 0.97, 0.95, 0.26 | 3 | 0.96, 0.81, 0.07 | 0.11, 0.17, 0.24 | 2 |
| ds | 2 | free flow | 2 | 0.95, 0.66 | 2 | 0.94, 0.49 | 0.12, 0.19 | 0 |
| ds | 2 | free flow, 64x2 | 2 | 0.95, 0.48 | 2 | 0.94, 0.35 | 0.10, 0.19 | 1 |
| ds | 2 | stream, Lipschitz h | 2 | 0.97, 0.96 | 0 | - | - | 0 |
| ds | 2 | stream, global rescale | 2 | 0.95, 0.94 | 2 | 0.95, 0.94 | 0.10, 0.10 | 1 |
| rs | 0 | free flow | 2 | 0.95, 0.95 | 2 | 0.95, 0.93 | 0.13, 0.13 | 1 |
| rs | 0 | free flow, 64x2 | 2 | 0.95, 0.88 | 2 | 0.93, 0.62 | 0.13, 0.23 | 0 |
| rs | 0 | stream, Lipschitz h | 1 | 0.96 | 0 | - | - | 0 |
| rs | 0 | stream, global rescale | 3 | 0.96, 0.75, 0.14 | 3 | 0.96, 0.58, 0.03 | 0.06, 0.25, 0.25 | 2 |
| rs | 0 | free flow, RFF c=0.3 | 3 | 0.88, 0.88, 0.07 | 3 | 0.73, 0.70, 0.07 | 0.15, 0.19, 0.22 | 2 |
| rs | 0 | free flow, RFF c=1 | 3 | 0.95, 0.89, 0.48 | 4 | 0.58, 0.95, 0.82, 0.37 | 0.09, 0.11, 0.12, 0.23 | 2 |
| rs | 0 | stream, RFF c=0.3 | 2 | 0.78, 0.46 | 4 | 0.81, 0.79, 0.19, 0.53 | 0.65, 0.98, 2.00, 2.32 | 13 |
| rs | 0 | stream, RFF c=1 | 1 | 0.03 | 3 | 0.73, 0.47, 0.05 | 0.11, 0.13, 0.15 | 2 |
| rs | 0 | stream 32x1 | 4 | 0.96, 0.95, 0.56, 0.70 | 3 | 0.96, 0.35, 0.42 | 0.10, 0.19, 0.38 | 1 |
| rs | 0 | stream 32x1, RFF c=0.3 | 5 | 0.95, 0.70, 0.55, 0.33, 0.02 | 6 | 0.72, 0.58, 0.50, 0.51, 0.95, 0.67 | 0.12, 0.17, 0.83, 1.27, 1.56, 4.59 | 7 |
| rs | 0 | stream 32x1, RFF c=1 | 1 | 0.37 | 4 | 0.75, 0.08, 0.84, 0.03 | 0.06, 0.07, 0.07, 0.09 | 3 |
| rs | 1 | free flow | 3 | 0.96, 0.96, 0.28 | 2 | 0.94, 0.96 | 0.14, 0.16 | 0 |
| rs | 1 | free flow, 64x2 | 4 | 0.96, 0.96, 0.61, 0.63 | 4 | 0.96, 0.94, 0.54, 0.55 | 0.10, 0.13, 0.19, 0.30 | 3 |
| rs | 1 | stream, Lipschitz h | 2 | 0.97, 0.32 | 0 | - | - | 0 |
| rs | 1 | stream, global rescale | 3 | 0.97, 0.75, 0.18 | 3 | 0.96, 0.49, 0.21 | 0.10, 0.28, 0.33 | 1 |
| rs | 2 | free flow | 1 | 0.96 | 1 | 0.96 | 0.11 | 0 |
| rs | 2 | free flow, 64x2 | 2 | 0.95, 0.08 | 3 | 0.94, 0.95, 0.12 | 0.12, 0.12, 0.24 | 4 |
| rs | 2 | stream, Lipschitz h | 1 | 0.97 | 0 | - | - | 0 |
| rs | 2 | stream, global rescale | 2 | 0.95, 0.84 | 2 | 0.95, 0.76 | 0.08, 0.13 | 1 |
