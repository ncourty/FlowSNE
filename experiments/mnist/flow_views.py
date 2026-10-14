"""Compare three ways of drawing the fitted flow on the MNIST learning-
dynamics frames (dynamics.py), as alternatives to streamlines, which get
cluttered with many points (see flowsne.viz):
  grid arrows       one arrow per occupied cell of a coarse grid (mean field)
  direction colour  points coloured by flow direction (hue) and speed
  divergence        the field's divergence as a background (red = source,
                    blue = sink), with grid arrows on top
Each frame is framed on its own points. Arrow scale is shared across frames:
the 90th percentile of the cell-mean |omega| (pooled) is drawn one cell long,
with a legend; the divergence colour scale is shared too.

Run:    conda run -n dev python experiments/mnist/flow_views.py --dir n1000_lam0.03
Writes: experiments/mnist/dynamics_results/flow_views_<dir>.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from flowsne.viz import (  # noqa: E402
    direction_colors, direction_wheel, divergence_map, field_from_flow, grid_arrows, grid_field,
)

RESULTS = os.path.join(HERE, "dynamics_results")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="n1000_lam0.03")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 2, 10, 20, -1])
    ap.add_argument("--bins", type=int, default=12, help="grid cells across (coarser at small n)")
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--only-divergence", action="store_true",
                    help="only the divergence + grid arrows row, larger panels")
    a = ap.parse_args()

    run = os.path.join(RESULTS, a.dir)
    S = np.load(os.path.join(run, "frames.npz"))
    M = json.load(open(os.path.join(run, "metrics.json")))["metrics"]
    Z, F, lab = S["Z"].astype(float), S["F"].astype(float), S["labels"]
    T = len(Z)
    frames = [f % T for f in a.frames]
    digit = np.array(matplotlib.colormaps["tab10"].colors)[lab]
    fmax = np.linalg.norm(F, axis=2).max()
    fields = {t: field_from_flow(S["flow_params"][t], S["z_std"][t], int(S["flow_hidden_dim"]))
              for t in frames}

    def bounds(t):
        lo, hi = np.percentile(Z[t], [0.5, 99.5], axis=0)
        p = 0.06 * (hi - lo)
        return lo - p, hi + p

    # shared scales: reference arrow = 90th percentile of the cell-mean |omega|
    # pooled over the frames shown (drawn one cell long), and the divergence
    # colour range (first pass without drawing)
    ref = np.percentile(np.concatenate([
        np.linalg.norm(grid_field(Z[t], F[t], bins=a.bins, min_count=a.min_count,
                                  lo=bounds(t)[0], hi=bounds(t)[1])[1], axis=1)
        for t in frames]), 90)
    import matplotlib.pyplot as _plt
    probe = _plt.figure()
    vmax = max(divergence_map(probe.add_subplot(), fields[t], Z[t], lo=bounds(t)[0],
                              hi=bounds(t)[1])[1] for t in frames)
    _plt.close(probe)

    rows = (["divergence + grid arrows"] if a.only_divergence
            else ["grid arrows", "direction colour", "divergence + grid arrows"])
    size = 4.4 if a.only_divergence else 3.3
    fig, axes = plt.subplots(len(rows), len(frames), figsize=(size * len(frames), (size + 0.1) * len(rows)),
                             squeeze=False)
    for c, t in enumerate(frames):
        lo, hi = bounds(t)
        for r, name in enumerate(rows):
            ax = axes[r, c]
            if name == "grid arrows":
                ax.scatter(Z[t, :, 0], Z[t, :, 1], c=digit, s=4, alpha=0.45, linewidths=0)
                grid_arrows(ax, Z[t], F[t], bins=a.bins, min_count=a.min_count, lo=lo, hi=hi,
                            ref=ref, key=(c == len(frames) - 1))
            elif name == "direction colour":
                ax.scatter(Z[t, :, 0], Z[t, :, 1], c=direction_colors(F[t], vmax=fmax), s=6,
                           linewidths=0)
                if c == len(frames) - 1:
                    direction_wheel(ax.inset_axes([0.74, 0.74, 0.24, 0.24]))
            else:
                divergence_map(ax, fields[t], Z[t], lo=lo, hi=hi, vmax=vmax)
                ax.scatter(Z[t, :, 0], Z[t, :, 1], c=digit, s=4, alpha=0.6, linewidths=0, zorder=1)
                grid_arrows(ax, Z[t], F[t], bins=a.bins, min_count=a.min_count, lo=lo, hi=hi,
                            ref=ref, key=(c == len(frames) - 1), zorder=3)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if r == 0:
                ax.set_title(f"iteration {M[t]['step']}  acc {M[t]['acc']:.2f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
    if a.only_divergence:
        fig.suptitle(f"MNIST learning dynamics ({a.dir}): fitted flow -- background: divergence "
                     "(red = source, blue = sink); arrows: mean drift per grid cell; points "
                     "coloured by digit", fontsize=11)
    else:
        fig.suptitle(f"MNIST learning dynamics ({a.dir}): three views of the fitted flow\n"
                     "rows 1 and 3: points coloured by digit; row 3 background: divergence of "
                     "the fitted field (red = source, blue = sink)", fontsize=11)
    out = os.path.join(RESULTS, f"flow_views_{a.dir}{'_divergence' if a.only_divergence else ''}.png")
    fig.savefig(out, dpi=95, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
