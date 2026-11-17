"""One-column 2x2 version of vortex_affinity_strip.py for the ICML draft
(section 4, Randers affinity): the vortex data and drift, then the
neighbourhood of one probe point under the symmetric affinity, and the
outgoing row / incoming column of the Randers affinity (Definition 1 defaults:
averaged cost, q = 2, perplexity sqrt(n), rho = 0.95, balanced).

The arrow in each zoom is the affinity-weighted mean offset
sum_j P_ij (x_j - x_i) / sum_j P_ij at its true scale (x ARROW_GAIN = 1, the same
for every panel), not normalised, so a direction-blind neighbourhood gives an
arrow of length ~0. For the incoming column it is the mean offset of the
sources, so it points upstream.

Run: conda run -n dev python experiments/toy/randers_affinity_fig.py
Writes: article/ICML_FlowSNE/figures/randers_affinity.{pdf,png}
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
from flowsne import EntropicRandersAffinity  # noqa: E402
from vortex_affinity_strip import make_vortex  # noqa: E402

OUT = os.path.join(REPO, "article", "ICML_FlowSNE", "figures", "randers_affinity")
HALF = 0.4        # half-width of the zoom window
ARROW_GAIN = 1.0  # display gain of the mean-offset arrows (all panels)
CMAP = "Blues"

plt.rcParams.update({"font.size": 7, "font.family": "serif", "mathtext.fontset": "cm"})


def mean_offset(w, X, i):
    return (w[:, None] * (X - X[i])).sum(0) / w.sum()


def main():
    n = 300
    X, V, theta = make_vortex(n=n, seed=0)
    Xt, Vt = torch.tensor(X), torch.tensor(V)
    P_sym = EntropicRandersAffinity()(Xt).detach().numpy()
    P_fin = EntropicRandersAffinity(omega=Vt)(Xt).detach().numpy()
    i0 = int(np.argmin(np.abs(theta)))
    panels = [(P_sym[i0], "symmetric", r"$P_{i_0\cdot}$"),
              (P_fin[i0], "Randers, out", r"$P_{i_0\cdot}$"),
              (P_fin[:, i0], "Randers, in", r"$P_{\cdot\, i_0}$")]
    vmax = max(np.percentile(w, 99.5) for w, _, _ in panels)
    for w, name, _ in panels:
        m = mean_offset(w, X, i0)
        print(f"{name:18s} mean offset {m.round(4)}  |m|={np.linalg.norm(m):.4f}")

    fig = plt.figure(figsize=(3.25, 3.0))
    W, H, gx, gy, x0, y0 = 0.42, 0.44, 0.025, 0.055, 0.01, 0.005
    pos = [(x0, y0 + H + gy), (x0 + W + gx, y0 + H + gy), (x0, y0), (x0 + W + gx, y0)]
    axs = [fig.add_axes([px, py, W, H]) for px, py in pos]
    cax = fig.add_axes([x0 + 2 * W + gx + 0.015, y0 + 0.02, 0.022, 2 * H + gy - 0.04])

    # (a) data and drift
    ax = axs[0]
    ax.scatter(*X.T, s=3, color="0.6", linewidths=0, zorder=1)
    s = slice(None, None, 3)
    ax.quiver(X[s, 0], X[s, 1], V[s, 0], V[s, 1], color="k", angles="xy", scale_units="xy",
              scale=6.5, width=0.006, headwidth=3.5, headlength=4, headaxislength=3.5, zorder=2)
    ax.add_patch(Rectangle(X[i0] - HALF, 2 * HALF, 2 * HALF, fill=False, ls="--", lw=0.7,
                           color="C3", zorder=3))
    ax.scatter(*X[i0], marker="*", s=45, color="C3", edgecolors="k", linewidths=0.3, zorder=4)
    ax.set_title(r"(a) data, drift $v$", pad=2)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)

    # (b)-(d) zooms on the probe's neighbourhood
    for ax, (w, name, lab), tag in zip(axs[1:], panels, "bcd"):
        sc = ax.scatter(*X.T, c=w, cmap=CMAP, vmin=0, vmax=vmax, s=16, edgecolors="0.55",
                        linewidths=0.3, zorder=2)
        ax.scatter(*X[i0], marker="*", s=70, color="C3", edgecolors="k", linewidths=0.3, zorder=4)
        ax.annotate("", xy=X[i0] + 0.22 * V[i0], xytext=X[i0], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.9, ls="--",
                                    mutation_scale=6))
        m = ARROW_GAIN * mean_offset(w, X, i0)
        ax.annotate("", xy=X[i0] + m, xytext=X[i0], zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.4, mutation_scale=8))
        ax.set_xlim(X[i0, 0] - HALF, X[i0, 0] + HALF)
        ax.set_ylim(X[i0, 1] - HALF, X[i0, 1] + HALF)
        ax.set_title(f"({tag}) {name}: {lab}", pad=2)
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    for sp in axs[0].spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(sc, cax=cax, ticks=[0, vmax])
    cb.ax.set_yticklabels(["0", "max"])
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    fig.savefig(OUT + ".pdf", dpi=300)
    fig.savefig(OUT + ".png", dpi=300)
    print("wrote", OUT + ".{pdf,png}")


if __name__ == "__main__":
    main()
