"""Layouts of every method of Table 2 (seed 0), with each method's drift.

First column: the 3-D data with the true velocity (arrow length = speed;
the Swiss roll's velocity has unit speed).

Rows: sphere vortex (coloured by height), Swiss roll (coloured by the roll
parameter t), noisy cycle in R^50 (first column: its 2-D latent with the clean
velocity; ring coloured by phase, cyclic map; exit branch by arclength, greens). Arrows: the method's drift in flow direction (-omega for the
Randers kernels, the spectral estimate for Debaussart-Joniec et al.),
averaged per grid cell; length relative to the panel's largest cell mean
(the methods' drifts have different units), clipped at its 95th percentile. The symmetric t-SNE panel carries
the spectral estimator's arrows, so it covers both of its table rows.

    conda run -n dev python experiments/baselines/table2_figure.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finsler_embedding_comparison import RES, load_dataset  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "article", "ICML_FlowSNE", "figures", "table2_layouts")
COLS = [("finsler_mds", "Finsler MDS"),
        ("finsler_tsne", "Finsler t-SNE"),
        ("theirs_isomap", "Debaussart-Joniec\net al., Isomap"),
        ("flowsne_ds_joint", "FlowSNE\nbalanced"),
        ("flowsne_rs_joint", "FlowSNE\nrow-stochastic")]
ROWS = [("sphere", "sphere vortex"), ("swiss", "Swiss roll"), ("cycle", "noisy cycle, $\\mathbb{R}^{50}$")]
GRID = 11
VIEW = {"sphere": (48, -60), "swiss": (8, -78)}


def draw_data(ax, X, om, c, cmap, vmin, vmax, view):
    """3-D cloud, faint back points, true-velocity arrows on the front."""
    Xc = (X - X.mean(0)) / np.abs(X - X.mean(0)).max()
    el, az = np.radians(view[0]), np.radians(view[1])
    vdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    depth = Xc @ vdir
    front = depth > np.quantile(depth, 0.35)
    ax.scatter(*Xc[~front].T, c=c[~front], cmap=cmap, vmin=vmin, vmax=vmax, s=0.5,
               alpha=0.2, linewidths=0, depthshade=False, rasterized=True)
    ax.scatter(*Xc[front].T, c=c[front], cmap=cmap, vmin=vmin, vmax=vmax, s=0.8,
               linewidths=0, depthshade=False, rasterized=True)
    idx = np.flatnonzero(depth > np.quantile(depth, 0.5))
    idx = idx[np.random.default_rng(1).choice(len(idx), 45, replace=False)]
    ax.quiver(*Xc[idx].T, *om[idx].T, length=0.22, normalize=False, color="k",
              linewidth=0.45, arrow_length_ratio=0.4)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect([1, 1, 1])
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-0.75, 0.75)
    ax.set_axis_off()
    ax.set_facecolor("none")


def scatter(ax, P, c, cmap, vmin, vmax, branch=None, s=0.8, **kw):
    """Points coloured by c; for the cycle, the exit branch on its own ramp."""
    if branch is None:
        return ax.scatter(*P.T, c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=s, linewidths=0,
                          rasterized=True, **kw)
    ax.scatter(*P[~branch].T, c=c[~branch], cmap="twilight", vmin=0, vmax=1, s=s, linewidths=0,
               rasterized=True, **kw)
    ax.scatter(*P[branch].T, c=c[branch], cmap="Greens", vmin=-0.4, vmax=1, s=s, linewidths=0,
               rasterized=True, **kw)


def draw_latent(ax, D):
    """The cycle's 2-D latent with the clean velocity (arrow length = speed)."""
    U, V = D["U"], D["V_latent"]
    U = (U - U.mean(0)) / np.abs(U - U.mean(0)).max()
    scatter(ax, U, D["color"], None, None, None, D["branch"])
    idx = np.random.default_rng(1).choice(len(U), 70, replace=False)
    ax.quiver(*U[idx].T, *V[idx].T, color="k", angles="xy", scale_units="xy", scale=8,
              width=0.006, headwidth=4, headlength=4, headaxislength=3.5)
    ax.text(0.02, 0.98, "2-D latent;\ndata = smooth lift\nto $\\mathbb{R}^{50}$ + noise", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.2, linespacing=1.1)
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def cell_means(Z, F, g=GRID):
    lo, hi = Z.min(0), Z.max(0)
    ij = np.floor((Z - lo) / (hi - lo).max() * g * (1 - 1e-9)).astype(int)
    keys = ij[:, 0] * (g + 1) + ij[:, 1]
    P, V = [], []
    for k in np.unique(keys):
        m = keys == k
        if m.sum() >= 4:
            P.append(Z[m].mean(0)); V.append(F[m].mean(0))
    return np.array(P), np.array(V)


def main(seed=0):
    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(len(ROWS), len(COLS) + 1, figsize=(6.75, 0.6 + 1.0 * len(ROWS)))
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.005, top=1 - 0.28 / (0.6 + len(ROWS)),
                        wspace=0.04, hspace=0.04)
    for r, (ds, dname) in enumerate(ROWS):
        S = np.load(os.path.join(RES, f"{ds}_seed{seed}.npz"))
        D = load_dataset(ds, seed)
        X, branch = D["X"], D.get("branch")
        if ds == "cycle":
            c, cmap, vmin, vmax = S["color"], None, None, None
        elif ds == "sphere":
            c, cmap, vmin, vmax = X[:, 2], "coolwarm", -1, 1
        else:
            c, cmap = S["color"], "viridis"
            vmin, vmax = c.min(), c.max()
        pos = axes[r, 0].get_position()
        if ds == "cycle":
            draw_latent(axes[r, 0], D)
        else:
            axes[r, 0].remove()
            ax3 = fig.add_axes([pos.x0 - 0.015, pos.y0 - 0.02, pos.width + 0.025,
                                pos.height + 0.04], projection="3d")
            draw_data(ax3, X, D["omega"], c, cmap, vmin, vmax, VIEW[ds])
        fig.text(pos.x0 - 0.008, pos.y0 + pos.height / 2, dname, rotation=90,
                 ha="center", va="center", fontsize=7)
        if r == 0:
            fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.012, "data + velocity\n(length = speed)",
                     ha="center", va="bottom", fontsize=6.5)
        for j, (m, title) in enumerate(COLS, start=1):
            ax = axes[r, j]
            Z, F = S[m], S[m + "_drift"]
            Z = (Z - Z.mean(0)) / np.abs(Z - Z.mean(0)).max()
            scatter(ax, Z, c, cmap, vmin, vmax, branch)
            if np.abs(F).max() > 0:
                P, V = cell_means(Z, F)
                L = np.linalg.norm(V, axis=1)
                q = np.quantile(L, 0.95)
                V = V / np.maximum(L, q)[:, None] * (0.9 * 2 / GRID)
                ax.quiver(P[:, 0], P[:, 1], V[:, 0], V[:, 1], color="black", angles="xy",
                          scale_units="xy", scale=1, width=0.006, headwidth=4, headlength=4,
                          headaxislength=3.5)
            else:
                ax.text(0.97, 0.03, r"$\alpha=0$" + "\nno drift", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6)
            ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.6")
            if r == 0:
                ax.set_title(title, fontsize=6.5, pad=2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
