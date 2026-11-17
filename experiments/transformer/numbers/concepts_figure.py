"""Compact Section 6.3 figure: GPT-2 small, FlowSNE (row-stochastic input).

Row 1: months in 60 contexts, the full layout at layers 2, 6, 10, coloured by
month, the context of row 2 circled. Row 2: one context ("We started work in")
picked out of the same layouts, with its drift at each month. Row 3: integers
0..359 (template means) at layers 0, 4, 8, coloured by n -- the contrast,
where the successor drift runs along a line.

    conda run -n dev python experiments/transformer/numbers/concepts_figure.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "baselines"))
from table2_figure import cell_means  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from flowsne.viz import field_from_flow  # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "..", "..", "article", "ICML_FlowSNE", "figures", "transformer_concepts")
METHOD = "flowsne_rs"
CTX = 3
MONTH_LAYERS = (2, 6, 10)
NUMBER_LAYERS = (0, 4, 8)
CMAP_M = "hsv"


def norm(Z):
    Z = Z - Z.mean(0)
    return Z / np.abs(Z).max()


def norm_map(Z):
    """the affine map of ``norm``: (shift, scale), so z_norm = (z - shift) / scale"""
    mu = Z.mean(0)
    return mu, np.abs(Z - mu).max()


def field_grid(S, Z_raw, n=33, reach=0.09):
    """the fitted flow (flow direction) on an n x n grid of the normalised panel,
    masked farther than ``reach`` from every point (the network extrapolates there)"""
    from scipy.spatial import cKDTree
    field = field_from_flow(S[METHOD + "_flow"], float(S[METHOD + "_zstd"]), hidden_dim=32)
    mu, sc = norm_map(Z_raw)
    g = np.linspace(-1.05, 1.05, n)
    GX, GY = np.meshgrid(g, g)
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    U = field(G * sc + mu)
    far = cKDTree((Z_raw - mu) / sc).query(G)[0] > reach
    U[far] = np.nan
    return GX, GY, U[:, 0].reshape(n, n), U[:, 1].reshape(n, n)


def grid_arrows(ax, Z, F, grid=9, color="k"):
    P, V = cell_means(Z, F, grid)
    L = np.linalg.norm(V, axis=1)
    q = np.quantile(L, 0.95)
    V = V / np.maximum(L, q)[:, None] * (0.85 * 2 / grid)
    ax.quiver(P[:, 0], P[:, 1], V[:, 0], V[:, 1], color=color, angles="xy", scale_units="xy",
              scale=1, width=0.008, headwidth=3.5, headlength=3.5, headaxislength=3)


def style(ax, lim=1.08):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("0.6")


def main(row1="points"):
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5})
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3, 3, figsize=(3.25, 3.95))
    for j, l in enumerate(MONTH_LAYERS):
        S = np.load(os.path.join(RES, "cyclic", f"months_layer{l}_{METHOD}_seed0.npz"))
        Z, F = norm(S[METHOD]), S[METHOD + "_drift"]
        c, ctx, names = S["concept"], S["context"], S["names"]
        # row 1: full layout
        ax = axes[0, j]
        if row1 == "points":
            ax.scatter(*Z.T, c=c, cmap=CMAP_M, vmin=0, vmax=12, s=0.9, linewidths=0,
                       rasterized=True)
        else:
            ax.scatter(*Z.T, c=c, cmap=CMAP_M, vmin=0, vmax=12, s=0.9, linewidths=0,
                       rasterized=True, alpha=0.45, zorder=1)
            GX, GY, U, V = field_grid(S, S[METHOD], n=27 if row1 == "quiver" else 41)
            if row1 == "quiver":     # arrow length = |omega|, one scale for all panels
                ax.quiver(GX, GY, U, V, angles="xy", scale_units="xy", scale=1 / 0.2,
                          width=0.005, headwidth=3.5, headlength=3.5, headaxislength=3,
                          color="k", zorder=3)
            else:                    # streamlines, width = |omega| (max 0.95)
                spd = np.hypot(U, V)
                ax.streamplot(GX, GY, U, V, density=1.9, color="k",
                              linewidth=np.nan_to_num(0.1 + 0.7 * spd / 0.95), arrowsize=0.3,
                              broken_streamlines=True, zorder=3)
        mk = ctx == CTX
        ax.scatter(*Z[mk].T, s=5, facecolors="none", edgecolors="k", linewidths=0.4)
        style(ax)
        ax.set_title(f"layer {l}", pad=2)
        # row 2: one context
        ax = axes[1, j]
        Zk, Fk = Z[mk], F[mk]
        o = np.argsort(c[mk]); Zk, Fk = Zk[o], Fk[o]
        ctr = Zk.mean(0); half = 0.62 * np.ptp(Zk, 0).max()
        Zk = (Zk - ctr) / half
        ax.scatter(*((Z - ctr) / half).T, c="0.88", s=0.5, linewidths=0, rasterized=True)
        ax.plot(*np.vstack([Zk, Zk[:1]]).T, color="0.6", lw=0.5, zorder=1)
        ax.scatter(*Zk.T, c=np.arange(12), cmap=CMAP_M, vmin=0, vmax=12, s=9, linewidths=0, zorder=3)
        Fu = Fk / np.linalg.norm(Fk, axis=1, keepdims=True)
        ax.quiver(*Zk.T, *Fu.T, angles="xy", scale_units="xy", scale=1 / 0.22, width=0.008,
                  headwidth=3.5, headlength=3.5, headaxislength=3, zorder=4)
        for k in (0, 3, 6, 9):
            ax.annotate(str(names[k])[:3], Zk[k], xytext=(0, 3.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.5, zorder=5)
        style(ax, 1.0)
        # row 3: numbers
        l3 = NUMBER_LAYERS[j]
        T = np.load(os.path.join(RES, f"layer{l3}_seed0.npz"))
        Zn, Fn, num = norm(T[METHOD]), T[METHOD + "_drift"], T["numbers"]
        ax = axes[2, j]
        ax.plot(*Zn.T, color="0.75", lw=0.25, zorder=0)
        sc = ax.scatter(*Zn.T, c=num, cmap="viridis", s=1.2, linewidths=0, rasterized=True)
        grid_arrows(ax, Zn, Fn)
        style(ax)
        ax.set_title(f"layer {l3}", pad=2)
    top, bot = 0.96, 0.15
    fig.subplots_adjust(left=0.075, right=0.995, bottom=bot, top=top, wspace=0.04, hspace=0.14)
    for r, lab in enumerate(("months, all 60 contexts", "one context:\n“We started work in …”",
                             "integers 0–359")):
        pos = axes[r, 0].get_position()
        fig.text(0.033, (pos.y0 + pos.y1) / 2, lab, rotation=90, ha="center", va="center",
                 linespacing=1.1)
    # legend band
    handles = [
        Line2D([], [], color="0.6", lw=0.8,
               label="path through consecutive items\n(Jan→Feb→…→Dec→Jan; n→n+1)"),
        Line2D([], [], color="k", marker=r"$\rightarrow$", ls="none", markersize=7,
               label="fitted drift (flow direction)"),
        Line2D([], [], color="k", marker="o", mfc="none", ls="none", markersize=3, mew=0.5,
               label="points of the middle row's context"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.01), frameon=False,
               fontsize=5.5, handlelength=1.6, borderaxespad=0.2, labelspacing=0.35)
    axm = fig.add_axes([0.70, 0.098, 0.28, 0.02])          # month colour key
    axm.imshow(np.arange(12)[None], cmap=CMAP_M, vmin=0, vmax=12, aspect="auto")
    axm.set_xticks(range(12)); axm.set_xticklabels(list("JFMAMJJASOND"), fontsize=5)
    axm.tick_params(length=0, pad=1); axm.set_yticks([])
    axm.set_title("colour: month", fontsize=5.5, pad=1.5)
    axn = fig.add_axes([0.70, 0.035, 0.28, 0.013])          # n colour key
    cb = fig.colorbar(sc, cax=axn, orientation="horizontal", ticks=[0, 180, 359])
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
    axn.set_title("colour: n", fontsize=5.5, pad=1.5)
    out = OUT + ("" if row1 == "points" else f"_{row1}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print("wrote", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--row1", default="points", choices=["points", "quiver", "stream"])
    main(ap.parse_args().row1)
