"""Strip figure: FlowSNE (row-stochastic input, joint fit) on the Swiss roll at
increasing input perplexity, seed 0. Layouts from `run-perp` (results/
swiss_seed0_p{P}_rs.npz) plus the default sqrt(n) = 38.7 run from
swiss_seed0.npz. Colour = position along the roll, arrows = mean fitted drift
per grid cell (same drawing as table2_figure.py); each panel's subtitle gives
direction (cos_ref), speed (Spearman) and kNN / kNN-> against the latent.

    conda run -n dev python experiments/baselines/swiss_perplexity_strip.py
Writes article/ICML_FlowSNE/figures/swiss_perplexity.{pdf,png}
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finsler_embedding_comparison import RES  # noqa: E402
from table2_figure import GRID, cell_means, scatter  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "article", "ICML_FlowSNE", "figures", "swiss_perplexity")
PERPS = [10, 20, None, 60, 100, 150]   # None = default sqrt(n) = 38.7


def load(p, seed=0):
    if p is None:
        S = np.load(os.path.join(RES, f"swiss_seed{seed}.npz"))
        J = json.load(open(os.path.join(RES, f"swiss_seed{seed}.json")))
        r = [x for x in J["runs"] if x["method"] == "flowsne_rs_joint"][0]
        sym = [x for x in J["runs"] if x["method"] == "sym_rs"][0]
        r = dict(r, knn_dir=r.get("knn_dir", np.nan))
        return S["flowsne_rs_joint"], S["flowsne_rs_joint_drift"], S["color"], r
    base = os.path.join(RES, f"swiss_seed{seed}_p{p}_rs")
    S = np.load(base + ".npz")
    r = json.load(open(base + ".json"))
    C = np.load(os.path.join(RES, f"swiss_seed{seed}.npz"))["color"]
    return S["Z"], S["drift"], C, r


def main():
    plt.rcParams.update({"font.size": 7})
    n = len(PERPS)
    fig, axes = plt.subplots(1, n, figsize=(6.75, 1.55))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.76, wspace=0.04)
    for ax, p in zip(axes, PERPS):
        Z, F, c, r = load(p)
        Z = (Z - Z.mean(0)) / np.abs(Z - Z.mean(0)).max()
        scatter(ax, Z, c, "viridis", c.min(), c.max())
        P, V = cell_means(Z, F)
        L = np.linalg.norm(V, axis=1)
        q = np.quantile(L, 0.95)
        V = V / np.maximum(L, q)[:, None] * (0.9 * 2 / GRID)
        ax.quiver(P[:, 0], P[:, 1], V[:, 0], V[:, 1], color="black", angles="xy",
                  scale_units="xy", scale=1, width=0.006, headwidth=4, headlength=4,
                  headaxislength=3.5)
        label = r"perp. $\sqrt{n}\approx39$ (default)" if p is None else f"perplexity {p}"
        kd = "" if np.isnan(r.get("knn_dir", np.nan)) else f"  kNN$\\to$ {r['knn_dir']:.2f}"
        ax.set_title(f"{label}\ndir {r['cos_ref']:.2f}  speed {r['speed_spearman']:.2f}\n"
                     f"kNN {r['knn']:.2f}{kd}", fontsize=5.8, pad=2, linespacing=1.15)
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.6")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
