"""Weighted Hodge decomposition on the ring of the draft's Figure 1
(experiments/toy/vortex_affinity_strip.py: points around the unit
circle, counter-clockwise tangent drift; 120 points instead of 300 so that
the edges stay readable), on its symmetrised kNN graph.

Same layout as hodge_figure.py, plus a top row with the drift itself in R^2
(arrows on the samples) and its ambient split (gradient + divergence-free). The drift is the ring's rotation plus a
constant drift to the right (an exact gradient, phi = c x):
    omega = a tangent + c e_x.
On the ring the rotation is a HARMONIC flow -- divergence-free, no local curl,
circulating around the hole -- so the cyclic part here is the flow around the
hole, not a vortex. Arrows are coloured by
orientation around the ring (counter-clockwise / clockwise / radial). Rows: the edge cost A_ij = (omega_i + omega_j)/2 . (x_j - x_i)
and its two parts; the balanced affinity's reverse net transport P_ji - P_ij;
the row-stochastic one's, minus its drift-free control.

Run:    conda run -n dev python experiments/theory/hodge_figure_ring.py
Writes: article/ICML/figures/hodge_ring.pdf
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "toy"))
sys.path.insert(0, os.path.join(HERE, "..", "single_cell"))

from hodge_figure import BLUE, FIG, INK, INK2, ORANGE, directed_edges, plt, sinkhorn, weighted_hodge
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D  # noqa: E402
from vortex_affinity_strip import make_vortex  # noqa: E402
from gastrulation_hodge import edges_from_knn  # noqa: E402
from gastrulation_repr import knn  # noqa: E402

ROT, SLOPE, K, N_PTS = 0.6, 0.3, 5, 120


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    k_nn = ap.parse_args().k
    X, tangent, theta = make_vortex(n=N_PTS, seed=0)
    n = len(X)
    om_grad = SLOPE * np.array([1.0, 0.0]) + 0 * X                 # grad(SLOPE x)
    om_sol = ROT * tangent                                           # divergence-free in R^2
    om = om_grad + om_sol
    E = edges_from_knn(knn(X, k_nn))
    import scipy.sparse as sp_
    from scipy.sparse.csgraph import connected_components
    ncomp = connected_components(sp_.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(X),) * 2),
                                 directed=False)[0]
    dX = X[E[:, 1]] - X[E[:, 0]]
    le = np.linalg.norm(dX, axis=1)
    h = 1.5 * np.median(le)
    W = np.exp(-le ** 2 / h ** 2)
    A = ((om[E[:, 0]] + om[E[:, 1]]) / 2 * dX).sum(1)
    psi, g, r, B = weighted_hodge(E, A, W, n)

    # W-weighted flux of each part across the ray theta = 0 (x > 0, y = 0): constant for
    # a W-divergence-free flow, i.e. the circulation around the hole
    th = np.arctan2(X[:, 1], X[:, 0])
    cross = (X[E[:, 0], 0] > 0) & (X[E[:, 1], 0] > 0) & (np.sign(th[E[:, 0]]) != np.sign(th[E[:, 1]]))
    up = np.where(th[E[:, 1]] > th[E[:, 0]], 1.0, -1.0)

    def flux(F):
        return float((W * F * up)[cross].sum())
    print(f"{n} nodes, {len(E)} edges, k = {k_nn}, {ncomp} connected component(s)")
    print(f"gradient share (W-weighted) {np.sum(W * g ** 2) / np.sum(W * A ** 2):.3f}")
    print(f"max net W-flow of cyclic part {np.abs(B.T @ (W * r)).max():.1e} (of A: {np.abs(B.T @ (W * A)).max():.1e})")
    print(f"W-flux across theta = 0: A {flux(A):+.3f}, gradient {flux(g):+.3f}, cyclic {flux(r):+.3f}")
    print(f"potential vs x: corr {np.corrcoef(psi, X[:, 0])[0, 1]:+.3f}")

    eps = 0.5 * np.median(le)

    def affinities(F):
        C = np.full((n, n), 1e3)
        C[E[:, 0], E[:, 1]] = le + F
        C[E[:, 1], E[:, 0]] = le - F
        P = sinkhorn(C, eps, iters=30000)
        Kx = np.exp(-(C - C.min(1, keepdims=True)) / eps)
        R = Kx / Kx.sum(1, keepdims=True) / n
        net = lambda M: n * (M[E[:, 1], E[:, 0]] - M[E[:, 0], E[:, 1]])  # reverse net transport, P_ji - P_ij
        return P, net(P), R, net(R)
    (P, N, R, NR), (Pg, Ng, _, NRg), (Pr, Nr, Rr, NRr) = affinities(A), affinities(g), affinities(r)
    NR0 = affinities(0 * A)[3]
    NR, NRg, NRr = NR - NR0, NRg - NR0, NRr - NR0
    print(f"marginal error {np.abs(P.sum(1) * n - 1).max():.1e}, eps {eps:.3f}")
    print(f"balanced: max |P(A) - P(cyc)| / max P = {np.abs(P - Pr).max() / P.max():.1e}, "
          f"gradient-only net transport {np.abs(Ng).max():.1e} (full {np.abs(N).max():.2f})")
    print(f"row-stochastic: max |R(A) - R(cyc)| / max R = {np.abs(R - Rr).max() / R.max():.2f}, "
          f"gradient-only net transport {np.abs(NRg).max():.2f} (full {np.abs(NR).max():.2f}); "
          f"drift-free {np.abs(NR0).max():.2f}")

    # ---- figure: rows = drift in R^2, graph cost, balanced, row-stochastic
    fig = plt.figure(figsize=(6.75, 8.75))
    gs = fig.add_gridspec(4, 3, left=0.06, right=0.998, top=0.978, bottom=0.002, wspace=0.03, hspace=0.12)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(4)])
    fmax = np.abs(A).max()
    nmax = max(np.abs(N).max(), np.abs(NR).max())
    vmax = np.linalg.norm(om, axis=1).max()
    cols = [(om, A, N, NR, "full", r"$\omega$", r"$A_{ij}$"),
            (om_grad, g, Ng, NRg, "gradient part", r"$\nabla(0.3\,x)$", r"$\psi_j-\psi_i$"),
            (om_sol, r, Nr, NRr, "cyclic part", r"$0.6\,t(\theta)$" + "\n(divergence-free)",
             r"$\mathrm{cyc}_W(A)$" + "\nflow around the hole")]
    lo, hi = X.min(0) - 0.04, X.max(0) + 0.04
    radial_grey = "#9a9893"

    def orient_vec(m, d):
        c = (m[0] * d[1] - m[1] * d[0]) / (np.linalg.norm(m) * np.linalg.norm(d) + 1e-300)
        return ORANGE if c > 0.4 else BLUE if c < -0.4 else radial_grey

    def orient(a, b):
        """Orange: anticlockwise around the ring, blue: clockwise, grey: mostly radial."""
        return orient_vec((X[a] + X[b]) / 2, X[b] - X[a])
    grey = LinearSegmentedColormap.from_list("grey", ["#f4f3f0", "#6f6d68"])
    kw = dict(wmax=2.0, head=0.9, shrink=2.0, color=orient)
    for k, (V, F, Nk, NRk, title, lab_amb, lab_cost) in enumerate(cols):
        ax0, ax = axs[0, k], axs[1, k]
        ax0.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1], color=[orient_vec(x, v) for x, v in zip(X, V)],
                   angles="xy", scale_units="xy", scale=vmax / 0.22, width=0.008, headwidth=3.5,
                   headlength=4, headaxislength=3.5, zorder=3)
        ax0.text(0, 0, lab_amb, ha="center", va="center", fontsize=7.5, color=INK)
        ax0.set_title(title, fontsize=8, color=INK, pad=3)
        if k == 1:
            m = ax.scatter(X[:, 0], X[:, 1], s=30, c=psi, cmap=grey, linewidths=0, zorder=0.5)
            cax = ax.inset_axes([0.25, -0.045, 0.5, 0.025])
            cb = fig.colorbar(m, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
            cb.outline.set_visible(False)
            cax.text(-0.04, 0.5, r"$\psi$", transform=cax.transAxes, ha="right", va="center", fontsize=7)
        if k > 0:
            ax.text(0, 0, lab_cost, ha="center", va="center", fontsize=7.5, color=INK)
        directed_edges(ax, X, E, F, fmax, **kw)
        directed_edges(axs[2, k], X, E, Nk, nmax, **kw)
        directed_edges(axs[3, k], X, E, NRk, nmax, **kw)
        for a_ in axs[:, k]:
            a_.scatter(X[:, 0], X[:, 1], s=4, c=INK2, linewidths=0, zorder=2)
            a_.set_xticks([]); a_.set_yticks([]); a_.set_aspect("equal")
            a_.set_xlim(lo[0], hi[0]); a_.set_ylim(lo[1], hi[1])
            for sp_ in a_.spines.values():
                sp_.set_visible(False)
    axs[1, 0].legend(handles=[Line2D([], [], color=ORANGE, lw=1.5, label="anticlockwise"),
                              Line2D([], [], color=BLUE, lw=1.5, label="clockwise"),
                              Line2D([], [], color=radial_grey, lw=1.5, label="radial")],
                     title=r"$A_{ij}$", title_fontsize=7.5, loc="center", fontsize=6, frameon=False,
                     handlelength=1.0, handletextpad=0.4, labelspacing=0.3)
    for a_, lab in zip(axs[:, 0], ["drift in $\\mathbb{R}^2$ (on the samples)",
                                   "graph cost (arrow where $A_{ij}>0$)", "balanced: $P_{ji}-P_{ij}$",
                                   "row-stochastic: $P_{ji}-P_{ij}$\nminus drift-free"]):
        a_.set_ylabel(lab, fontsize=8, color=INK, labelpad=2)
    out = os.path.join(FIG, "hodge_ring.pdf" if k_nn == K else f"hodge_ring_k{k_nn}.pdf")
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
