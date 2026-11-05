"""Illustration of the weighted Hodge decomposition (article/ICML, Section 2).

A 2-D point set (jittered grid) with its Delaunay graph and edge weights
W_ij = exp(-|x_i - x_j|^2 / h^2). The drift omega = grad(phi) + J grad(chi)
(a slope plus a vortex) defines the antisymmetric part of the averaged
Randers cost on every edge,
    A_ij = (omega_i + omega_j)/2 . (x_j - x_i),
which is split as in eq. (cyc) into
    gradient  psi_j - psi_i, psi minimising sum_e W_e (A_e - (psi_j - psi_i))^2
    cyclic    cyc_W(A) = A - gradient  (no net W-flow out of any node).
Panels: A; the gradient part over psi; the cyclic part over the circulation
of each triangle (loop sum / area, a discrete curl). Each panel is a directed
graph: every edge oriented by the sign of its flow, line width proportional
to |flow|, one width scale for the three panels.

Bottom row: net transport minus that of the drift-free row-stochastic
affinity, whose rows are normalised by node degree (the balanced one's is 0).

Checks printed: the cyclic part's net W-flow, the gradient part's loop sums,
and that Sinkhorn balancing of the full cost C = |x_j - x_i| + A and of
C - gradient gives the same affinity (the gauge, Theorem 3.1).

Run:    conda run -n dev python experiments/theory/hodge_figure.py
Writes: article/ICML/figures/hodge_split.pdf
"""
import os

import matplotlib
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "article", "ICML", "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, MID = "#2a78d6", "#eb6834", "#f1f0ed"
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#9cc3ee", BLUE])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, MID, ORANGE])


def points(G=10, jitter=0.22, seed=0):
    rng = np.random.default_rng(seed)
    g = (np.arange(G) + 0.5) / G
    X = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    return X + rng.uniform(-jitter, jitter, X.shape) / G


def drift(X):
    """grad(phi) + J grad(chi): phi a curved slope, chi a Gaussian bump."""
    x, y = X[:, 0], X[:, 1]
    grad = 0.6 * np.stack([0.30 + 0.25 * y, 0.25 * x - 0.10], 1)    # phi = 0.6 (0.3x + 0.25xy - 0.1y)
    c, s = np.array([0.56, 0.50]), 0.24
    d = X - c
    e = np.exp(-(d ** 2).sum(1) / s ** 2)
    gchi = -2 * d / s ** 2 * e[:, None] * 0.16
    rot = np.stack([-gchi[:, 1], gchi[:, 0]], 1)                     # J grad chi
    return grad + rot


def edges(tri):
    E = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    return np.unique(np.sort(E, 1), axis=0)


def weighted_hodge(E, A, W, n):
    """psi minimising sum_e W_e (A_e - (psi_j - psi_i))^2; returns psi, grad, cyclic."""
    m = len(E)
    B = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)],
                                                      np.r_[E[:, 0], E[:, 1]])), shape=(m, n))
    L = (B.T @ sp.diags(W) @ B).tolil()
    rhs = B.T @ (W * A)
    L[0, :] = 0; L[0, 0] = 1; rhs[0] = 0                             # fix the additive constant
    psi = spsolve(L.tocsr(), rhs)
    psi -= psi.mean()
    g = B @ psi
    return psi, g, A - g, B


def sinkhorn(C, eps, iters=20000):
    n = len(C)
    K = -C / eps
    f, g = np.zeros(n), np.zeros(n)
    la = np.full(n, -np.log(n))
    for _ in range(iters):
        f = la - np.logaddexp.reduce(K + g[None, :], 1)
        g = la - np.logaddexp.reduce(K + f[:, None], 0)
    return np.exp(K + f[:, None] + g[None, :])


def directed_edges(ax, X, E, F, fmax, wmax=2.2, tiny=0.04, head=1.0, shrink=2.2, color=None):
    """Each edge as an arrow i -> j in the direction of positive flow F_ij,
    line width proportional to |F_ij| / fmax; edges below `tiny` drawn as
    thin undirected lines. `color(a, b)` gives the colour of an arrow a -> b."""
    for (i, j), f in zip(E, F):
        a, b = (i, j) if f >= 0 else (j, i)
        w = abs(f) / fmax
        if w < tiny:
            ax.plot(*X[[a, b]].T, color=INK2, lw=0.3, alpha=0.5, zorder=1, solid_capstyle="round")
            continue
        ax.add_patch(FancyArrowPatch(X[a], X[b], arrowstyle=f"-|>,head_length={head * (1.2 + 2.2 * w):.2f},"
                                     f"head_width={head * (0.6 + 1.2 * w):.2f}", lw=0.25 + wmax * w, color=INK if color is None else color(a, b),
                                     shrinkA=shrink, shrinkB=shrink, zorder=3, joinstyle="miter"))


def main():
    X = points()
    n = len(X)
    D = Delaunay(X)
    tri = D.simplices
    side = np.stack([np.linalg.norm(X[tri[:, a]] - X[tri[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (0, 2))], 1)
    tri = tri[side.max(1) < 1.7 / np.sqrt(n)]                        # drop hull slivers
    e1, e2 = X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    tri[area2 < 0] = tri[area2 < 0][:, [0, 2, 1]]                    # counter-clockwise
    area = np.abs(area2) / 2
    E = edges(tri)
    dX = X[E[:, 1]] - X[E[:, 0]]
    le = np.linalg.norm(dX, axis=1)
    h = 1.5 / np.sqrt(n)
    W = np.exp(-le ** 2 / h ** 2)
    om = drift(X)
    A = ((om[E[:, 0]] + om[E[:, 1]]) / 2 * dX).sum(1)
    psi, g, r, B = weighted_hodge(E, A, W, n)

    # loop sums around triangles (counter-clockwise)
    eid = {tuple(e): k for k, e in enumerate(E)}

    def loop(F, t):
        s = 0.0
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            s += F[eid[(a, b)]] if a < b else -F[eid[(b, a)]]
        return s
    curl_r = np.array([loop(r, t) for t in tri]) / area
    print(f"{n} nodes, {len(E)} edges, {len(tri)} triangles")
    print(f"gradient share (W-weighted) {np.sum(W * g ** 2) / np.sum(W * A ** 2):.3f}")
    print(f"max net W-flow of cyclic part {np.abs(B.T @ (W * r)).max():.1e} "
          f"(of A: {np.abs(B.T @ (W * A)).max():.1e})")
    print(f"max |loop sum| of gradient part {max(abs(loop(g, t)) for t in tri):.1e}")

    # balanced and row-stochastic affinities on the graph for the costs |x_j - x_i| + F_ij
    def affinities(F, eps=0.05):
        C = np.full((n, n), 1e3)                                     # off-graph: exp(-C/eps) = 0
        C[E[:, 0], E[:, 1]] = le + F
        C[E[:, 1], E[:, 0]] = le - F
        P = sinkhorn(C, eps)
        K = np.exp(-(C - C.min(1, keepdims=True)) / eps)
        R = K / K.sum(1, keepdims=True) / n
        net = lambda M: n * (M[E[:, 1], E[:, 0]] - M[E[:, 0], E[:, 1]])  # reverse net transport, P_ji - P_ij
        return P, net(P), R, net(R)
    (P, N, R, NR), (Pg, Ng, Rg, NRg), (Pr, Nr, Rr, NRr) = affinities(A), affinities(g), affinities(r)
    print(f"marginal error {np.abs(P.sum(1) * n - 1).max():.1e}")
    NR0 = affinities(0 * A)[3]                                       # drift-free control (degree effect)
    print(f"drift-free row-stochastic net transport max {np.abs(NR0).max():.2f}")
    NR, NRg, NRr = NR - NR0, NRg - NR0, NRr - NR0
    print(f"balanced: max |P(A) - P(cyc)| / max P = {np.abs(P - Pr).max() / P.max():.1e}, "
          f"gradient-only net transport max {np.abs(Ng).max():.1e} (full: {np.abs(N).max():.1e})")

    print(f"row-stochastic: max |R(A) - R(cyc)| / max R = {np.abs(R - Rr).max() / R.max():.2f}, "
          f"gradient-only net transport (minus drift-free) max {np.abs(NRg).max():.2f} "
          f"(full: {np.abs(NR).max():.2f})")

    # ---- figure: rows = cost, balanced net transport, row-stochastic net transport
    fig = plt.figure(figsize=(6.75, 6.55))
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.998, top=0.97, bottom=0.002, wspace=0.03, hspace=0.1)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)])
    fmax = np.abs(A).max()                                            # cost row: one width scale
    nmax = max(np.abs(N).max(), np.abs(NR).max())                     # transport rows: one width scale
    lim = np.quantile(np.abs(curl_r), 0.98)                           # boundary slivers
    cols = [(A, N, NR, r"antisymmetric cost $A_{ij}$", None),
            (g, Ng, NRg, r"gradient part $\psi_j-\psi_i$", "psi"),
            (r, Nr, NRr, r"cyclic part $\mathrm{cyc}_W(A)$", "curl")]
    lo, hi = X.min(0) - 0.015, X.max(0) + 0.015
    for k, (F, Nk, NRk, title, bg) in enumerate(cols):
        ax = axs[0, k]
        if bg == "psi":
            m = ax.tripcolor(X[:, 0], X[:, 1], tri, psi, cmap=SEQ, shading="gouraud", rasterized=True)
            label = r"$\psi$"
        elif bg == "curl":
            m = ax.tripcolor(X[:, 0], X[:, 1], tri, facecolors=curl_r, cmap=DIV, vmin=-lim, vmax=lim,
                             edgecolors="none", rasterized=True)
            label = "curl"
        if bg is not None:
            cax = ax.inset_axes([0.25, -0.045, 0.5, 0.025])
            cb = fig.colorbar(m, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
            cb.outline.set_visible(False)
            t = 0.1 if bg == "psi" else 6
            cb.set_ticks([-t, 0, t])
            cax.text(-0.04, 0.5, label, transform=cax.transAxes, ha="right", va="center", fontsize=7)
        else:
            ax.triplot(X[:, 0], X[:, 1], tri, color=GRID, lw=0.4, zorder=1)
        directed_edges(ax, X, E, F, fmax)
        directed_edges(axs[1, k], X, E, Nk, nmax)
        directed_edges(axs[2, k], X, E, NRk, nmax)
        ax.set_title(title, fontsize=8, color=INK, pad=3)
        for a_ in axs[:, k]:
            a_.scatter(X[:, 0], X[:, 1], s=4, c=INK2, linewidths=0, zorder=2)
            a_.set_xticks([]); a_.set_yticks([]); a_.set_aspect("equal")
            a_.set_xlim(lo[0], hi[0]); a_.set_ylim(lo[1], hi[1])
            for sp_ in a_.spines.values():
                sp_.set_visible(False)
    for a_, lab in zip(axs[:, 0], ["cost (arrow where $A_{ij}>0$)",
                                   "balanced: $P_{ji}-P_{ij}$",
                                   "row-stochastic: $P_{ji}-P_{ij}$\nminus drift-free"]):
        a_.set_ylabel(lab, fontsize=8, color=INK, labelpad=2)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "hodge_split.pdf")
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
