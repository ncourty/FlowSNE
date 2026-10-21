"""How much of the RNA velocity is a gradient (a potential / pseudotime
tilt) and how much is rotational -- graph Hodge decomposition on the cells'
kNN graph (the HodgeRank construction, Jiang, Lim, Yao & Ye 2011).

Why it matters for FlowSNE: if the drift is a gradient, omega = grad(phi),
the Randers distance is exact, ||y - x|| + phi(y) - phi(x), and with a Gibbs
kernel and a global bandwidth that term factorises: Sinkhorn absorbs it
entirely (balanced affinity: no effect), row normalisation turns it into a
reweighting of the targets by exp(-phi(y)/eps) (row-stochastic: a landscape
tilt, not a direction). Only the non-gradient part is genuinely
path-dependent (Finsler) content.

The velocity becomes an edge flow on the symmetrised kNN graph -- the
averaged drift term used by bdF, f_ij = (v_i + v_j)/2 . (x_j - x_i) --
split as f = grad(phi) + r, phi by least squares (graph Laplacian solve),
r = curl + harmonic parts (not separated here). Reported, for the direct
velocity projection and for scVelo's transition-matrix projection:

  mean       share of the velocity's mean squared norm carried by the one vector
             common to all cells (a constant field is an exact gradient, phi = v.x;
             each velocity is also decomposed with that mean removed, "_centred")
  share      ||grad(phi)||^2 / ||f||^2 over edges, and the same for the centred
             velocities permuted among cells (null: any edge flow on a graph has
             some gradient part)
  phi        Spearman correlation of the potential with the embryonic stage
  time       correlation of each part's edge values with the stage difference
             along the edge (does that part point forward in time?)
  coherence  each part turned back into one vector per cell (least squares over
             the cell's edges); median over cells of |mean of neighbours'
             vectors| / mean of their norms (1 = smooth field, ~1/sqrt(k) noise)
  walk       gastrulation_repr.py's cosine walk in the 30-PC space with each
             part's vectors: days gained after s steps over the direction-blind
             walk (k = 30)
  by type    gradient share on the edges inside each cell type

Run (GPU server, needs the cached full preprocessing):
    python experiments/single_cell/gastrulation_hodge.py --subset full --n-cells 30000
Writes: experiments/single_cell/gastrulation_results/<tag>/{hodge.json, HODGE.md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from gastrulation_data import load  # noqa: E402
from gastrulation_markov import advances  # noqa: E402
from gastrulation_repr import chain, knn  # noqa: E402


def edges_from_knn(idx):
    """Undirected edges (i < j) of the symmetrised kNN graph, (m, 2)."""
    n, k = idx.shape
    e = np.stack([np.repeat(np.arange(n), k), idx.ravel()], 1)
    return np.unique(np.sort(e, 1), axis=0)


def incidence(E, n):
    m = len(E)
    return sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)],
                                                          np.r_[E[:, 0], E[:, 1]])), shape=(m, n))


def edge_flow(X, V, E):
    i, j = E[:, 0], E[:, 1]
    return 0.5 * ((V[i] + V[j]) * (X[j] - X[i])).sum(1)


def hodge(B, f):
    """phi = argmin ||B phi - f||, by CG on the graph Laplacian (+ tiny ridge for
    its constant null space). Returns phi, the gradient part and the rest."""
    L = (B.T @ B + 1e-8 * sp.eye(B.shape[1])).tocsr()
    phi, info = cg(L, B.T @ f, rtol=1e-10, maxiter=20000) if "rtol" in cg.__code__.co_varnames \
        else cg(L, B.T @ f, tol=1e-10, maxiter=20000)
    g = B @ phi
    return phi - phi.mean(), g, f - g


def cell_vectors(X, E, values, ridge=1e-3):
    """One vector per cell from edge values: argmin_u sum_j (u . (x_j - x_i) - f_ij)^2
    over the cell's edges (both orientations), ridge-regularised."""
    n, d = X.shape
    src = np.r_[E[:, 0], E[:, 1]]
    dst = np.r_[E[:, 1], E[:, 0]]
    val = np.r_[values, -values]
    D = X[dst] - X[src]
    A = np.zeros((n, d, d))
    b = np.zeros((n, d))
    np.add.at(A, src, D[:, :, None] * D[:, None, :])
    np.add.at(b, src, D * val[:, None])
    lam = ridge * np.trace(A, axis1=1, axis2=2)[:, None, None] / d
    return np.linalg.solve(A + lam * np.eye(d), b[..., None])[..., 0]


def coherence(U, idx):
    mean_vec = np.linalg.norm(U[idx].mean(1), axis=1)
    mean_norm = np.linalg.norm(U[idx], axis=2).mean(1)
    return float(np.median(mean_vec / np.maximum(mean_norm, 1e-30)))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["full", "e75", "erythroid"], default="full")
    ap.add_argument("--n-cells", type=int, default=30000)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 25])
    ap.add_argument("--n-null", type=int, default=3)
    ap.add_argument("--tag", default="hodge")
    a = ap.parse_args()
    res = os.path.join(HERE, "gastrulation_results", a.tag)
    os.makedirs(res, exist_ok=True)

    d = load(a.subset, n_cells=a.n_cells, seed=0)
    ok = ~np.isnan(d["stage"])
    X = d["X_pca"][ok].astype(np.float64)
    stage, celltype = d["stage"][ok].astype(np.float64), d["celltype"][ok]
    vels = {"direct": d["V_pca_direct"][ok].astype(np.float64)[:, :X.shape[1]],
            "scvelo_T": d["V_pca"][ok].astype(np.float64)}
    vels.update({f"{k}_centred": v - v.mean(0) for k, v in list(vels.items())})
    n = len(X)
    idx = knn(X, a.k)
    E = edges_from_knn(idx)
    B = incidence(E, n)
    dstage = stage[E[:, 1]] - stage[E[:, 0]]
    print(f"{a.subset}: {n} cells, {len(E)} edges (k = {a.k})", flush=True)

    def walk(F):
        adv = advances(chain(X, F, a.k), stage, a.steps)
        adv0 = advances(chain(X, F, a.k, directed=False), stage, a.steps)
        return {f"s{s}": float((adv[s] - adv0[s]).mean()) for s in a.steps}

    out = dict(args=vars(a), n=n, m=len(E), velocities={})
    rng = np.random.default_rng(0)
    for name, V in vels.items():
        mean_share = float((V.mean(0) ** 2).sum() / (V ** 2).sum(1).mean())
        f = edge_flow(X, V, E)
        phi, g, r = hodge(B, f)
        nulls = []
        for _ in range(a.n_null):
            fn = edge_flow(X, (V - V.mean(0))[rng.permutation(n)], E)
            _, gn, _ = hodge(B, fn)
            nulls.append(float((gn ** 2).sum() / (fn ** 2).sum()))
        parts = {"full": f, "gradient": g, "rest": r}
        U = {p: cell_vectors(X, E, v) for p, v in parts.items()}
        within = {}
        for c in np.unique(celltype):
            sel = (celltype[E[:, 0]] == c) & (celltype[E[:, 1]] == c)
            if sel.sum() >= 50:
                within[c] = float((g[sel] ** 2).sum() / max((f[sel] ** 2).sum(), 1e-300))
        o = dict(mean_share=mean_share, gradient_share=float((g ** 2).sum() / (f ** 2).sum()),
                 gradient_share_null=nulls,
                 phi_stage_spearman=float(spearmanr(phi, stage)[0]),
                 time_corr={p: corr(v, dstage) for p, v in parts.items()},
                 coherence={p: coherence(u, idx) for p, u in U.items()},
                 coherence_raw_velocity=coherence(V, idx),
                 walk={p: walk(u) for p, u in U.items()},
                 walk_raw_velocity=walk(V),
                 gradient_share_by_celltype=within)
        out["velocities"][name] = o
        print(f"  {name}: gradient share {o['gradient_share']:.3f} (null "
              f"{np.mean(nulls):.3f}); phi~stage rho {o['phi_stage_spearman']:+.3f}; time corr "
              + ", ".join(f"{p} {v:+.3f}" for p, v in o["time_corr"].items())
              + "; walk s25 " + ", ".join(f"{p} {v[f's{a.steps[-1]}']:+.3f}" for p, v in o["walk"].items()),
              flush=True)
        json.dump(out, open(os.path.join(res, "hodge.json"), "w"), indent=1)
    write_report(out, res)


def write_report(out, res):
    steps = out["args"]["steps"]
    L = [f"# Hodge decomposition of the velocity ({out['args']['subset']}, n = {out['n']}, "
         f"{out['m']} edges, k = {out['args']['k']})", "",
         "Edge flow f_ij = ½(v_i + v_j)·(x_j − x_i) on the symmetrised kNN graph in 30-PC space, "
         "split as f = grad φ + rest (curl + harmonic). Mean = share of the velocity carried by the "
         "vector common to all cells (a constant field is an exact gradient); `_centred` = that mean "
         "removed. Null = centred velocities permuted among cells.", "",
         "| velocity | mean | gradient share | null | φ vs stage (Spearman) | time corr: full / gradient / rest | "
         "coherence: raw / full / gradient / rest |", "|---|---|---|---|---|---|---|"]
    for nm, o in out["velocities"].items():
        tc, co = o["time_corr"], o["coherence"]
        L.append(f"| {nm} | {o['mean_share']:.2f} | {o['gradient_share']:.3f} | {np.mean(o['gradient_share_null']):.3f} | "
                 f"{o['phi_stage_spearman']:+.3f} | {tc['full']:+.3f} / {tc['gradient']:+.3f} / "
                 f"{tc['rest']:+.3f} | {o['coherence_raw_velocity']:.2f} / {co['full']:.2f} / "
                 f"{co['gradient']:.2f} / {co['rest']:.2f} |")
    L += ["", "## Walk: days gained over the direction-blind walk (30-PC space, cosine rule)", "",
          "| velocity | part | " + " | ".join(f"s={s}" for s in steps) + " |",
          "|---|---|" + "---|" * len(steps)]
    for nm, o in out["velocities"].items():
        rows = {"raw velocity": o["walk_raw_velocity"], **o["walk"]}
        for p, w in rows.items():
            L.append(f"| {nm} | {p} | " + " | ".join(f"{w[f's{s}']:+.3f}" for s in steps) + " |")
    L += ["", "## Gradient share on edges inside each cell type", "",
          "| cell type | " + " | ".join(out["velocities"]) + " |", "|---|" + "---|" * len(out["velocities"])]
    types = sorted({c for o in out["velocities"].values() for c in o["gradient_share_by_celltype"]},
                   key=lambda c: -out["velocities"]["direct"]["gradient_share_by_celltype"].get(c, 0))
    for c in types:
        L.append(f"| {c} | " + " | ".join(
            f"{o['gradient_share_by_celltype'][c]:.2f}" if c in o["gradient_share_by_celltype"] else "–"
            for o in out["velocities"].values()) + " |")
    open(os.path.join(res, "HODGE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "HODGE.md"))


if __name__ == "__main__":
    main()
